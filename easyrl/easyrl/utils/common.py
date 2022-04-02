import importlib
import json
import numbers
import pickle as pkl
import random
import shutil
from pathlib import Path

import cv2
import git
import numpy as np
import torch
import yaml

from easyrl.utils.rl_logger import logger


def get_all_subdirs(directory):
    directory = pathlib_file(directory)
    folders = list(directory.iterdir())
    folders = [x for x in folders if x.is_dir()]
    return folders


def get_all_files_with_suffix(directory, suffix):
    directory = pathlib_file(directory)
    if not suffix.startswith('.'):
        suffix = '.' + suffix
    files = directory.glob(f'**/*{suffix}')
    files = [x for x in files if x.is_file() and x.suffix == suffix]
    return files


def set_random_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)


def chunker_list(seq_list, nchunks):
    # split the list into n parts/chunks
    return [seq_list[i::nchunks] for i in range(nchunks)]


def module_available(module_path: str) -> bool:
    """Testing if given module is avalaible in your env.

    Copied from https://github.com/PyTorchLightning/pytorch-lightning/blob/master/pytorch_lightning/utilities/__init__.py.

    >>> module_available('os')
    True
    >>> module_available('bla.bla')
    False
    """
    try:
        mods = module_path.split('.')
        assert mods, 'nothing given to test'
        # it has to be tested as per partets
        for i in range(len(mods)):
            module_path = '.'.join(mods[:i + 1])
            if importlib.util.find_spec(module_path) is None:
                return False
        return True
    except AttributeError:
        return False


def check_if_run_distributed(cfg):
    from easyrl import HOROVOD_AVAILABLE
    if HOROVOD_AVAILABLE:
        import horovod.torch as hvd
        hvd.init()
        if hvd.size() > 1:
            cfg.distributed = True
    if cfg.distributed and not HOROVOD_AVAILABLE:
        logger.error('Horovod is not installed! Will not run in distributed training')
    distributed = HOROVOD_AVAILABLE and cfg.distributed
    cfg.distributed = distributed
    if distributed:
        gpu_id = hvd.local_rank() + cfg.gpu_shift
        if cfg.gpus is not None:
            gpu_id = cfg.gpus[gpu_id]
        logger.info(f'Rank {hvd.local_rank()} GPU ID: {gpu_id}')
        torch.cuda.set_device(gpu_id)
        logger.info(f'Using Horovod for distributed training, number of processes:{hvd.size()}')
    return distributed


def is_dist_and_root_rank(cfg):
    if cfg.distributed:
        import horovod.torch as hvd
        if hvd.rank() == 0:
            return True
    return False


def is_dist_not_root_rank(cfg):
    if cfg.distributed:
        import horovod.torch as hvd
        if hvd.rank() != 0:
            return True
    return False


def get_horovod_size(cfg):
    if cfg.distributed:
        import horovod.torch as hvd
        return hvd.size()
    return 0


def list_to_numpy(data, expand_dims=None):
    if isinstance(data, numbers.Number):
        data = np.array([data])
    else:
        data = np.array(data)
    if expand_dims is not None:
        data = np.expand_dims(data, axis=expand_dims)
    return data


def save_traj(traj, save_dir):
    save_dir = pathlib_file(save_dir)
    if not save_dir.exists():
        Path.mkdir(save_dir, parents=True)
    save_ob = traj[0].ob is not None
    save_state = traj[0].state is not None
    if save_ob:
        ob_is_state = len(traj[0].ob[0].shape) <= 1
    infos = traj.infos
    action_infos = traj.action_infos
    actions = traj.actions
    tsps = traj.steps_til_done.copy().tolist()
    sub_dirs = sorted([x for x in save_dir.iterdir() if x.is_dir()])
    folder_idx = len(sub_dirs)
    for ei in range(traj.num_envs):
        ei_save_dir = save_dir.joinpath(f'{folder_idx:06d}')
        ei_render_imgs = []
        concise_info = {}
        for t in range(tsps[ei]):
            if 'render_image' in infos[t][ei]:
                img_t = infos[t][ei]['render_image']
                ei_render_imgs.append(img_t)
            c_info = {}
            for k, v in infos[t][ei].items():
                if k != 'render_image':
                    if isinstance(v, (np.generic, np.ndarray)):
                        v = v.tolist()
                    c_info[k] = v
            for k, v in action_infos[t].items():
                if isinstance(v, (np.generic, np.ndarray)):
                    v = v[ei].tolist()
                c_info[k] = v
            concise_info[t] = c_info
        if 'success' in concise_info[t]:
            ei_save_dir = ei_save_dir.parent.joinpath(f'{folder_idx:06d}_success_{concise_info[t]["success"]}')
        if len(ei_render_imgs) > 1:
            img_folder = ei_save_dir.joinpath('render_imgs')
            save_images(ei_render_imgs, img_folder)
            video_file = ei_save_dir.joinpath('render_video.mp4')
            convert_imgs_to_video(ei_render_imgs, video_file.as_posix())
        if save_ob:
            if ob_is_state:
                ob_file = ei_save_dir.joinpath('obs.json')
                save_to_json(traj.obs[:tsps[ei], ei].tolist(),
                             ob_file)
            else:
                ob_folder = ei_save_dir.joinpath('obs')
                save_images(traj.obs[:tsps[ei], ei], ob_folder)
        action_file = ei_save_dir.joinpath('actions.json')
        save_to_json(actions[:tsps[ei], ei].tolist(),
                     action_file)

        info_file = ei_save_dir.joinpath('info.json')
        save_to_json(concise_info, info_file)

        if save_state:
            state_file = ei_save_dir.joinpath('states.json')
            save_to_json(traj.states[:tsps[ei], ei].tolist(),
                         state_file)
        folder_idx += 1


def save_images(images, save_dir):
    save_dir = pathlib_file(save_dir)
    if save_dir.exists():
        shutil.rmtree(save_dir, ignore_errors=True)
    Path.mkdir(save_dir, parents=True)
    for i in range(len(images)):
        img = images[i]
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img_file_name = save_dir.joinpath('{:06d}.png'.format(i))
        cv2.imwrite(img_file_name.as_posix(), img)


def convert_imgs_to_video(images, video_file, fps=20):
    height = images[0].shape[0]
    width = images[0].shape[1]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(video_file, fourcc, fps, (width, height))
    for image in images:
        out.write(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    out.release()


def save_to_json(data, file_name):
    file_name = pathlib_file(file_name)
    if not file_name.parent.exists():
        Path.mkdir(file_name.parent, parents=True)
    with file_name.open('w') as f:
        json.dump(data, f, indent=2)


def load_from_json(file_name):
    file_name = pathlib_file(file_name)
    with file_name.open('r') as f:
        data = json.load(f)
    return data


def load_from_yaml(file_name):
    file_name = pathlib_file(file_name)
    with file_name.open('r') as f:
        data = yaml.load(f, Loader=yaml.FullLoader)
    return data


def save_to_yaml(data, file_name):
    file_name = pathlib_file(file_name)
    if not file_name.parent.exists():
        Path.mkdir(file_name.parent, parents=True)
    with file_name.open('w') as f:
        yaml.dump(data, f, default_flow_style=False)


def save_to_pickle(data, file_name):
    file_name = pathlib_file(file_name)
    if not file_name.parent.exists():
        Path.mkdir(file_name.parent, parents=True)
    with file_name.open('wb') as f:
        pkl.dump(data, f, protocol=pkl.HIGHEST_PROTOCOL)


def load_from_pickle(file_name):
    file_name = pathlib_file(file_name)
    with file_name.open('rb') as f:
        data = pkl.load(f)
    return data


def pathlib_file(file_name):
    if isinstance(file_name, str):
        file_name = Path(file_name)
    elif not isinstance(file_name, Path):
        raise TypeError(f'Please check the type of '
                        f'the filename:{file_name}')
    return file_name


def tile_images(img_nhwc):
    """
    Tile N images into one big PxQ image
    (P,Q) are chosen to be as close as possible, and if N
    is square, then P=Q.

    input: img_nhwc, list or array of images, ndim=4 once turned into array
        n = batch index, h = height, w = width, c = channel
    returns:
        bigim_HWc, ndarray with ndim=3
    """
    img_nhwc = np.asarray(img_nhwc)
    N, h, w, c = img_nhwc.shape
    H = int(np.ceil(np.sqrt(N)))
    W = int(np.ceil(float(N) / H))
    img_nhwc = np.array(list(img_nhwc) + [img_nhwc[0] * 0 for _ in range(N, H * W)])
    img_HWhwc = img_nhwc.reshape(H, W, h, w, c)
    img_HhWwc = img_HWhwc.transpose(0, 2, 1, 3, 4)
    img_Hh_Ww_c = img_HhWwc.reshape(H * h, W * w, c)
    return img_Hh_Ww_c


def linear_decay_percent(epoch, total_epochs):
    return 1 - epoch / float(total_epochs)


def smooth_value(current_value, past_value, tau):
    if past_value is None:
        return current_value
    else:
        return past_value * tau + current_value * (1 - tau)


def get_list_stats(data):
    if len(data) < 1:
        return dict()
    min_data = np.amin(data)
    max_data = np.amax(data)
    mean_data = np.mean(data)
    median_data = np.median(data)
    stats = dict(
        min=min_data,
        max=max_data,
        mean=mean_data,
        median=median_data
    )
    return stats


def get_git_infos(path):
    git_info = None
    try:
        repo = git.Repo(path)
        try:
            branch_name = repo.active_branch.name
        except TypeError:
            branch_name = '[DETACHED]'
        git_info = dict(
            directory=str(path),
            code_diff=repo.git.diff(None),
            code_diff_staged=repo.git.diff('--staged'),
            commit_hash=repo.head.commit.hexsha,
            branch_name=branch_name,
        )
    except git.exc.InvalidGitRepositoryError as e:
        logger.error(f'Not a valid git repo: {path}')
    except git.exc.NoSuchPathError as e:
        logger.error(f'{path} does not exist.')
    return git_info


class RunningMeanStd:
    # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
    def __init__(self, epsilon=1e-4, shape=()):
        self.mean = np.zeros(shape, 'float64')
        self.var = np.ones(shape, 'float64')
        self.count = epsilon

    def update(self, x):
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        self.mean, self.var, self.count = update_mean_var_count_from_moments(
            self.mean, self.var, self.count, batch_mean, batch_var, batch_count)

    def get_states(self):
        return self.mean, self.var, self.count

    def set_states(self, mean, var, count):
        self.mean = mean
        self.var = var
        self.count = count


def update_mean_var_count_from_moments(mean, var, count, batch_mean, batch_var, batch_count):
    delta = batch_mean - mean
    tot_count = count + batch_count

    new_mean = mean + delta * batch_count / tot_count
    m_a = var * count
    m_b = batch_var * batch_count
    M2 = m_a + m_b + np.square(delta) * count * batch_count / tot_count
    new_var = M2 / tot_count
    new_count = tot_count

    return new_mean, new_var, new_count
