import re
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.distributions import Independent
from torch.distributions import TransformedDistribution
from torch.utils.data import Dataset

from easyrl.utils.rl_logger import logger


def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(
            target_param.data * tau + param.data * (1.0 - tau)
        )


def hard_update(target, source):
    target.load_state_dict(source.state_dict())


def clip_grad(params, max_grad_norm):
    if max_grad_norm is not None:
        grad_norm = torch.nn.utils.clip_grad_norm_(params,
                                                   max_grad_norm)
        grad_norm = grad_norm.item()
    else:
        grad_norm = get_grad_norm(params)
    return grad_norm


def freeze_model(model, eval=True):
    if isinstance(model, list) or isinstance(model, tuple):
        for md in model:
            freeze_model(md)
    else:
        if eval:
            model.eval()
        for param in model.parameters():
            param.requires_grad = False


def unfreeze_model(model):
    if isinstance(model, list) or isinstance(model, tuple):
        for md in model:
            unfreeze_model(md)
    else:
        for param in model.parameters():
            param.requires_grad = True


def move_to(models, device):
    if isinstance(models, list):
        for model in models:
            model.to(device)
    else:
        models.to(device)


def get_grad_norm(model):
    total_norm = 0
    iterator = model.parameters() if isinstance(model, nn.Module) else model
    for p in iterator:
        if p.grad is None:
            continue
        total_norm += p.grad.data.pow(2).sum().item()
    total_norm = total_norm ** 0.5
    return total_norm


def save_model(data, cfg, is_best=False, step=None):
    if not cfg.save_best_only and step is not None:
        ckpt_file = cfg.model_dir \
            .joinpath('ckpt_{:012d}.pt'.format(step))
    else:
        ckpt_file = None
    if is_best:
        best_model_file = cfg.model_dir.joinpath('model_best.pt')
    else:
        best_model_file = None

    if not cfg.save_best_only:
        saved_model_files = sorted(cfg.model_dir.glob('*.pt'))
        if len(saved_model_files) > cfg.max_saved_models:
            saved_model_files[0].unlink()

    logger.info(f'Exploration steps: {step}')
    for fl in [ckpt_file, best_model_file]:
        if fl is not None:
            logger.info(f'Saving checkpoint: {fl}.')
            torch.save(data, fl)


def load_torch_model(model_file):
    logger.info(f'Loading model from {model_file}')
    if isinstance(model_file, str):
        model_file = Path(model_file)
    if not model_file.exists():
        raise ValueError(f'Checkpoint file ({model_file}) '
                         f'does not exist!')
    ckpt_data = torch.load(model_file)
    return ckpt_data


def load_state_dict(model, pretrained_dict):
    model_dict = model.state_dict()
    p_dict = dict()

    for k, v in pretrained_dict.items():
        if k in model_dict and v.shape == model_dict[k].shape:
            p_dict[k] = v
        else:
            logger.warning(f'Param [{k}] not loaded!')
    model_dict.update(p_dict)
    model.load_state_dict(model_dict)


def load_ckpt_data(cfg, step=None, pretrain_model=None):
    if pretrain_model is not None:
        # if the pretrain_model is the path of the folder
        # that contains the checkpoint files, then it will
        # load the most recent one.
        if isinstance(pretrain_model, str):
            pretrain_model = Path(pretrain_model)
        if pretrain_model.suffix != '.pt':
            pretrain_model = get_latest_ckpt(pretrain_model)
        ckpt_data = load_torch_model(pretrain_model)
        return ckpt_data
    if step is None:
        ckpt_file = Path(cfg.model_dir).joinpath('model_best.pt')
    else:
        ckpt_file = Path(cfg.model_dir).joinpath('ckpt_{:012d}.pt'.format(step))

    ckpt_data = load_torch_model(ckpt_file)
    return ckpt_data


def torch_to_np(tensor):
    if not isinstance(tensor, torch.Tensor):
        raise TypeError('tensor has to be a torch tensor!')
    return tensor.cpu().detach().numpy()


def torch_float(array, device='cpu'):
    if isinstance(array, torch.Tensor):
        return array.float().to(device)
    elif isinstance(array, np.ndarray):
        return torch.from_numpy(array).float().to(device)
    elif isinstance(array, list):
        return torch.FloatTensor(array).to(device)


def torch_long(array, device='cpu'):
    if isinstance(array, torch.Tensor):
        return array.long().to(device)
    elif isinstance(array, np.ndarray):
        return torch.from_numpy(array).long().to(device)
    elif isinstance(array, list):
        return torch.LongTensor(array).to(device)


def torch_bool(array, device='cpu'):
    if isinstance(array, torch.Tensor):
        return array.bool().to(device)
    elif isinstance(array, np.ndarray):
        return torch.from_numpy(array).bool().to(device)
    elif isinstance(array, list):
        return torch.BoolTensor(array).to(device)


def action_from_dist(action_dist, sample=True):
    if isinstance(action_dist, Categorical):
        if sample:
            return action_dist.sample()
        else:
            return action_dist.probs.argmax(dim=-1)
    elif isinstance(action_dist, Independent):
        if sample:
            return action_dist.rsample()
        else:
            return action_dist.mean
    elif isinstance(action_dist, TransformedDistribution):
        if not sample:
            if isinstance(action_dist.base_dist, Independent):
                out = action_dist.base_dist.mean
                out = action_dist.transforms[0](out)
                return out
            else:
                raise TypeError('Deterministic sampling is not '
                                'defined for transformed distribution!')
        if action_dist.has_rsample:
            return action_dist.rsample()
        else:
            return action_dist.sample()
    else:
        raise TypeError('Getting actions for the given '
                        'distribution is not implemented!')


def action_log_prob(action, action_dist):
    try:
        log_prob = action_dist.log_prob(action)
    except NotImplementedError:
        raise NotImplementedError('Getting log_prob of actions for the '
                                  'given distribution is not implemented!')
    return log_prob


def action_entropy(action_dist, log_prob=None):
    try:
        entropy = action_dist.entropy()
    except NotImplementedError:
        # Some transformations might not have entropy implemented
        # such as the tanh normal distribution (transformed distribution)
        # TODO which one to use (zero or one sample)
        logger.warning('Entropy function not well defined for the given action distribution.')
        entropy = torch.zeros(tuple(action_dist.batch_shape))
        # try using one sample to approximate the entropy (monte carlo)
        # entropy = -log_prob
    return entropy


def get_jacobian(model, x, out_dim):
    """
    Calculate the jacobian matrix of network output with respect to input x.
    If model only takes one input, then x has to be a 1d tensor (one input)
        or 2d tensor (a batch of inputs).
    If the model takes multiple inputs, then x should be a list/tuple of inputs.
        And each element in the list/tuple should be a 1d or 2d tensor.
    The output is the Jacobian matrix of the network output with respect to input(s).

    Suppose the input feature dimension is E, output feature dimension is O,
        batch size is B.

    If the input is a 1d tensor with shape [E], then the output is a tensor
        with shape [O, E].
    If the input is a 2d tensor with shape [B, E], then the output is a tensor
        with shape [B, O, E]
    If the input is a list/tuple of tensors, then the output will be a list of
       Jacobian matrices. Each element in the list corresponds to the Jacobian
       matrix of the network output with respect to the input in x.

    Args:
        model: neural network model
        x: x can be a 1d or 2d tensor input, or a list/tuple of inputs.
            If the tensor has 2 dimensions, the first dimension means the batch size.

    Returns:
        [O, E] if input is 1d, [B, O, E] if input is 2d,
        a list of [O, E] or a list of [B, O, E] if input is a list/tuple
    """

    def preprocess(x):
        if len(x.shape) > 2:
            raise TypeError('Only 1-dim or 2-dim inputs are supported for now.')
        vec_in = False
        if len(x.shape) == 1:
            vec_in = True
            x = x.unsqueeze(0)
        x = x.unsqueeze(1)
        b = x.shape[0]
        # repeat makes a copy of x, so original x won't be affected
        x = x.repeat(1, out_dim, 1)
        x.requires_grad_(True)
        return x, b, vec_in

    b = 1
    vec_in = False
    multi_in = isinstance(x, tuple) or isinstance(x, list)
    if not multi_in:
        x = [x]

    inputs = []
    for ele in x:
        xp, b, vec_in = preprocess(ele)
        inputs.append(xp)
    y = model(*inputs)
    grad_out = torch.eye(out_dim).expand(b,
                                         out_dim,
                                         out_dim)
    y.backward(grad_out)

    jac = []
    for ele in inputs:
        i_jac = ele.grad.data
        if vec_in:
            i_jac = i_jac.squeeze(0)
        jac.append(i_jac)
    if not multi_in:
        jac = jac[0]
    return jac


def cosine_similarity(x1, x2):
    """

    Args:
        x1: shape [M, N]
        x2: shape [K, N]

    Returns:
        shape [M, K]
    """
    x1 = F.normalize(x1, p=2, dim=1)
    x2 = F.normalize(x2, p=2, dim=1)
    cos_sim = torch.mm(x1, x2.transpose(0, 1))
    return cos_sim


def cdist_l2(x1, x2):
    """
    pairwise l2 distance between x1 and x2

    Args:
        x1: shape [M, N]
        x2: shape [K, N]

    Returns:
        shape [M, K]
    """
    # on pytorch 1.3 and cuda 10
    # this function is much faster then torch.cdist
    x1_norm = x1.pow(2).sum(dim=-1, keepdim=True)
    x2_norm = x2.pow(2).sum(dim=-1, keepdim=True)
    res = torch.addmm(
        x2_norm.transpose(-2, -1),
        x1,
        x2.transpose(-2, -1),
        alpha=-2
    ).add_(x1_norm).clamp_min_(1e-30).sqrt_()
    return res


def batched_cdist_l2(x1, x2):
    """
    cdist in batch mode
    [
     cdist(x1[0], x2[0]),
     cdist(x1[1], x2[1]),
     cdist(x1[2], x2[2]),
     ...
    ]

    Args:
        x1: shape [B, M, N]
        x2: shape [B, K, N]

    Returns:
        shape [B, M, K]
    """
    x1_norm = x1.pow(2).sum(dim=-1, keepdim=True)
    x2_norm = x2.pow(2).sum(dim=-1, keepdim=True)
    res = torch.baddbmm(
        x2_norm.transpose(-2, -1),
        x1,
        x2.transpose(-2, -1),
        alpha=-2
    ).add_(x1_norm).clamp_min_(1e-30).sqrt_()
    return res


def ortho_init(module, nonlinearity=None, weight_scale=1.0, constant_bias=0.0):
    r"""Applies orthogonal initialization for the parameters of a given module.

    Args:
        module (nn.Module): A module to apply orthogonal initialization over its parameters.
        nonlinearity (str, optional): Nonlinearity followed by forward pass of the module. When nonlinearity
            is not ``None``, the gain will be calculated and :attr:`weight_scale` will be ignored.
            Default: ``None``
        weight_scale (float, optional): Scaling factor to initialize the weight. Ignored when
            :attr:`nonlinearity` is not ``None``. Default: 1.0
        constant_bias (float, optional): Constant value to initialize the bias. Default: 0.0

    .. note::

        Currently, the only supported :attr:`module` are elementary neural network layers, e.g.
        nn.Linear, nn.Conv2d, nn.LSTM. The submodules are not supported.

    Example::

        >>> a = nn.Linear(2, 3)
        >>> ortho_init(a)

    """
    if nonlinearity is not None:
        gain = nn.init.calculate_gain(nonlinearity)
    else:
        gain = weight_scale

    if isinstance(module, (nn.RNNBase, nn.RNNCellBase)):
        for name, param in module.named_parameters():
            if 'weight_' in name:
                nn.init.orthogonal_(param, gain=gain)
            elif 'bias_' in name:
                nn.init.constant_(param, constant_bias)
    else:  # other modules with single .weight and .bias
        nn.init.orthogonal_(module.weight, gain=gain)
        nn.init.constant_(module.bias, constant_bias)


class EpisodeDataset(Dataset):
    def __init__(self, swap_leading_axes=True, **kwargs):
        self.data = dict()
        for key, val in kwargs.items():
            if swap_leading_axes:
                self.data[key] = self._swap_leading_axes(val)
            else:
                self.data[key] = val
        self.length = next(iter(self.data.values())).shape[0]

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        sample = dict()
        for key, val in self.data.items():
            sample[key] = val[idx]
        return sample

    def _swap_leading_axes(self, array):
        """
        Swap and then flatten the array along axes 0 and 1

        Args:
            array (np.ndarray): array data

        Returns:
            np.ndarray: reshaped array
        """
        s = array.shape
        return array.swapaxes(0, 1).reshape(s[0] * s[1],
                                            *s[2:])


class DictDataset(Dataset):
    def __init__(self, **kwargs):
        self.data = dict()
        for key, val in kwargs.items():
            self.data[key] = val
        self.length = next(iter(self.data.values())).shape[0]

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        sample = dict()
        for key, val in self.data.items():
            sample[key] = val[idx]
        return sample


def get_latest_ckpt(path):
    ckpt_files = [x for x in list(path.iterdir()) if x.suffix == '.pt']
    num_files = len(ckpt_files)
    if num_files < 1:
        raise ValueError('No checkpoint files found!')
    elif num_files == 1:
        return ckpt_files[0]
    else:
        filenames = [x.name for x in ckpt_files]
        latest_file = None
        latest_step = -np.inf
        for idx, fn in enumerate(filenames):
            num = re.findall(r'\d+', fn)
            if not num:
                continue
            step_num = int(num[0])
            if step_num > latest_step:
                latest_step = step_num
                latest_file = ckpt_files[idx]
        return latest_file
