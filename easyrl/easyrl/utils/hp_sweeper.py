import argparse
import itertools
import math
import shlex
import subprocess
import sys
import time
from pathlib import Path

import GPUtil
import numpy as np
from easyrl.utils.common import load_from_yaml
from easyrl.utils.non_block_streamreader import NonBlockingStreamReader as NBSR
from easyrl.utils.rl_logger import logger


def expand_hparam_items(hparams, hp_keys, hp_vals, key_prefix=None):
    for key, elem in hparams.items():
        if isinstance(elem, list) or isinstance(elem, tuple):
            key = key if key_prefix is None else f'{key_prefix}/{key}'
            hp_keys.append(key)
            hp_vals.append(elem)
        elif isinstance(elem, dict):
            expand_hparam_items(elem, hp_keys, hp_vals, key_prefix=key)
        else:
            key = key if key_prefix is None else f'{key_prefix}/{key}'
            hp_keys.append(key)
            hp_vals.append([elem])


def get_hparams_combo(hparams):
    """
    This function takes in just the hyperparameters (dict) and return the
    combination of all possible hp configuration.

    inputs:
      hparams is a dict, where each key is the name of a commandline arg and
      the value is the target value of the arg.

      However any arg can also be a list and so this function will calculate
      the cross product for all combinations of all args.

    output:
      The return value is a sequence of lists. Each list is one of the
      permutations of argument values.
    """
    hp_vals = []
    hp_keys = []
    expand_hparam_items(hparams, hp_keys, hp_vals)

    new_hp_vals = list(itertools.product(*hp_vals))
    new_hparams_combo = []
    for idx, hp_val in enumerate(new_hp_vals):
        new_hparams_combo.append({k: v for k, v in zip(hp_keys, hp_val)})
    return new_hparams_combo


def cmd_for_hparams(hparams):
    """
    Construct the training script args from the hparams
    """
    cmd = ''
    for field, val in hparams.items():
        # by default, if a boolean variable is not specified
        # with a default config value, we assume the default
        # value is False
        if type(val) is bool:
            if '/' in field and 'true' in field:
                cmd = boolean_cmd(cmd, field, val, default_false=False)
            else:
                cmd = boolean_cmd(cmd, field, val, default_false=True)
        elif val != 'None' and val:
            cmd += f'--{field} {val} '
    return cmd


def boolean_cmd(cmd, field, val, default_false=True):
    if '/' in field:
        field = field.split('/')[-1]
    if val is default_false:
        if default_false:
            cmd += f'--{field} '
        else:
            cmd += f'--no_{field} '
    return cmd


def get_sweep_cmds(yaml_file, use_gpu=True):
    configs = load_from_yaml(yaml_file)
    base_cmd = configs['cmd']
    hparams = configs['hparams']

    hparams_combo = get_hparams_combo(hparams)
    cmds = []
    for idx, hps in enumerate(hparams_combo):
        cmd = base_cmd + ' ' + cmd_for_hparams(hps)
        cmds.append(cmd)

    final_cmds = []
    if use_gpu:
        all_gpus_stats = GPUtil.getGPUs()
        exclude_gpus = configs['exclude_gpus']
        gpu_mem_per_job = configs['gpu_memory_per_job']
        gpu_mem_pct_per_job = float(gpu_mem_per_job) / all_gpus_stats[0].memoryTotal
        if exclude_gpus == 'None':
            exclude_gpus = []
        gpus_to_use = GPUtil.getAvailable(order='first',
                                          limit=100,
                                          maxLoad=0.8,
                                          maxMemory=1 - gpu_mem_pct_per_job,
                                          includeNan=False,
                                          excludeID=exclude_gpus,
                                          excludeUUID=[])
        num_exps = len(cmds)
        gpus_free_mem = [all_gpus_stats[x].memoryFree for x in gpus_to_use]
        sorted_gpu_ids = np.argsort(gpus_free_mem)[::-1]
        allowable_gpu_jobs = [int(math.floor(x / gpu_mem_per_job)) for x in gpus_free_mem]
        jobs_run_on_gpu = [0 for i in range(len(gpus_to_use))]
        can_run_on_gpu = [True for i in range(len(gpus_to_use))]
        gpu_id = 0
        for idx in range(num_exps):
            if not any(can_run_on_gpu):
                logger.warning(f'Run out of GPUs!')
                break
            sorted_gpu_id = sorted_gpu_ids[gpu_id]
            while not can_run_on_gpu[sorted_gpu_id]:
                gpu_id = (gpu_id + 1) % len(gpus_to_use)
                sorted_gpu_id = sorted_gpu_ids[gpu_id]
            final_cmds.append(cmds[idx] + f' --device=cuda:{gpus_to_use[sorted_gpu_id]}')
            jobs_run_on_gpu[sorted_gpu_id] += 1
            can_run_on_gpu[sorted_gpu_id] = jobs_run_on_gpu[sorted_gpu_id] < allowable_gpu_jobs[sorted_gpu_id]
            gpu_id = (gpu_id + 1) % len(gpus_to_use)
    else:
        final_cmds = cmds
    return final_cmds


def run_sweep_cmds(cmds):
    output_dir = Path.cwd().joinpath('sp_outputs')
    output_dir.mkdir(parents=True, exist_ok=True)
    processes = []
    nbsrs = []
    for idx, cmd in enumerate(cmds):
        logger.info(f'CMD_{idx}:{cmd}')
        p = subprocess.Popen(cmd,
                             shell=True,
                             stderr=subprocess.STDOUT,
                             stdout=subprocess.PIPE)
        # p = subprocess.Popen(shlex.split(cmd),
        #                      stderr=subprocess.STDOUT,
        #                      stdout=subprocess.PIPE)
        processes.append(p)
        nbsrs.append(NBSR(p.stdout))
    try:
        all_done = [False for i in range(len(processes))]
        while True:
            for idx, p in enumerate(processes):
                stime = time.time()
                proc_print = False
                while True:
                    lines = nbsrs[idx].readline(0.2)
                    if lines:
                        if not proc_print:
                            logger.info(f'====================================')
                            logger.info(f'Process {idx}:')
                            proc_print = True
                        logger.info(lines.decode('utf-8'))
                        if time.time() - stime > 10:
                            break
                    else:
                        break
                if p.poll() is not None:
                    all_done[idx] = True
            if all(all_done):
                break
            time.sleep(2)
        logger.info('All processes are completed.')
    except KeyboardInterrupt:
        logger.warning('Keyboard interruption.')
    finally:
        print('Exiting...')
        for p in processes:
            p.terminate()
        sys.exit()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg_file', '-f', type=str,
                        required=True, help='config file (yaml)')
    parser.add_argument('--cpu', action='store_true', help='not use gpu')
    args = parser.parse_args()
    cmds = get_sweep_cmds(args.cfg_file, use_gpu=not args.cpu)
    run_sweep_cmds(cmds)


if __name__ == '__main__':
    main()
