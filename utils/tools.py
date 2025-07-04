import os
import yaml
import logging
import numpy as np

import torch
import torch.distributed as dist


def make_dir(path):
    '''make_dir'''
    if os.path.exists(path):
        return
    else:
        os.makedirs(path)

def _make_paths_absolute(dir_, config):
    """
    Make all values for keys ending with `_path` absolute to dir_.

    Args:
        dir_ (str): The path of yaml configuration file.
        config (dict): The yaml for configuration file.

    Returns:
        Dict. The configuration information in dict format.
    """
    for key in config.keys():
        if key.endswith("_path"):
            config[key] = os.path.join(dir_, config[key])
            config[key] = os.path.abspath(config[key])
        if isinstance(config[key], dict):
            config[key] = _make_paths_absolute(dir_, config[key])
    return config


def load_yaml_config(file_path):
    """
    Load a YAML configuration file.

    Args:
        file_path (str): The path of yaml configuration file.

    Returns:
        Dict. The configuration information in dict format.

    Supported Platforms:
        ``Ascend`` ``CPU`` ``GPU``
    """
    # Read YAML experiment definition file
    with open(file_path, 'r') as stream:
        config = yaml.safe_load(stream)
    config = _make_paths_absolute(os.path.join(os.path.dirname(file_path), ".."), config)
    return config


def get_logger(config):
    """Get logger for saving log"""
    summary_params = config.get('summary')
    logger = create_logger(path=os.path.join(summary_params.get("summary_dir"), "results.log"))
    for key in config:
        logger.info(config[key])
    return logger


def create_logger(path="./log.log", level=logging.INFO):
    logger = logging.getLogger()
    logger.setLevel(level)

    logfile = path
    fh = logging.FileHandler(logfile, mode='a', encoding='utf-8')
    fh.setLevel(level)

    ch = logging.StreamHandler()
    ch.setLevel(level)

    formatter = logging.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger

def init_distributed_mode(args):
    args.rank = args.local_rank
    args.gpu = args.local_rank

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_url = 'env://'  # 设置url
    args.dist_backend = 'nccl'  # 通信后端，nvidia GPU推荐使用NCCL
    print('| distributed init (rank {}): {}'.format(args.rank, args.dist_url), flush=True)
    dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url, world_size=args.world_size, rank=args.rank)
    dist.barrier()  # 等待所有进程都初始化完毕，即所有GPU都要运行到这一步以后在继续
    print('done')

def is_dist_avail_and_initialized():
    """检查是否支持分布式环境"""
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True

def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()

def is_main_process():
    return get_rank() == 0
