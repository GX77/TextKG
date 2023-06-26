# -*- coding: utf-8 -*-
# @Time    : 2022/11/12 22:31
# @Author  : Yaojie Shen
# @Project : MM-Video
# @File    : train_utils.py

import datetime
import os
import typing
import torch
import time
import random
import itertools
import numpy as np
import logging
from tabulate import tabulate
from collections import defaultdict
from typing import *

import torch.distributed as dist
from fvcore.common.config import CfgNode

logger = logging.getLogger(__name__)


class Timer:
    def __init__(self, synchronize=False, history_size=1000, precision=3):
        self._precision = precision
        self._stage_index = 0
        self._time_info = {}
        self._time_history = defaultdict(list)
        self._history_size = history_size
        if synchronize:
            assert torch.cuda.is_available(), "cuda is not available for synchronize"
        self._synchronize = synchronize
        self._time = self._get_time()

    def _get_time(self):
        return round(time.time() * 1000, self._precision)

    def __call__(self, stage_name=None, reset=True):
        if self._synchronize:
            torch.cuda.synchronize(torch.cuda.current_device())

        current_time = self._get_time()
        duration = (current_time - self._time)
        if reset:
            self._time = current_time

        if stage_name is None:
            self._time_info[self._stage_index] = duration
        else:
            self._time_info[stage_name] = duration
            self._time_history[stage_name] = self._time_history[stage_name][-self._history_size:]
            self._time_history[stage_name].append(duration)

        return duration

    def reset(self):
        if self._synchronize:
            torch.cuda.synchronize(torch.cuda.current_device())
        self._time = self._get_time()

    def __str__(self):
        return str(self.get_info())

    def get_info(self):
        info = {
            "current": {k: round(v, self._precision) for k, v in self._time_info.items()},
            "average": {k: round(sum(v) / len(v), self._precision) for k, v in self._time_history.items()}
        }
        return info

    def print(self):
        data = [[k, round(sum(v) / len(v), self._precision)] for k, v in self._time_history.items()]
        print(tabulate(data, headers=["Stage", "Time (ms)"], tablefmt="simple"))


class CudaPreFetcher:
    def __init__(self, data_loader):
        self.dl = data_loader
        self.loader = iter(data_loader)
        self.stream = torch.cuda.Stream()
        self.batch = None

    def preload(self):
        try:
            self.batch = next(self.loader)
        except StopIteration:
            self.batch = None
            return
        with torch.cuda.stream(self.stream):
            self.batch = self.cuda(self.batch)

    @staticmethod
    def cuda(x: typing.Any):
        if isinstance(x, list) or isinstance(x, tuple):
            return [CudaPreFetcher.cuda(i) for i in x]
        elif isinstance(x, dict):
            return {k: CudaPreFetcher.cuda(v) for k, v in x.items()}
        elif isinstance(x, torch.Tensor):
            return x.cuda(non_blocking=True)
        else:
            return x

    def __next__(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        batch = self.batch
        if batch is None:
            raise StopIteration
        self.preload()
        return batch

    def __iter__(self):
        self.preload()
        return self

    def __len__(self):
        return len(self.dl)


def manual_seed(cfg: CfgNode):
    if cfg.SYS.DETERMINISTIC:
        torch.manual_seed(cfg.SYS.SEED)
        random.seed(cfg.SYS.SEED)
        np.random.seed(cfg.SYS.SEED)
        torch.cuda.manual_seed(cfg.SYS.SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True
        logger.debug("Manual seed is set")
    else:
        logger.warning("Manual seed is not used")


def init_distributed(proc: int, cfg: CfgNode):
    if cfg.SYS.MULTIPROCESS:  # initialize multiprocess
        word_size = cfg.SYS.NUM_GPU * cfg.SYS.NUM_SHARDS
        rank = cfg.SYS.NUM_GPU * cfg.SYS.SHARD_ID + proc
        dist.init_process_group(backend="nccl", init_method=cfg.SYS.INIT_METHOD, world_size=word_size, rank=rank)
        torch.cuda.set_device(cfg.SYS.GPU_DEVICES[proc])


def save_config(cfg: CfgNode):
    if not dist.is_initialized() or dist.get_rank() == 0:
        config_file = os.path.join(cfg.LOG.DIR, f"config_{get_timestamp()}.yaml")
        with open(config_file, "w") as f:
            f.write(cfg.dump())
        logger.debug("config is saved to %s", config_file)


def get_timestamp():
    return datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S')


def gather_object_multiple_gpu(list_object: List[Any]):
    """
    gather a list of something from multiple GPU
    :param list_object:
    """
    gathered_objects = [None for _ in range(dist.get_world_size())]
    dist.all_gather_object(gathered_objects, list_object)
    return list(itertools.chain(*gathered_objects))
