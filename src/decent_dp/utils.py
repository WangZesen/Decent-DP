from typing import Tuple
from loguru import logger
import torch.distributed as dist
import os

def initialize_dist() -> Tuple[int, int]:
    local_world_size = int(os.environ['LOCAL_WORLD_SIZE'])
    local_rank = int(os.environ['LOCAL_RANK'])
    gpus = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    if gpus:
        if not (len(gpus.split(',')) == int(local_world_size)):
            logger.error(f'LOCAL_WORLD_SIZE and CUDA_VISIBLE_DEVICES are not consistent, \
                         {local_world_size} vs {len(gpus.split(","))}')
            raise ValueError()
        os.environ['CUDA_VISIBLE_DEVICES'] = gpus.split(',')[local_rank]
        dist.init_process_group(backend='nccl')
    else:
        dist.init_process_group(backend='gloo')
    return dist.get_rank(), dist.get_world_size()
