__docformat__ = 'google'

import os
import copy
from loguru import logger
from functools import partial
from collections import deque
from typing import Callable, Dict, Iterator, List, Optional, Tuple
import torch
from torch import Tensor
from torch.nn import Module
import torch.distributed as dist
from torch.optim import Optimizer
from torch.distributed import Work
from torch.cuda.amp import GradScaler
from torch.nn.parameter import Parameter
from torch.utils.hooks import RemovableHandle
from torch.optim.lr_scheduler import LRScheduler
from .topo import TopologyReg, Topology

OPTIM_FN_TYPE = Callable[[List[Tuple[Tensor, str]]], Optimizer]
"""Data type for the optimizer function"""

LR_SCHEDULER_FN_TYPE = Callable[[Optimizer], LRScheduler]
"""Data type for the learning rate scheduler function"""

class DecentralizedDataParallel(Module):
    """Decentralized Data Parallel Class"""

    FLOAT_DTYPES = [torch.float16, torch.float32, torch.float64]
    """Buffer data types that need to be synchronized in global average"""

    def __init__(self,
                 model: Module,
                 optim_fn: OPTIM_FN_TYPE,
                 lr_scheduler_fn: Optional[LR_SCHEDULER_FN_TYPE] = None,
                 topology: str = 'complete',
                 scaler: Optional[GradScaler] = None,
                 param_as_bucket_view: bool = True,
                 sync_buffer_in_global_avg: bool = False,
                 bucket_size_in_mb: int = 25,
                 profile_mode: bool = False,
                 local_world_size: Optional[int] = None):
        """Decentralized data parallel wrapper for PyTorch module

        Args:
            model (Module): PyTorch module to be wrapped
            optim_fn (OPTIM_FN_TYPE): Function to create the optimizer, which takes a list of tuples of parameters and their names
            lr_scheduler_fn (Optional[LR_SCHEDULER_FN_TYPE], optional): Function to create the learning rate scheduler, which takes the optimizer as input. Defaults to None.
            topology (str, optional): Topology of the decentralized communication graph. Defaults to 'complete'.
            scaler (Optional[GradScaler], optional): Gradient scaler for mixed precision training. Defaults to None.
            param_as_bucket_view (bool, optional): Whether to use the parameter as a view of part of the continuous buffer. Defaults to True.
            sync_buffer_in_global_avg (bool, optional): Whether to synchronize the float buffers in the global average. Defaults to False.
            bucket_size_in_mb (int, optional): Size of the bucket in MB. Defaults to 25 MB.
            profile_mode (bool, optional): Whether to enable the profile mode. Defaults to False.
            local_world_size (Optional[int], optional): Provide the local world size if not using the environment variable. Defaults to None.
        """
        super(DecentralizedDataParallel, self).__init__()
        assert dist.is_available() and dist.is_initialized(), 'Distributed environment is not initialized'

        self._model = model.cuda() if torch.cuda.is_available() else model
        self._optim_fn = optim_fn
        self._lr_schd_fn = lr_scheduler_fn
        self._scaler = scaler
        self._param_as_bucket_view = param_as_bucket_view
        self._sync_buffer_in_global_avg = sync_buffer_in_global_avg
        self._bucket_size = bucket_size_in_mb * 1024 * 1024
        if profile_mode and (not torch.cuda.is_available()) and (dist.get_rank() == 0):
            logger.warning('Profile mode is enabled but CUDA is not available, disabling profile mode')
        self._profile_mode = profile_mode and torch.cuda.is_available()
        self._local_world_size = local_world_size if local_world_size is not None else int(os.environ.get('LOCAL_WORLD_SIZE', 1))

        # get the rank and world size
        self._rank = dist.get_rank()
        self._world_size = dist.get_world_size()

        if self._rank == 0:
            logger.debug(f'----- Decentralized Data Parallel ------')
            logger.debug(f'Rank: {self._rank}, Local World Size: {self._local_world_size}, World Size: {self._world_size}')
            logger.debug(f'Topology: {topology}')

        self._topo: Optional[Topology] = None
        self._params: List[Tensor] = list([x for x in self._model.parameters() if x.requires_grad])
        self._param_names: List[str] = list([n for n, x in self._model.named_parameters() if x.requires_grad])
        self._traced_param_ids: List[int] = []
        self._traced_params: List[Tensor] = []
        self._step: int = -1
        self._comm_op: Optional[Work] = None
        self._is_initialized: bool = False
        self._last_param_cnt: List[int] = []
        self._last_param_cnt_b: List[int] = []
        self._trace_hooks: List[RemovableHandle] = []
        self._ddp_hooks: List[RemovableHandle] = []
        self._param_buckets: List[List[Tensor]] = []
        self._param_blocks: List[Tensor] = []
        self._comm_buffers: List[List[Tensor]] = []
        self._comm_blocks: List[Tensor] = []
        self._optims: List[Optimizer] = []
        self._lr_schedulers: List[LRScheduler] = []
        if self._profile_mode:
            self._time_stats: Dict[str, deque] = {
                'forward': deque(maxlen=100000),
                'iter': deque(maxlen=100000)
            }
            self._iter_end: Optional[torch.cuda.Event] = None
            self._fw_start: Optional[torch.cuda.Event] = None
            self._fw_end: Optional[torch.cuda.Event] = None

        # initialize the topology
        self._topo = TopologyReg.registry[topology](self._local_world_size)

        # create hooks to trace the used parameters in backward
        self._create_hooks()

        # sync the parameters at the start
        self._sync_at_start()
    
    def _create_hooks(self):
        for pid, param in enumerate(self._params):
            self._trace_hooks.append(
                param.register_post_accumulate_grad_hook(
                    partial(
                        lambda data, pid: self._trace_fn(data, pid),
                        pid=pid
                    )
                )
            )
    
    @torch.no_grad()
    def _sync_at_start(self):
        for param in self._params:
            dist.broadcast(param, 0)
    
    """Hook functions"""

    @torch.no_grad()
    def _trace_fn(self, _: Tensor, pid: int):
        self._traced_param_ids.append(pid)

    @torch.no_grad()
    def _ddp_fn(self, _: Tensor, bucket_id: int):
        self._last_param_cnt[bucket_id] -= 1
        if self._last_param_cnt[bucket_id] == 0:
            self._last_param_cnt[bucket_id] = self._last_param_cnt_b[bucket_id]

            if self._comm_op[bucket_id] is not None:
                if self._profile_mode:
                    self._comm_events[bucket_id][0] = torch.cuda.Event(enable_timing=True)
                    self._comm_events[bucket_id][0].record(torch.cuda.current_stream())
                self._comm_op[bucket_id].wait()
                if self._profile_mode:
                    self._comm_events[bucket_id][1] = torch.cuda.Event(enable_timing=True)
                    self._comm_events[bucket_id][1].record(torch.cuda.current_stream())
                self._comm_op[bucket_id] = None

                edge = self._topo.get_edge(self._step)
                weight = edge.weights[edge.ranks.index(self._rank)]

                if hasattr(self._optims[bucket_id], 'pre_average_hook'):
                    self._optims[bucket_id].pre_average_hook(edge, weight)

                if not self._param_as_bucket_view:
                    torch._foreach_mul_(self._param_buckets[bucket_id], weight - (1 - weight) / (len(edge.ranks) - 1))
                    torch._foreach_add_(self._param_buckets[bucket_id], self._comm_buffers[bucket_id])
                else:
                    self._param_blocks[bucket_id].mul_(weight - (1 - weight) / (len(edge.ranks) - 1))
                    self._param_blocks[bucket_id].add_(self._comm_blocks[bucket_id])
            
            if self._scaler:
                self._scaler.step(self._optims[bucket_id])
                if bucket_id == len(self._param_buckets) - 1:
                    self._scaler.update()
            else:
                self._optims[bucket_id].step()
            self._optims[bucket_id].zero_grad()

            if self._lr_schedulers[bucket_id] is not None:
                self._lr_schedulers[bucket_id].step()

            # launch the next communication
            if not self._param_as_bucket_view:
                torch._foreach_copy_(self._comm_buffers[bucket_id], self._param_buckets[bucket_id])
            else:
                self._comm_blocks[bucket_id].copy_(self._param_blocks[bucket_id])
            edge = self._topo.get_edge(self._step + 1)
            weight = edge.weights[edge.ranks.index(self._rank)]
            self._comm_blocks[bucket_id].mul_((1 - weight) / (len(edge.ranks) - 1))

            self._comm_op[bucket_id] = dist.all_reduce(
                self._comm_blocks[bucket_id],
                op=dist.ReduceOp.SUM,
                group=edge.group,
                async_op=True
            )
            
            if self._profile_mode and (bucket_id == len(self._param_buckets) - 1):
                iter_end = torch.cuda.Event(enable_timing=True, blocking=True)
                iter_end.record()
                iter_end.synchronize()
                if self._iter_end:
                    non_overlap_comm = 0.
                    for i in range(len(self._param_buckets)):
                        if self._comm_events[i][0] is not None:
                            self._comm_events[i][1].synchronize()
                            non_overlap_comm += self._comm_events[i][0].elapsed_time(self._comm_events[i][1])
                    # self._time_stats['non_overlap_comm'].append(non_overlap_comm)
                    # self._time_stats['iter'].append(self._iter_end.elapsed_time(iter_end))
                    # self._time_stats['compute'].append(self._time_stats['iter'][-1] - self._time_stats['non_overlap_comm'][-1])
                self._iter_end = iter_end

    @torch.no_grad()
    def _initialize_params(self):
        verify = [[[i, self._params[i].numel()] for i in self._traced_param_ids]]
        result = [None] if self._rank != 0 else verify
        dist.broadcast_object_list(result, src=0)
        if not all([x == y for x, y in zip(verify[0], result[0])]):
            logger.error('Number/Order of elements in used parameters is different on different nodes')
            raise RuntimeError()
        for hook in self._trace_hooks:
            hook.remove()
        del self._trace_hooks
        
        # keep the last occurance of each parameter
        traced_param_ids_unique: List[int] = []
        for id in reversed(self._traced_param_ids):
            if id not in traced_param_ids_unique:
                traced_param_ids_unique.append(id)
        traced_param_ids_unique = list(reversed(traced_param_ids_unique))

        # split the parameters into roughly equal buckets, and register hooks on the last parameter of each bucket
        start = 0
        size = 0
        for i in range(len(traced_param_ids_unique)):
            size += self._params[traced_param_ids_unique[i]].numel() * self._params[traced_param_ids_unique[i]].element_size()
            if (size >= self._bucket_size) or (i == len(traced_param_ids_unique) - 1):
                self._ddp_hooks.append(
                    self._params[traced_param_ids_unique[i]].register_post_accumulate_grad_hook(
                        partial(
                            lambda data, bucket_id: self._ddp_fn(data, bucket_id),
                            bucket_id=len(self._ddp_hooks)
                        )
                    )
                )
                self._last_param_cnt.append(self._traced_param_ids.count(traced_param_ids_unique[i]))
                self._param_buckets.append([self._params[j] for j in traced_param_ids_unique[start:i+1]])
                param_names = [self._param_names[j] for j in traced_param_ids_unique[start:i+1]]
                self._optims.append(self._optim_fn(list(zip(self._param_buckets[-1], param_names))))
                self._lr_schedulers.append(self._lr_schd_fn(self._optims[-1]) if self._lr_schd_fn is not None else None)
                size = 0
                start = i + 1
        
        self._last_param_cnt_b = copy.deepcopy(self._last_param_cnt)
        size_dict = {}

        for i in range(len(self._param_buckets)):
            total_size = sum([p.numel() for p in self._param_buckets[i]])

            # make sure the total size is unique for each bucket
            while total_size in size_dict:
                total_size += 1
            size_dict[total_size] = True

            comm_buffer = torch.empty(total_size,
                                      device=self._param_buckets[i][0].device,
                                      requires_grad=False,
                                      dtype=self._param_buckets[i][0].dtype)
            if self._param_as_bucket_view:
                self._param_blocks.append(torch.empty(total_size,
                                                      device=self._param_buckets[i][0].device,
                                                      requires_grad=True,
                                                      dtype=self._param_buckets[i][0].dtype))
                start = 0
                for j in range(len(self._param_buckets[i])):
                    size = self._param_buckets[i][j].numel()
                    self._param_blocks[-1].narrow(0, start, size).copy_(self._param_buckets[i][j].view(-1))
                    self._param_buckets[i][j].data = self._param_blocks[-1].narrow(0, start, size).view_as(self._param_buckets[i][j])
                    start += size

            self._comm_buffers.append([])
            self._comm_blocks.append(comm_buffer)
            start = 0
            for j in range(len(self._param_buckets[i])):
                size = self._param_buckets[i][j].numel()
                self._comm_buffers[-1].append(comm_buffer.narrow(0, start, size).view_as(self._param_buckets[i][j]))
                setattr(self._param_buckets[i][j], 'comm_buffer', self._comm_buffers[-1][-1])
                start += size
        
        self._comm_op = [None] * len(self._param_buckets)
        if self._profile_mode:
            self._comm_events: List[List[Optional[torch.cuda.Event]]] = [[None, None]] * len(self._param_buckets)


    """Delegate functions"""

    def train(self, mode: bool = True):
        """Set the module in training mode

        Args:
            mode (bool, optional): Whether to set the module in training mode. Defaults to True.
        """
        self._model.train(mode)
    
    def eval(self):
        """Set the module in evaluation mode"""
        self._model.eval()

    def forward(self, *args, **kwargs):
        if (self._step == 0) and (not self._is_initialized): # lazy initialization at the second iteration
            self._is_initialized = True
            self._initialize_params()

            # manually trigger the hooks for the first iteration
            with torch.no_grad():
                edge = self._topo.get_edge(self._step)
                weight = edge.weights[edge.ranks.index(self._rank)]
                for i in range(len(self._param_buckets)):
                    # update weights
                    if self._scaler:
                        self._scaler.step(self._optims[i])
                        if i == len(self._param_buckets) - 1:
                            self._scaler.update()
                    else:
                        self._optims[i].step()
                    self._optims[i].zero_grad()
                    if self._lr_schedulers[i] is not None:
                        self._lr_schedulers[i].step()
                    
                    # launch the first communication
                    if not self._param_as_bucket_view:
                        torch._foreach_copy_(self._comm_buffers[i], self._param_buckets[i])
                    else:
                        self._comm_blocks[i].copy_(self._param_blocks[i])
                    
                    self._comm_blocks[i].mul_((1 - weight) / (len(edge.ranks) - 1))
                    self._comm_op[i] = dist.all_reduce(
                        self._comm_blocks[i],
                        op=dist.ReduceOp.SUM,
                        group=edge.group,
                        async_op=True
                    )

        if self._model.training:
            self._step += 1

        with torch.autograd.profiler.record_function("DecentralizedDataParallel.forward"):
            if self._model.training and self._profile_mode:
                fw_start = torch.cuda.Event(enable_timing=True)
                fw_start.record()
                fw_start.synchronize()
                if self._fw_start is not None:
                    self._fw_end.synchronize()
                    self._time_stats['forward'].append(self._fw_start.elapsed_time(self._fw_end))
                    self._time_stats['iter'].append(self._fw_start.elapsed_time(fw_start))                
                self._fw_start = fw_start
            
            output = self._model(*args, **kwargs)
            if self._model.training and self._profile_mode:
                self._fw_end = torch.cuda.Event(enable_timing=True)
                self._fw_end.record()
            return output

    def parameters(self, recurse: bool = True) -> Iterator[Parameter]:
        """Get the parameters of the model

        Args:
            recurse (bool, optional): Whether to get the parameters recursively. Defaults to True.

        Yields:
            Iterator[Parameter]: The iterator of the parameters
        """
        yield from self._model.parameters(recurse)
    
    def named_parameters(self, prefix: str = '', recurse: bool = True, remove_duplicate: bool = True) -> Iterator[Tuple[str, Parameter]]:
        return super().named_parameters(prefix, recurse, remove_duplicate)


    """Utility functions"""

    def get_time_stats(self) -> Dict[str, deque]:
        """Get the time statistics

        The time statistics are collected only when the profile mode is enabled.

        Raises:
            RuntimeError: If profile mode is not enabled

        Returns:
            Dict[str, deque]: The time statistics. The keys are 'compute', 'non_overlap_comm', and 'iter',
            which stand for the computation time, non-overlapping communication time, and iteration time, respectively.
        """
        if not self._profile_mode:
            raise RuntimeError('Profile mode is not enabled')
        return self._time_stats

    def reset_time_stats(self):
        """Reset the time statistics"""
        for key in self._time_stats:
            self._time_stats[key].clear()
        self._fw_start = None
        self._fw_end = None

    @torch.no_grad()
    def global_avg(self):
        """Perform global average on the parameters (or buffers if sync_buffer_in_global_avg is True)"""
        for op in self._comm_op:
            if op is not None:
                op.wait()
        self._comm_op = [None for _ in range(len(self._param_buckets))]

        torch._foreach_div_([x.data for x in self._params], self._world_size)
        dist.all_reduce_coalesced([x.data for x in self._params], op=dist.ReduceOp.SUM)
        
        if self._sync_buffer_in_global_avg:
            # sync float buffers
            dist.all_reduce_coalesced([x.data for x in self._model.buffers() if x.dtype in self.FLOAT_DTYPES], op=dist.ReduceOp.SUM)
            torch._foreach_div_([x.data for x in self._model.buffers() if x.dtype in self.FLOAT_DTYPES], self._world_size)

