__docformat__ = 'google'

import os
import copy
import math
from loguru import logger
from functools import partial
from collections import deque
from typing import Callable, Dict, Iterator, List, Optional, Tuple, cast
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


"""Data type for the optimizer function"""
OPTIM_FN_TYPE = Callable[[List[Tuple[Tensor, str]]], Optimizer]


"""Data type for the learning rate scheduler function"""
LR_SCHEDULER_FN_TYPE = Callable[[Optimizer], LRScheduler]


class DecentralizedDataParallel(Module):
    """Decentralized Data Parallel Class"""


    """Buffer data types that need to be synchronized in global average"""
    FLOAT_DTYPES = [torch.float16, torch.float32, torch.float64]
    

    def __init__(self,
                 model: Module,
                 optim_fn: OPTIM_FN_TYPE,
                 lr_scheduler_fn: Optional[LR_SCHEDULER_FN_TYPE] = None,
                 topology: str = 'complete',
                 scaler: Optional[GradScaler] = None,
                 grad_clip_norm: float = 0.0,
                 param_as_bucket_view: bool = True,
                 sync_buffer_in_global_avg: bool = False,
                 bucket_size_in_mb: int = 25,
                 local_world_size: Optional[int] = None):
        """Decentralized data parallel wrapper for PyTorch module

        Args:
            model (Module): PyTorch module to be wrapped
            optim_fn (OPTIM_FN_TYPE): Function to create the optimizer, which takes a list of tuples of parameters and their names
            lr_scheduler_fn (Optional[LR_SCHEDULER_FN_TYPE], optional): Function to create the learning rate scheduler, \
                which takes the optimizer as input. Defaults to None.
            topology (str, optional): Topology of the decentralized communication graph. Defaults to 'complete'.
            scaler (Optional[GradScaler], optional): Gradient scaler for mixed precision training. Defaults to None.
            grad_clip_norm (float, optional): Gradient clipping norm, set to 0.0 if no gradient clipping is applied. Defaults to 0.0.
            param_as_bucket_view (bool, optional): Whether to use the parameter as a view of part of the contiguous buffer. Defaults to True.
            sync_buffer_in_global_avg (bool, optional): Whether to synchronize the float buffers in the global average. Defaults to False.
            bucket_size_in_mb (int, optional): Size of the bucket in MB. Defaults to 25 MB.
            local_world_size (Optional[int], optional): Provide the local world size if not using the environment variable. Defaults to None.
        """
        super(DecentralizedDataParallel, self).__init__()
        assert dist.is_available() and dist.is_initialized(), 'Distributed environment is not initialized'
        assert torch.cuda.is_available(), 'CUDA is not available'

        self._model = model.cuda()
        self._optim_fn = optim_fn
        self._lr_schd_fn = lr_scheduler_fn
        self._scaler = scaler
        self._grad_clip_norm = grad_clip_norm
        self._param_as_bucket_view = param_as_bucket_view
        self._sync_buffer_in_global_avg = sync_buffer_in_global_avg
        self._bucket_size = bucket_size_in_mb * 1024 * 1024
        self._local_world_size = local_world_size if local_world_size is not None else int(os.environ.get('LOCAL_WORLD_SIZE', 1))

        # get the rank and world size
        self._rank = dist.get_rank()
        self._world_size = dist.get_world_size()

        # check if the model is in "channels_last" memory format
        self._is_channels_last = self._check_channels_last()

        if self._rank == 0:
            logger.debug(f'----- Decentralized Data Parallel ------')
            logger.debug(f'Rank: {self._rank}, Local World Size: {self._local_world_size}, World Size: {self._world_size}')
            logger.debug(f'Topology: {topology}')

        self._params: List[Tensor] = list([x for x in self._model.parameters() if x.requires_grad])
        self._param_names: List[str] = list([n for n, x in self._model.named_parameters() if x.requires_grad])
        self._traced_param_ids: List[int] = []
        self._traced_params: List[Tensor] = []
        self._step: int = -1
        self._comm_op: List[Optional[Work]] = []
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
        self._lr_schedulers: List[Optional[LRScheduler]] = []

        # initialize the topology
        self._topo: Topology = TopologyReg.registry[topology](self._local_world_size)

        # create hooks to trace the used parameters in backward
        self._create_hooks()

        # sync the parameters at the start
        self._sync_at_start()

        # flag for gradient accumulation
        self._accumulating: bool = False


    def _check_channels_last(self) -> bool:
        if not all([x.is_contiguous(memory_format=torch.channels_last) for x in self._model.parameters() if len(x.shape) == 4]):
            return False
        if self._rank == 0:
            logger.debug('Model is with "channels_last" memory format')
        return True


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
    

    def accumulate_grad(self, accumulate: bool = True):
        """Set the gradient accumulation mode

        Args:
            accumulate (bool, optional): Whether to accumulate the gradients. Defaults to True.
        """
        self._accumulating = accumulate

    
    """Hook functions"""

    @torch.no_grad()
    def _trace_fn(self, _: Tensor, pid: int):
        """Hook function to trace the order of used parameters in backward pass

        Args:
            _ (Tensor): corresponding tensor (not used)
            pid (int): parameter id
        """
        self._traced_param_ids.append(pid)

    @torch.no_grad()
    def _ddp_fn(self, _: Tensor, bucket_id: int):
        """Hook function to perform the bucket-wise update and communication

        Args:
            _ (Tensor): corresponding tensor (not used)
            bucket_id (int): bucket id
        """

        # update the count of the last parameter in the bucket
        self._last_param_cnt[bucket_id] -= 1

        # perform the bucket-wise update and communication when all gradients in the bucket are accumulated
        if self._last_param_cnt[bucket_id] == 0:
            self._last_param_cnt[bucket_id] = self._last_param_cnt_b[bucket_id]

            # skip the update and communication if the model is processing micro-batches
            if self._accumulating:
                return

            comm_op = self._comm_op[bucket_id]
            if comm_op is not None:
                # wait for the communication in the previous iteration
                comm_op.wait()
                self._comm_op[bucket_id] = None

                # get the peers to communicate with in this iteration
                edge = self._topo.get_edge(self._step)
                weight = edge.weights[edge.ranks.index(self._rank)]

                # optionally call the pre_average_hook for optimizers using the communication information
                if hasattr(self._optims[bucket_id], 'pre_average_hook'):
                    self._optims[bucket_id].pre_average_hook(edge, weight) # type: ignore

                # replace the local model with the mixed model
                if self._param_as_bucket_view:
                    self._param_blocks[bucket_id].mul_(weight - (1 - weight) / (len(edge.ranks) - 1))
                    self._param_blocks[bucket_id].add_(self._comm_blocks[bucket_id])
                else:
                    torch._foreach_mul_(self._param_buckets[bucket_id], weight - (1 - weight) / (len(edge.ranks) - 1))
                    torch._foreach_add_(self._param_buckets[bucket_id], self._comm_buffers[bucket_id])
            
            # perform local update
            if self._scaler:
                if self._grad_clip_norm > 0:
                    self._scaler.unscale_(self._optims[bucket_id])
                    torch.nn.utils.clip_grad_norm_(self._param_buckets[bucket_id], self._grad_clip_norm)
                self._scaler.step(self._optims[bucket_id])
                if bucket_id == len(self._param_buckets) - 1:
                    self._scaler.update()
            else:
                if self._grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self._param_buckets[bucket_id], self._grad_clip_norm)
                self._optims[bucket_id].step()
            self._optims[bucket_id].zero_grad()

            if self._lr_schedulers[bucket_id] is not None:
                scheduler = cast(LRScheduler, self._lr_schedulers[bucket_id])
                scheduler.step()

            # launch the next communication after updating the weights
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

    @torch.no_grad()
    def _initialize_params(self):
        """Initialize the parameter buckets and communication buffers
        """

        # verify the number of elements and the order of the parameters on different nodes are the same
        verify = [[[i, self._params[i].numel()] for i in self._traced_param_ids]]
        result = [None] if self._rank != 0 else verify
        dist.broadcast_object_list(result, src=0)
        result = cast(List[List[List[int]]], result)
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

        # split the parameters into roughly equal-size buckets, and register hooks on the last parameter of each bucket
        start = 0
        size = 0
        for i in range(len(traced_param_ids_unique)):
            size += self._params[traced_param_ids_unique[i]].numel() * self._params[traced_param_ids_unique[i]].element_size()
            if (size >= self._bucket_size) or (i == len(traced_param_ids_unique) - 1):
                # register hooks on the last parameter of each bucket, passing the bucket id
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

                # create optimizer and learning rate scheduler for parameters in each bucket
                self._optims.append(self._optim_fn(list(zip(self._param_buckets[-1], param_names))))
                self._lr_schedulers.append(self._lr_schd_fn(self._optims[-1]) if self._lr_schd_fn is not None else None)
                size = 0
                start = i + 1
        
        self._last_param_cnt_b = copy.deepcopy(self._last_param_cnt)
        size_dict = {}

        for i in range(len(self._param_buckets)):
            total_size = sum([self._align(p.numel()) for p in self._param_buckets[i]])

            # make sure the total size is unique for each bucket \
            # (not necessary, but make sure the communication operations are unique for each bucket)
            while total_size in size_dict:
                total_size += 1
            size_dict[total_size] = True

            # create the communication buffer for each bucket
            comm_buffer = torch.empty(total_size,
                                      device=self._param_buckets[i][0].device,
                                      requires_grad=False,
                                      dtype=self._param_buckets[i][0].dtype)

            if self._param_as_bucket_view:
                # create the contiguous buffer for each bucket, and let the parameters be views of the fragments of the buffer
                self._param_blocks.append(torch.empty(total_size,
                                                      device=self._param_buckets[i][0].device,
                                                      requires_grad=True,
                                                      dtype=self._param_buckets[i][0].dtype))
                start = 0
                for j in range(len(self._param_buckets[i])):
                    size = self._param_buckets[i][j].numel()
                    if (len(self._param_buckets[i][j].shape) == 4) and self._is_channels_last:
                        # permute the tensor to the channels_last format
                        self._param_blocks[-1].narrow(0, start, size).copy_(self._param_buckets[i][j].permute(0, 2, 3, 1).view(-1))
                        self._param_buckets[i][j].data = self._param_blocks[-1].narrow(0, start, size).view(
                            (self._param_buckets[i][j].shape[0],
                             self._param_buckets[i][j].shape[2],
                             self._param_buckets[i][j].shape[3],
                             self._param_buckets[i][j].shape[1])
                        ).permute(0, 3, 1, 2)
                    else:
                        # otherwise, copy the tensor directly
                        self._param_blocks[-1].narrow(0, start, size).copy_(self._param_buckets[i][j].contiguous(memory_format=torch.contiguous_format).view(-1))
                        self._param_buckets[i][j].data = self._param_blocks[-1].narrow(0, start, size).view_as(self._param_buckets[i][j])
                    start += self._align(size)

            self._comm_buffers.append([])
            self._comm_blocks.append(comm_buffer)
            start = 0
            for j in range(len(self._param_buckets[i])):
                size = self._param_buckets[i][j].numel()
                self._comm_buffers[-1].append(comm_buffer.narrow(0, start, size).view_as(self._param_buckets[i][j]))
                setattr(self._param_buckets[i][j], 'comm_buffer', self._comm_buffers[-1][-1])
                start += self._align(size)
        
        self._comm_op = [None] * len(self._param_buckets)

    def _align(self, size: int):
        """Align the size to 128-byte boundary
        """
        return math.ceil(size / 32) * 32


    """Delegation functions"""

    def train(self, mode: bool = True):
        """Set the module in training mode

        Args:
            mode (bool, optional): Whether to set the module in training mode. Defaults to True.
        """
        self._model.train(mode)
        return self
    
    def eval(self):
        """Set the module in evaluation mode"""
        self._model.eval()
        return self

    def forward(self, *args, **kwargs):
        if (self._step == 0) and (not self._is_initialized): # lazy initialization at the second iteration
            self._is_initialized = True
            # initialize the parameters and communication buffers
            self._initialize_params()

            # manually trigger the communications for the first iteration only
            with torch.no_grad():
                edge = self._topo.get_edge(self._step)
                weight = edge.weights[edge.ranks.index(self._rank)]
                for i in range(len(self._param_buckets)):
                    # update weights
                    if self._scaler:
                        if self._grad_clip_norm > 0:
                            self._scaler.unscale_(self._optims[i])
                            torch.nn.utils.clip_grad_norm_(self._param_buckets[i], self._grad_clip_norm)
                        self._scaler.step(self._optims[i])
                        if i == len(self._param_buckets) - 1:
                            self._scaler.update()
                    else:
                        if self._grad_clip_norm > 0:
                            torch.nn.utils.clip_grad_norm_(self._param_buckets[i], self._grad_clip_norm)
                        self._optims[i].step()
                    self._optims[i].zero_grad()
                    if self._lr_schedulers[i] is not None:
                        scheduler = cast(LRScheduler, self._lr_schedulers[i])
                        scheduler.step()
                    
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

        if self._model.training and (not self._accumulating):
            self._step += 1

        with torch.autograd.profiler.record_function("DecentralizedDataParallel.forward"):
            output = self._model(*args, **kwargs)
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
        """Get the named parameters of the model
        """
        return super().named_parameters(prefix, recurse, remove_duplicate)


    """Utility functions"""

    def reset_time_stats(self):
        """Reset the time statistics"""
        for key in self._time_stats:
            self._time_stats[key].clear()
        self._fw_start = None # type: ignore
        self._fw_end = None # type: ignore

    @torch.no_grad()
    def global_avg(self):
        """Perform global average on the parameters (and buffers if sync_buffer_in_global_avg is True)
            The function is called at the end of the training loop to synchronize the parameters across all nodes for evaluation
        """
        for op in self._comm_op:
            if op is not None:
                op.wait()
        self._comm_op = [None for _ in range(len(self._param_buckets))]

        torch._foreach_div_([x.data for x in self._params], self._world_size)
        for x in self._params:
            dist.all_reduce(x.data, op=dist.ReduceOp.SUM)
        
        if self._sync_buffer_in_global_avg:
            # sync float buffers (like running mean and variance in batch normalization)
            for x in self._model.buffers():
                if x.dtype in self.FLOAT_DTYPES:
                    dist.all_reduce(x.data, op=dist.ReduceOp.SUM)
            torch._foreach_div_([x.data for x in self._model.buffers() if x.dtype in self.FLOAT_DTYPES], self._world_size)

