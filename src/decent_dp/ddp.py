import os
import copy
import math
from loguru import logger
from functools import partial
from typing import Callable, Iterator, List, Optional, Tuple, cast
import torch
from torch import Tensor
from torch.nn import Module
import torch.distributed as dist
from torch.optim import Optimizer
from torch.distributed import Work
from torch import GradScaler
from torch.nn.parameter import Parameter
from torch.utils.hooks import RemovableHandle
from torch.optim.lr_scheduler import LRScheduler
from .topo import TopologyReg, Topology


"""Data type for the optimizer function"""
OPTIM_FN_TYPE = Callable[[List[Tuple[str, Tensor]]], Optimizer]


"""Data type for the learning rate scheduler function"""
LR_SCHEDULER_FN_TYPE = Callable[[Optimizer], LRScheduler]


class DecentralizedDataParallel(Module):
    """Decentralized data parallel wrapper for PyTorch module

    1. The wrapper places hooks during the backward pass to trace the order of used parameters in the first iteration, and \
    2. Split the parameters into buckets and create optimizers and LR schedulers for each bucket, \
        Add hooks on the last parameter of each bucket to perform the bucket-wise update and communication, \
    3. During the backward passes in the training loop, the hooks are triggered to perform the bucket-wise update and communication

    :Warning: The wrapper currently does not support "channels_last" memory format

    :Warning: The wrapper assumes that the parameter will only be used once in the backward pass

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
                 _local_world_size: Optional[int] = None):
        
        super(DecentralizedDataParallel, self).__init__()
        assert dist.is_available() and dist.is_initialized(), 'Distributed environment is not initialized'

        self._model = model.cuda() if torch.cuda.is_available() else model
        self._optim_fn = optim_fn
        self._lr_schd_fn = lr_scheduler_fn
        self._scaler = scaler
        self._grad_clip_norm = grad_clip_norm
        self._param_as_bucket_view = param_as_bucket_view
        self._sync_buffer_in_global_avg = sync_buffer_in_global_avg
        self._bucket_size = bucket_size_in_mb * 1024 * 1024
        self._local_world_size = _local_world_size if _local_world_size is not None else int(os.environ.get('LOCAL_WORLD_SIZE', 1))

        # get the rank and world size
        self._rank = dist.get_rank()
        self._world_size = dist.get_world_size()

        # check if the model is with "channels_last" memory format
        if self._check_channels_last():
            if self._rank == 0:
                logger.debug(f'The model is with "channels_last" memory format')

        if self._rank == 0:
            logger.debug(f'Initializing Decentralized Data Parallel')
            logger.debug(f'Rank: {self._rank}, Local World Size: {self._local_world_size}, World Size: {self._world_size}, Topology: {topology}')

        # model parameters
        self._params: List[Tensor] = list([x for _, x in self._model.named_parameters() if x.requires_grad])
        self._param_names: List[str] = list([n for n, x in self._model.named_parameters() if x.requires_grad])

        # trace hooks and traced parameter ids
        self._trace_hooks: List[RemovableHandle] = []
        self._traced_param_ids: List[int] = []

        self._step: int = 0
        self._comm_ops: List[Optional[Work]] = []
        
        self._ddp_hooks: List[RemovableHandle] = []
        self._param_buckets: List[List[Tensor]] = []
        self._param_blocks: List[Tensor] = []
        self._comm_buffers: List[List[Tensor]] = []
        self._comm_blocks: List[Tensor] = []

        # Optimizer and LR scheduler
        self._optims: List[Optimizer] = []
        self._lr_schedulers: List[Optional[LRScheduler]] = []

        # initialize the topology
        self._topo: Topology = TopologyReg.registry[topology](self._local_world_size)

        # create hooks to trace the used parameters in backward
        self._create_trace_hooks()

        # sync the parameters at the start
        self._sync_at_start()

        # flag for gradient accumulation
        self._is_grad_accum_enable: bool = False

        # flag for initializing the parameters
        self._initialized: bool = False


    def _check_channels_last(self) -> bool:
        """Check if the model is with "channels_last" memory format

        Returns:
            bool: True if the model is with "channels_last" memory format
        """
        if any([x.is_contiguous(memory_format=torch.channels_last) and (not x.is_contiguous()) for x in self._model.parameters() if len(x.shape) == 4]):
            return True
        return False


    def _create_trace_hooks(self):
        """Create hooks to trace the order of used parameters in backward pass
        """
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
        """Broadcast the parameters of worker 0 to all other workers at the start
        """
        for param in self._params:
            dist.broadcast(param, 0)
    

    def set_accumulate_grad(self, enable: bool = True):
        """Set the gradient accumulation mode

        Args:
            enable (bool, optional): Whether to accumulate the gradients. Defaults to True.
        """
        self._is_grad_accum_enable = enable

    
    """Hook functions"""

    @torch.no_grad()
    def _trace_fn(self, _: Tensor, pid: int):
        """Hook function to trace the order of used parameters in backward pass

        Args:
            _ (Tensor): corresponding tensor (not used)
            pid (int): parameter id
        
        Raises:
            AssertionError: The parameter is used more than once in the backward pass
        """
        if self._is_grad_accum_enable:
            return
        assert not (pid in self._traced_param_ids), 'The parameter is used more than once in the backward pass'
        self._traced_param_ids.append(pid)

    @torch.no_grad()
    def _ddp_fn(self, _: Tensor, bucket_id: int):
        """Hook function to perform the bucket-wise update and communication

        Args:
            _ (Tensor): corresponding tensor (not used)
            bucket_id (int): bucket id
        """

        # skip the update and communication if the model is accumulating gradients
        if self._is_grad_accum_enable:
            return
        
        # perform the bucket-wise update and communication when all gradients in the bucket are accumulated
        comm_op = self._comm_ops[bucket_id]
        if comm_op is not None:
            # wait for the communication from the last iteration
            comm_op.wait()
            self._comm_ops[bucket_id] = None

            # get the peers to communicate with in this iteration
            edge = self._topo.get_edge(self._step)
            weight = edge.weight

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
        if self._param_as_bucket_view:
            self._comm_blocks[bucket_id].copy_(self._param_blocks[bucket_id])
        else:
            torch._foreach_copy_(self._comm_buffers[bucket_id], self._param_buckets[bucket_id])

        edge = self._topo.get_edge(self._step + 1)
        weight = edge.weight
        self._comm_blocks[bucket_id].mul_((1 - weight) / (len(edge.ranks) - 1))

        self._comm_ops[bucket_id] = dist.all_reduce(
            self._comm_blocks[bucket_id],
            op=dist.ReduceOp.SUM,
            group=edge.group,
            async_op=True
        )

    @torch.no_grad()
    def _initialize_params(self):
        """Initialize the parameter buckets and communication buffers

        Raises:
            RuntimeError: Number/Order of elements in used parameters is different on different nodes
        """

        # verify the number of elements and the order of the parameters on different nodes are the same
        verify = [[(i, self._params[i].numel()) for i in self._traced_param_ids]]
        result = [[(0, 0)]] if self._rank != 0 else verify
        dist.broadcast_object_list(result, src=0)
        if not all([x == y for x, y in zip(verify[0], result[0])]):
            raise RuntimeError('Number/Order of elements in used parameters is different on different nodes')
    
        # remove the trace hooks
        for hook in self._trace_hooks:
            hook.remove()
        del self._trace_hooks

        # split the parameters into roughly equal-size buckets, and register hooks on the last parameter of each bucket
        start = 0
        size = 0
        for i in range(len(self._traced_param_ids)):
            size += self._align(self._params[self._traced_param_ids[i]].numel()) * self._params[self._traced_param_ids[i]].element_size()
            if (size >= self._bucket_size) or (i == len(self._traced_param_ids) - 1):
                # register hooks on the last parameter of each bucket, passing the bucket id
                self._ddp_hooks.append(
                    self._params[self._traced_param_ids[i]].register_post_accumulate_grad_hook(
                        partial(
                            lambda data, bucket_id: self._ddp_fn(data, bucket_id),
                            bucket_id=len(self._ddp_hooks)
                        )
                    )
                )
                self._param_buckets.append([self._params[j] for j in self._traced_param_ids[start:i+1]])
                param_names = [self._param_names[j] for j in self._traced_param_ids[start:i+1]]

                # create optimizer and learning rate scheduler for parameters in each bucket
                self._optims.append(self._optim_fn(list(zip(param_names, self._param_buckets[-1]))))
                self._lr_schedulers.append(self._lr_schd_fn(self._optims[-1]) if self._lr_schd_fn is not None else None)
                size = 0
                start = i + 1

        size_dict = {}

        for i in range(len(self._param_buckets)):
            total_size = sum([self._align(p.numel()) for p in self._param_buckets[i]])

            # make sure the total size is unique for each bucket \
            # (not necessary, but make sure the communication operations are unique for each bucket with negligible overhead)
            while total_size in size_dict:
                total_size += 32
            size_dict[total_size] = True

            # create the communication buffer for each bucket
            comm_block = torch.zeros(total_size,
                                     device=self._param_buckets[i][0].device,
                                     requires_grad=False,
                                     dtype=self._param_buckets[i][0].dtype)

            if self._param_as_bucket_view:
                # create contiguous blocks for each bucket, and let the parameters be views of the fragments of the block
                self._param_blocks.append(torch.zeros(total_size,
                                                      device=self._param_buckets[i][0].device,
                                                      requires_grad=True,
                                                      dtype=self._param_buckets[i][0].dtype))
                start = 0
                for j in range(len(self._param_buckets[i])):
                    size = self._param_buckets[i][j].numel()
                    if (len(self._param_buckets[i][j].shape) == 4) and self._param_buckets[i][j].is_contiguous(memory_format=torch.channels_last) \
                        and (not self._param_buckets[i][j].is_contiguous()):
                        # permute the tensor to the channels_last format
                        self._param_blocks[-1].narrow(0, start, size).copy_(self._param_buckets[i][j].permute(0, 2, 3, 1).view(-1))
                        self._param_buckets[i][j].data = self._param_blocks[-1].narrow(0, start, size).view(
                            (self._param_buckets[i][j].shape[0],
                             self._param_buckets[i][j].shape[2],
                             self._param_buckets[i][j].shape[3],
                             self._param_buckets[i][j].shape[1])
                        ).permute(0, 3, 1, 2)
                        assert self._param_buckets[i][j].is_contiguous(memory_format=torch.channels_last)
                        assert not self._param_buckets[i][j].is_contiguous()
                    else:
                        # otherwise, copy the tensor directly
                        assert self._param_buckets[i][j].is_contiguous()
                        self._param_blocks[-1].narrow(0, start, size).copy_(self._param_buckets[i][j].view(-1))
                        self._param_buckets[i][j].data = self._param_blocks[-1].narrow(0, start, size).view_as(self._param_buckets[i][j])
                    start += self._align(size)

            self._comm_blocks.append(comm_block)
            start = 0
            self._comm_buffers.append([])
            for j in range(len(self._param_buckets[i])):
                size = self._param_buckets[i][j].numel()
                if (len(self._param_buckets[i][j].shape) == 4) and self._param_buckets[i][j].is_contiguous(memory_format=torch.channels_last) \
                    and (not self._param_buckets[i][j].is_contiguous()):
                    # permute the tensor to the channels_last format
                    self._comm_buffers[-1].append(comm_block.narrow(0, start, size).view(
                        (self._param_buckets[i][j].shape[0],
                         self._param_buckets[i][j].shape[2],
                         self._param_buckets[i][j].shape[3],
                         self._param_buckets[i][j].shape[1])
                    ).permute(0, 3, 1, 2))
                else:
                    self._comm_buffers[-1].append(comm_block.narrow(0, start, size).view_as(self._param_buckets[i][j]))
                start += self._align(size)

                # attach the communication buffer to the parameter for "pre_average_hook" in the optimizer
                if hasattr(self._optims[i], 'pre_average_hook'):
                    setattr(self._param_buckets[i][j], 'comm_buffer', self._comm_buffers[-1][-1])
            
            # initialize the communication buffer with the initial parameters
            torch._foreach_copy_(self._comm_buffers[-1], self._param_buckets[i])

        self._comm_ops = [None] * len(self._param_buckets)

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
        """Forward pass of the model
        """
        # lazy initialization at the second iteration
        if (self._step == 1) and (not self._initialized):
            self._initialized = True
            # initialize the parameters and communication buffers
            self._initialize_params()

            # manually trigger the communications for the first iteration only
            with torch.no_grad():
                edge = self._topo.get_edge(self._step)
                weight = edge.weight
                for i in range(len(self._param_buckets)):
                    # optionally call the pre_average_hook for optimizers using the communication information
                    if hasattr(self._optims[i], 'pre_average_hook'):
                        self._optims[i].pre_average_hook(edge, weight) # type: ignore

                    # update parameters and launch the first communication
                    if self._scaler:
                        if self._grad_clip_norm > 0:
                            self._scaler.unscale_(self._optims[i])
                            torch.nn.utils.clip_grad_norm_(self._param_buckets[i], self._grad_clip_norm)
                        self._scaler.step(self._optims[i])
                        if i == len(self._param_buckets) - 1:
                            self._scaler.update()
                            # TODO: synchronize the scaler state across all workers?
                    else:
                        if self._grad_clip_norm > 0:
                            torch.nn.utils.clip_grad_norm_(self._param_buckets[i], self._grad_clip_norm)
                        self._optims[i].step()
                    self._optims[i].zero_grad()
                    if self._lr_schedulers[i] is not None:
                        scheduler = cast(LRScheduler, self._lr_schedulers[i])
                        scheduler.step()
                    
                    # launch the first communication
                    if self._param_as_bucket_view:
                        self._comm_blocks[i].copy_(self._param_blocks[i])
                    else:
                        torch._foreach_copy_(self._comm_buffers[i], self._param_buckets[i])
                    
                    self._comm_blocks[i].mul_((1 - weight) / (len(edge.ranks) - 1))
                    comm_op = dist.all_reduce(
                        self._comm_blocks[i],
                        op=dist.ReduceOp.SUM,
                        group=edge.group,
                        async_op=True
                    )
                    self._comm_ops[i] = comm_op
                    # wait for the communication to finish to fully synchronize the workers
                    assert comm_op is not None
                    comm_op.wait()

        if self._model.training and (not self._is_grad_accum_enable):
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

    @torch.no_grad()
    def global_avg(self):
        """Perform global average on the parameters (and buffers if sync_buffer_in_global_avg is True)
            The function is called at the end of the training loop to synchronize the parameters across all nodes for evaluation
        """
        for op in self._comm_ops:
            if op is not None:
                op.wait()
        self._comm_ops = [None for _ in range(len(self._param_buckets))]

        if self._param_as_bucket_view:
            torch._foreach_div_(self._param_blocks, self._world_size)
            for i in range(len(self._param_blocks)):
                dist.all_reduce(self._param_blocks[i], op=dist.ReduceOp.SUM)
        else:
            torch._foreach_div_([x.data for x in self._params], self._world_size)
            for x in self._params:
                dist.all_reduce(x.data, op=dist.ReduceOp.SUM)

        if self._sync_buffer_in_global_avg:
            # globally average the float buffers (e.g. running mean and variance in batch normalization)
            for x in self._model.buffers():
                if x.dtype in self.FLOAT_DTYPES:
                    dist.all_reduce(x.data, op=dist.ReduceOp.SUM)
                    x.data.div_(self._world_size)

__all__ = ['DecentralizedDataParallel',
           'OPTIM_FN_TYPE',
           'LR_SCHEDULER_FN_TYPE']
