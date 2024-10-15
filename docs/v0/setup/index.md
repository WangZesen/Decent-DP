# Installation

Follow the official instruction of PyTorch ([https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/)) to install PyTorch **(with version > 2.1) with CUDA support**.

Then install the extension by
```shell
pip install decent-dp
```

# Basic Usage

```python
import torch.distributed as dist
from decent_dp.ddp import DecentralizedDataParallel as DecentDP

dist.init_process_group(backend="nccl") # only 'nccl' backend is supported for now

model = ... # create model

model = DecentDP(model,
                 <optimizer_fn>, # Callable[[List[Tuple[Tensor, str]]], Optimizer]
                 <scheduler_fn>, # Callable[[Optimizer], LRScheduler]
                 <topology>) # one of 'complete', 'ring', ... (see src/decent_dp/topo.py)

# setup data, criterion, etc.
train_ds = ...
criterion = ...

# training loop
for (data, target) in train_ds:
    output = model(data.to('cuda'))
    loss = criterion(output, target.to('cuda'))
    # optimizer step and scheduler step is integrated in backward pass
    loss.backward()
```