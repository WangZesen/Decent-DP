# Getting Started with Decent-DP

Decent-DP is a PyTorch extension that facilitates efficient multi-worker decentralized data parallel training. This guide will help you get started with installing and using Decent-DP in your projects.

## Installation

### Prerequisites

Before installing Decent-DP, ensure you have the following prerequisites:

- Python 3.9 or higher
- PyTorch 2.1.0 or higher
- CUDA (for GPU training, optional but recommended)

### Installation Methods

#### Via pip (Recommended)

Install Decent-DP directly from PyPI:

```bash
pip install decent-dp
```

#### Via uv

If you're using uv as your package manager:

```bash
uv add decent-dp
```

#### From Source

To install from source, clone the repository and install in editable mode:

```bash
git clone https://github.com/WangZesen/Decent-DP.git
cd Decent-DP
pip install -e .
```

## Environment Setup

Decent-DP requires a properly configured distributed environment. You can either manually set up the environment variables or use the provided utility function.

### Manual Setup

Set the following environment variables:

```bash
export LOCAL_WORLD_SIZE=<number_of_processes_per_node>
export LOCAL_RANK=<local_process_rank>
```

Then initialize the distributed process group:

```python
import torch.distributed as dist

dist.init_process_group(
    backend='nccl' if torch.cuda.is_available() else 'gloo',
    init_method='env://'
)
```

### Using Utility Function

Alternatively, use the provided utility function which automatically handles the setup:

```python
from decent_dp.utils import initialize_dist

rank, world_size = initialize_dist()
```

## Basic Usage Example

Here's a simple example to demonstrate how to use Decent-DP:

```python
import torch
import torch.nn as nn
import torch.optim as optim
from decent_dp.ddp import DecentralizedDataParallel as DecentDP
from decent_dp.optim import optim_fn_adamw

# Initialize model
model = nn.Linear(10, 1).cuda()

# Wrap model with DecentDP
ddp_model = DecentDP(
    model,
    optim_fn=optim_fn_adamw,
    topology="complete"
)

# Create dummy data
x = torch.randn(32, 10).cuda()
y = torch.randn(32, 1).cuda()

# Forward pass
output = ddp_model(x)
loss = nn.functional.mse_loss(output, y)

# Backward pass
ddp_model.zero_grad()
loss.backward()

# Note: optimizer.step() is automatically called by DecentDP
```

## Running Distributed Training

To run your training script with multiple processes, use `torchrun`:

```bash
torchrun --nproc_per_node=4 your_training_script.py
```

For multi-node training, you'll also need to specify the master address and port:

```bash
# On master node
torchrun --nproc_per_node=4 --nnodes=2 --node_rank=0 --master_addr="master.node.ip" --master_port=12345 your_training_script.py

# On worker node
torchrun --nproc_per_node=4 --nnodes=2 --node_rank=1 --master_addr="master.node.ip" --master_port=12345 your_training_script.py
```

## Next Steps

After getting familiar with the basic setup, explore these topics:

1. [Decentralized Data Parallel](tutorials/ddp.md) - Learn about the core DDP implementation
2. [Topology Design](tutorials/topology.md) - Understand different communication topologies
3. [Custom Optimizers](tutorials/custom_optimizers.md) - Create your own optimizer functions compatible with Decent-DP

For more advanced usage and performance benchmarks, check out our [benchmark documentation](benchmarks.md).
