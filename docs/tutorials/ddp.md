# Decentralized Data Parallel (DDP) Tutorial

The `DecentralizedDataParallel` (DecentDP) class is the core component of the Decent-DP library. It wraps your PyTorch model to enable decentralized training across multiple workers without a central parameter server.

## Overview

Unlike PyTorch's standard `DistributedDataParallel` which relies on a centralized synchronization mechanism, DecentDP implements a fully decentralized approach where each worker communicates directly with its neighbors according to a specified topology.

## Key Features

### Parameter Bucketing
DecentDP automatically groups model parameters into buckets based on size (default 25MB per bucket) to optimize communication efficiency. This is especially important in decentralized settings where communication patterns are more complex.

### Gradient Accumulation Support
The framework seamlessly handles gradient accumulation, which is crucial for simulating larger batch sizes in decentralized training scenarios.

### Automatic Optimizer Management
DecentDP creates and manages separate optimizers for each parameter bucket, automatically calling `step()` and `zero_grad()` at the appropriate times.

## Initialization

To initialize DecentDP, you need to provide:

1. Your model (already moved to the appropriate device)
2. An optimizer function that creates optimizers for parameter groups
3. (Optional) A learning rate scheduler function
4. (Optional) Communication topology

```python
from decent_dp.ddp import DecentralizedDataParallel as DecentDP

# Basic initialization
ddp_model = DecentDP(
    model,                    # Your PyTorch model (on GPU/CPU)
    optim_fn,                 # Optimizer constructor function
    lr_scheduler_fn=None,     # Optional LR scheduler constructor
    topology="complete",      # Communication topology
    bucket_size_in_mb=25      # Size of parameter buckets
)
```

## Optimizer Functions

DecentDP requires optimizer functions rather than direct optimizer instances because it manages multiple optimizers for different parameter buckets.

### Predefined Optimizer Functions

The library provides several predefined optimizer functions:

```python
from decent_dp.optim import (
    optim_fn_adam,
    optim_fn_adamw,
    optim_fn_accum_adam,
    optim_fn_accum_adamw
)

# Use directly
ddp_model = DecentDP(model, optim_fn=optim_fn_adamw)

# Or customize hyperparameters with partial
from functools import partial

custom_adamw = partial(
    optim_fn_adamw,
    lr=0.001,
    weight_decay=0.01
)

ddp_model = DecentDP(model, optim_fn=custom_adamw)
```

### Custom Optimizer Functions

You can create your own optimizer functions:

```python
def my_optim_fn(params):
    """Create a custom optimizer for the given parameters.
    
    Args:
        params: List of (name, tensor) tuples
        
    Returns:
        torch.optim.Optimizer: Configured optimizer instance
    """
    return torch.optim.SGD(
        [p for _, p in params],
        lr=0.01,
        momentum=0.9
    )

ddp_model = DecentDP(model, optim_fn=my_optim_fn)
```

## Training Loop

The training loop with DecentDP is similar to standard PyTorch but with some key differences:

```python
# Training loop
for epoch in range(num_epochs):
    ddp_model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.cuda(), target.cuda()
        
        # Forward pass
        output = ddp_model(data)
        loss = criterion(output, target)
        
        # Backward pass
        ddp_model.zero_grad()
        loss.backward()
        # Note: No need to call optimizer.step() - DecentDP handles this automatically
        
    # Evaluation
    ddp_model.eval()
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.cuda(), target.cuda()
            output = ddp_model(data)
            val_loss = criterion(output, target)
```

## Gradient Accumulation

To enable gradient accumulation:

```python
# Enable gradient accumulation
ddp_model.set_accumulate_grad(True)

# Accumulate gradients over multiple batches
for i in range(accumulation_steps):
    output = ddp_model(data)
    loss = criterion(output, target) / accumulation_steps
    loss.backward()

# Disable gradient accumulation and perform update
ddp_model.set_accumulate_grad(False)
```

## Communication Topologies

DecentDP supports various communication topologies that define how workers interact:

- `complete`: All workers communicate with each other
- `ring`: Workers form a ring and communicate with neighbors
- `one-peer-exp`: Exponential communication pattern
- `alternating-exp-ring`: Alternates between exponential and ring patterns

For more details on topologies, see the [Topology Design](topology.md) tutorial.

## Advanced Configuration

### Bucket Size
Control the size of parameter buckets for communication:

```python
ddp_model = DecentDP(
    model,
    optim_fn=optim_fn_adamw,
    bucket_size_in_mb=50  # Larger buckets for fewer communications
)
```

### Gradient Clipping
Apply gradient clipping during training:

```python
ddp_model = DecentDP(
    model,
    optim_fn=optim_fn_adamw,
    grad_clip_norm=1.0  # Clip gradients to norm 1.0
)
```

### Mixed Precision Training
Use gradient scaling for mixed precision training:

```python
from torch.cuda.amp import GradScaler

scaler = GradScaler()
ddp_model = DecentDP(
    model,
    optim_fn=optim_fn_adamw,
    scaler=scaler
)
```

## Performance Considerations

1. **Bucket Size**: Larger buckets reduce communication overhead but may increase memory usage
2. **Topology Selection**: Different topologies have different communication and convergence characteristics
3. **Gradient Accumulation**: Useful for simulating larger batch sizes without memory constraints
4. **Mixed Precision**: Can significantly reduce memory usage and improve training speed

## Troubleshooting

### Common Issues

1. **"Distributed environment is not initialized"**: Make sure to call `dist.init_process_group()` before creating DecentDP instances
2. **Parameter order mismatch**: Ensure all workers have the same model architecture and parameter ordering
3. **Memory issues**: Try reducing bucket size or using gradient accumulation

### Debugging Tips

Enable logging to see detailed information about the initialization and training process:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

This will show information about parameter bucketing, communication patterns, and other internal operations.
