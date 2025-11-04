# Custom Optimizers Tutorial

Decent-DP provides specialized optimizers designed for decentralized training scenarios, particularly with gradient accumulation. This tutorial explains how to use the built-in optimizers and create your own custom optimizer functions.

## Built-in Optimizers

### Standard Optimizers

Decent-DP includes wrapper functions for standard PyTorch optimizers:

- `optim_fn_adam`: Adam optimizer with parameter grouping
- `optim_fn_adamw`: AdamW optimizer with parameter grouping

These functions automatically handle parameter grouping for weight decay:

```python
from decent_dp.optim import optim_fn_adamw

# Basic usage
ddp_model = DecentDP(model, optim_fn=optim_fn_adamw)

# Customized hyperparameters
from functools import partial

custom_adamw = partial(
    optim_fn_adamw,
    lr=0.001,
    beta1=0.9,
    beta2=0.999,
    weight_decay=0.01,
    eps=1e-8
)

ddp_model = DecentDP(model, optim_fn=custom_adamw)
```

<!-- ### Accumulated Gradient Optimizers

For scenarios where you want to accumulate gradients over multiple steps before updating:

- `optim_fn_accum_adam`: Adam optimizer with gradient accumulation
- `optim_fn_accum_adamw`: AdamW optimizer with gradient accumulation

These optimizers are designed to work with Decent-DP's gradient accumulation feature:

```python
from decent_dp.optim import optim_fn_accum_adamw

# With gradient accumulation
ddp_model = DecentDP(model, optim_fn=optim_fn_accum_adamw)

# Enable gradient accumulation in training loop
ddp_model.set_accumulate_grad(True)
for i in range(accumulation_steps):
    output = ddp_model(data)
    loss = criterion(output, target) / accumulation_steps
    loss.backward()

# Disable gradient accumulation and perform update
ddp_model.set_accumulate_grad(False)
``` -->

## Creating Custom Optimizer Functions

Optimizer functions in Decent-DP must follow a specific signature:

```python
def optimizer_function(params: List[Tuple[str, Tensor]]) -> Optimizer:
    """Create an optimizer for the given parameters.
    
    Args:
        params: List of (parameter_name, parameter_tensor) tuples
        
    Returns:
        torch.optim.Optimizer: Configured optimizer instance
    """
    # Extract parameter tensors
    param_tensors = [p for _, p in params]
    
    # Create and return optimizer
    return torch.optim.YourOptimizer(param_tensors, your_hyperparameters)
```

### Example: Custom SGD Optimizer

```python
import torch
from torch.optim import Optimizer
from decent_dp.ddp import DecentralizedDataParallel as DecentDP

def optim_fn_sgd(params, lr=0.01, momentum=0.9, weight_decay=0.0):
    """Custom SGD optimizer function."""
    # Group parameters by weight decay (similar to built-in functions)
    params_no_decay = [x for n, x in params if not (("bn" in n) or ("bias" in n))]
    params_decay = [x for n, x in params if ("bn" in n) or ("bias" in n)]
    
    param_groups = [
        {"params": params_no_decay, "weight_decay": 0.0},
        {"params": params_decay, "weight_decay": weight_decay}
    ]
    
    return torch.optim.SGD(param_groups, lr=lr, momentum=momentum)

# Use the custom optimizer function
ddp_model = DecentDP(model, optim_fn=optim_fn_sgd)
```

### Example: Custom Adam with Different Parameter Grouping

```python
def optim_fn_custom_adam(params, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
    """Custom Adam optimizer with different parameter grouping logic."""
    
    # Group parameters based on their names
    encoder_params = [x for n, x in params if "encoder" in n]
    decoder_params = [x for n, x in params if "decoder" in n]
    
    # Create parameter groups with different learning rates
    param_groups = [
        {"params": encoder_params, "lr": lr * 0.1},  # Encoder with lower LR
        {"params": decoder_params, "lr": lr}        # Decoder with normal LR
    ]
    
    return torch.optim.Adam(param_groups, betas=(beta1, beta2), eps=eps)

# Use the custom optimizer function
ddp_model = DecentDP(model, optim_fn=optim_fn_custom_adam)
```

## Working with Learning Rate Schedulers

Decent-DP also supports learning rate schedulers through the `lr_scheduler_fn` parameter:

```python
from decent_dp.optim import (
    optim_fn_adamw,
    lr_scheduler_fn_cosine_with_warmup
)

# Create model with optimizer and LR scheduler
ddp_model = DecentDP(
    model,
    optim_fn=optim_fn_adamw,
    lr_scheduler_fn=lr_scheduler_fn_cosine_with_warmup
)
```

### Custom Learning Rate Scheduler Function

```python
def custom_lr_scheduler_fn(optimizer, step_size=10, gamma=0.1):
    """Custom learning rate scheduler function."""
    return torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

# Use with custom scheduler
ddp_model = DecentDP(
    model,
    optim_fn=optim_fn_adamw,
    lr_scheduler_fn=custom_lr_scheduler_fn
)
```

## Advanced: Optimizers with Communication Awareness

Decent-DP allows optimizers to be aware of the communication topology through a special `pre_average_hook` method:

```python
class TopologyAwareOptimizer(torch.optim.Optimizer):
    def __init__(self, params, lr=0.01):
        super().__init__(params, dict(lr=lr))
        
    def pre_average_hook(self, edge, weight):
        """Called before parameter averaging in each iteration.
        
        Args:
            edge: Communication edge information
            weight: Averaging weight for this worker
        """
        # Adjust optimizer behavior based on communication topology
        print(f"Communicating with ranks {edge.ranks} using weight {weight}")
        
        # Example: Adjust learning rate based on communication pattern
        for param_group in self.param_groups:
            param_group['lr'] = self.defaults['lr'] * len(edge.ranks)
```

## Best Practices

1. **Parameter Grouping**: Always consider how to group parameters for weight decay
2. **Hyperparameter Tuning**: Decentralized training may require different hyperparameters than standard training
3. **Gradient Accumulation**: Use accumulated gradient optimizers when simulating larger batch sizes
4. **Consistent Signatures**: Ensure your optimizer functions follow the expected signature
5. **Testing**: Test custom optimizers with simple models before using in production

## Troubleshooting

### Common Issues

1. **Optimizer Function Signature**: Ensure your function takes `params: List[Tuple[str, Tensor]]` and returns an `Optimizer`
2. **Parameter Grouping**: Make sure all parameters are included in exactly one group
3. **Device Placement**: Ensure parameters are on the correct device before creating optimizers

### Debugging Tips

Enable logging to see optimizer creation details:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

This will show information about optimizer creation for each parameter bucket.
