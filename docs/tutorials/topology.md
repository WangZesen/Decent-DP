# Topology Design Tutorial

In decentralized training, the communication topology defines how workers interact with each other during the parameter averaging process. Decent-DP provides several built-in topologies and supports custom topology implementations.

## Understanding Topologies

A topology in Decent-DP is a communication pattern that determines:
1. Which workers communicate with each other in each iteration
2. The weight assigned to each worker's parameters during averaging

The mathematical formulation for parameter averaging in Decent-DP is:
$$x_i = w \cdot x_i + \frac{1-w}{|R|-1} \sum_{j \in R, j \neq i} x_j$$

Where:
- $x_i$ is the parameter vector of worker $i$
- $w$ is the weight (between 0 and 1)
- $R$ is the set of ranks participating in the communication

## Built-in Topologies

### Complete Topology
All workers communicate with each other in every iteration.

```python
from decent_dp.ddp import DecentralizedDataParallel as DecentDP

model = DecentDP(your_model, optim_fn, topology="complete")
```

- **Weight**: $1/\text{world\_size}$ for each worker
- **Communication Pattern**: Fully connected
- **Use Case**: Best for small clusters where all-to-all communication is feasible

### Ring Topology
Workers form a ring and communicate with one left and one right neighbor in alternating iterations.

```python
model = DecentDP(your_model, optim_fn, topology="ring")
```

- **Weight**: 0.5 for each worker
- **Communication Pattern**: Ring-based, alternating directions
- **Use Case**: Good for large clusters where bandwidth is limited

### One-Peer Exponential Topology
Workers communicate with peers at exponentially increasing distances.

```python
model = DecentDP(your_model, optim_fn, topology="one-peer-exp")
```

- **Weight**: 0.5 for each worker
- **Communication Pattern**: Exponential mixing
- **Use Case**: Fast mixing properties, good convergence rates

### Alternating Exponential-Ring Topology
Alternates between exponential and ring communication patterns.

```python
model = DecentDP(your_model, optim_fn, topology="alternating-exp-ring")
```

- **Weight**: Varies by pattern
- **Communication Pattern**: Alternates between exp and ring
- **Use Case**: Combines benefits of both topologies

## Topology Selection Guide

| Topology | Communication Overhead | Convergence Speed | Memory Usage | Best For |
|----------|------------------------|-------------------|--------------|----------|
| Complete | High (all-to-all) | Fast | Low | Small clusters |
| Ring | Low | Moderate | Low | Large clusters, bandwidth-limited environments |
| One-Peer Exp | Moderate | Fast | Low | Medium clusters, when fast convergence is needed |
| Alternating Exp-Ring | Moderate | Fast | Low | General purpose, combines benefits |

## Custom Topology Implementation

To implement a custom topology, extend the `Topology` base class:

```python
from decent_dp.topo import Topology, Edge

class CustomTopology(Topology):
    def _get_topo_edges(self) -> List[List[Edge]]:
        """Define the communication edges for each iteration.
        
        Returns:
            List[List[Edge]]: A list of lists of Edge objects.
            Each inner list represents one communication iteration.
        """
        edges = []
        
        # Example: Custom pattern with two iterations
        # Iteration 1: Workers 0,1 communicate and 2,3 communicate
        edges.append([
            Edge(ranks=[0, 1], weight=0.5),
            Edge(ranks=[2, 3], weight=0.5)
        ])
        
        # Iteration 2: Workers 0,2 communicate and 1,3 communicate
        edges.append([
            Edge(ranks=[0, 2], weight=0.5),
            Edge(ranks=[1, 3], weight=0.5)
        ])
        
        return edges

# Register the custom topology
from decent_dp.topo import TopologyReg
TopologyReg.registry["custom"] = CustomTopology

# Use the custom topology
model = DecentDP(your_model, optim_fn, topology="custom")
```

## Performance Considerations

1. **Network Bandwidth**: Choose topologies that match your network capabilities
2. **Cluster Size**: Some topologies scale better with larger clusters
3. **Convergence Properties**: Different topologies have different theoretical convergence rates
4. **Fault Tolerance**: Decentralized topologies are inherently more fault-tolerant than centralized approaches

## Best Practices

1. **Start with Complete**: For initial experiments and small clusters
2. **Use Ring for Large Clusters**: When bandwidth is a concern
3. **Experiment with Exponential**: For faster convergence when feasible
4. **Monitor Communication**: Use profiling tools to understand communication patterns
5. **Validate Topology**: Ensure your custom topology is valid (each node participates exactly once per iteration)

## Troubleshooting

### Common Issues

1. **Invalid Topology**: Ensure each worker participates exactly once per iteration
2. **Process Group Creation**: Make sure all ranks are properly initialized
3. **Weight Constraints**: Weights should be between 0 and 1 for convergence

### Debugging Tips

Enable debug logging to see topology information:

```python
import logging
from loguru import logger

logger.add("topology_debug.log", level="DEBUG")
logging.basicConfig(level=logging.DEBUG)
```

This will show detailed information about the communication edges and process groups created for each topology.
