import math
from loguru import logger
import torch.distributed as dist
from torch.distributed import ProcessGroup
from typing import Callable, Dict, List, Optional, cast
from dataclasses import dataclass

@dataclass
class Edge:
    """Edge class for defining communication patterns among workers.

    Weight defines the fraction of the message that each worker keeps. For example, if the weight is 0.3, \
        then the worker keeps 30% of its message and shares 70% with other workers. \
        $x_i = x_i * weight + \\sum_{j \\in \\text{ranks}} x_j * (1 - weight) / \\text{len(ranks)}$. \
        The weight should be between 0 and 1 for convergence.

    Args:
        ranks (List[int]): List of ranks of workers that communicate in this edge
        weights (List[float]): List of weights for each worker in the edge (required to be the same length as ranks)
        group (Optional[ProcessGroup]): Process group for the edge, which will be created by Topology class
    """
    ranks: List[int]
    weight: float
    group: Optional[ProcessGroup] = None


class Topology:
    def __init__(self, local_world_size):
        '''Topology class for defining communication patterns between workers.

        The class is responsible for creating process groups for each edge in the topology.
        The edges are defined as a list of lists of Edge objects, where each list of edges \
            corresponds to one iteration of the communication pattern.
        The topology is defined by implementing the _get_topo_edges method, which should return \
            a list of lists of Edge objects. When creating a new topology, the method should be \
            implemented to return the edges for the topology. Usable variables are `self._world_size`, \
            `self._local_world_size`, and `self._n_nodes` for the number of processes, processes per node, \
            and number of nodes, respectively.
        A valid topology is one where each node participates in exactly one communication in each iteration.
        
        Args:
            local_world_size (int): Number of processes in each node (added as argument for some testing \
                purposes, should be set as the environment variable LOCAL_WORLD_SIZE for normal cases)
        '''

        assert dist.is_available() and dist.is_initialized(), "Distributed environment is not initialized"
        self._rank: int = dist.get_rank()
        self._world_size = dist.get_world_size()
        self._local_world_size = local_world_size
        assert self._world_size % local_world_size == 0, f"World size must be divisible by local world size, \
            but {self._world_size} is not divisible by {local_world_size}"
        self._n_nodes = self._world_size // local_world_size
        self._registry: Dict[str, ProcessGroup] = {}
        self._edges: List[Edge] = []
        self._create_edges()

    def _create_edges(self):
        """Create process groups for each "edge" (or group) in the topology
        """
        all_edges = self._get_topo_edges()
        self._validate_edges(all_edges)

        # Create default group
        all_ranks = [i for i in range(self._world_size)]
        self._registry['all'] = cast(ProcessGroup, dist.new_group(all_ranks))

        for idx in range(len(all_edges)):
            for edge in all_edges[idx]:
                identifier = str(edge.ranks)
                if not (identifier in self._registry):
                    self._registry[identifier] = cast(ProcessGroup, dist.new_group(edge.ranks))
                edge.group = self._registry[identifier]
        
        for idx in range(len(all_edges)):
            for edge in all_edges[idx]:
                if self._rank in edge.ranks:
                    self._edges.append(edge)
                    break

    def _validate_edges(self, edges: List[List[Edge]]):
        """Verify that the topology is valid. A valid topology is one where each \
            node participates in exactly communication in each iteration

        Args:
            edges (List[List[Edge]]): List of edges for each iteration
        """
        for idx in range(len(edges)):
            used = [False] * self._world_size
            for edge in edges[idx]:
                edge.ranks.sort()
                for rank in edge.ranks:
                    if used[rank]:
                        logger.error(f"Topology is not valid, node {rank} is used more than once in one iteration")
                        raise ValueError()
                    used[rank] = True
            if not all(used):
                logger.error(f"Topology is not valid, some nodes are not involved in an edge in one iteration")
                raise ValueError()

    def get_edge(self, step: int) -> Edge:
        """Get the edge for the given iteration
        """
        return self._edges[step % len(self._edges)]

    def _get_topo_edges(self) -> List[List[Edge]]:
        raise NotImplementedError()


class TopologyReg:
    registry: Dict[str, type[Topology]] = {}

    @classmethod
    def register(cls, name: str) -> Callable:
        def _register(topology) -> type[Topology]:
            if name in cls.registry:
                raise ValueError(f"Topology {name} already registered")
            if not issubclass(topology, Topology):
                raise ValueError(f"Topology {name} must extend class Topology")
            cls.registry[name] = topology
            return topology
        return _register


@TopologyReg.register('complete')
class CompleteTopology(Topology):
    """Complete topology where each node communicates with all other nodes. The weights are 1/n.
    """
    def _get_topo_edges(self) -> List[List[Edge]]:
        return [[Edge(
            ranks=list(range(self._world_size)),
            weight=1.0 / self._world_size,
        )]]


@TopologyReg.register('ring')
class RingTopology(Topology):
    """One-peer ring topology where each node communicates with one of its left and right \
        neighbors (by index) in each iteration. The weights are 0.5 for each neighbor.
    """

    def _get_topo_edges(self) -> List[List[Edge]]:
        if self._world_size % 2 != 0:
            logger.error('Ring topology is not supported for odd world size')
            raise ValueError()

        edges = [[], []]
        # Odd iterations
        for i in range(0, self._world_size, 2):
            edges[0].append(Edge(
                ranks=sorted([i, (i + 1) % self._world_size]),
                weight=0.5
            ))
        # Even iterations
        for i in range(0, self._world_size, 2):
            edges[1].append(Edge(
                ranks=sorted([i, (i - 1 + self._world_size) % self._world_size]),
                weight=0.5
            ))
        return edges


@TopologyReg.register('one-peer-exp')
class OnePeerExpTopology(Topology):
    """One-peer exponential topology.
    """

    def _get_topo_edges(self) -> List[List[Edge]]:
        rounds = round(math.log2(self._world_size))
        if self._world_size != 2 ** rounds:
            logger.error('Exponential topology is only supported for 2^x world size')
            raise ValueError()

        edges = []
        for i in range(rounds):
            edges.append([])
            used = [False] * self._world_size
            for j in range(self._world_size):
                if not used[j]:
                    used[j] = True
                    used[(j + 2 ** i) % self._world_size] = True
                    edges[i].append(Edge(
                        ranks=sorted([j, (j + 2 ** i) % self._world_size]),
                        weight=0.5
                    ))
        return edges


@TopologyReg.register('alternating-exp-ring')
class AlternatingExpRingTopology(Topology):
    def _get_topo_edges(self) -> List[List[Edge]]:
        rounds = round(math.log2(self._n_nodes))
        if (self._n_nodes != 2 ** rounds) or (rounds < 1):
            logger.error('Exponential ring topology is only supported for 2^x nodes and x > 1')
            raise ValueError()
        edges = []
        cnt = 0
        for i in range(rounds):
            edges.append([])
            edges.append([])
            used = [False] * self._n_nodes
            for j in range(self._n_nodes):
                if not used[j]:
                    used[j] = True
                    t = (j + 2 ** i) % self._n_nodes
                    used[t] = True
                    if cnt % 2 == 0:
                        edges[-1].append(Edge(
                            ranks=list(range(j * self._local_world_size, (j + 1) * self._local_world_size)) + \
                                  list(range(t * self._local_world_size, (t + 1) * self._local_world_size)),
                            weight=1. / self._local_world_size / 2.
                        ))
                        edges[-2].append(Edge(
                            ranks=list(range(j * self._local_world_size, (j + 1) * self._local_world_size)),
                            weight=1. / self._local_world_size
                        ))
                        edges[-2].append(Edge(
                            ranks=list(range(t * self._local_world_size, (t + 1) * self._local_world_size)),
                            weight=1. / self._local_world_size
                        ))
                    else:
                        edges[-1].append(Edge(
                            ranks=list(range(j * self._local_world_size, (j + 1) * self._local_world_size)),
                            weight=1. / self._local_world_size
                        ))
                        edges[-1].append(Edge(
                            ranks=list(range(t * self._local_world_size, (t + 1) * self._local_world_size)),
                            weight=1. / self._local_world_size
                        ))
                        edges[-2].append(Edge(
                            ranks=list(range(j * self._local_world_size, (j + 1) * self._local_world_size)) + \
                                  list(range(t * self._local_world_size, (t + 1) * self._local_world_size)),
                            weight=1. / self._local_world_size / 2.
                        ))
                    cnt += 1
        return edges

__all__ = ['Topology', 'TopologyReg', 'Edge']
