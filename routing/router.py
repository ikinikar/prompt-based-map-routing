from abc import ABC, abstractmethod
from typing import Literal, Callable

import networkx as nx
import osmnx as ox


class Router(ABC):
    """
    Abstract base class for prompt-based map routers.

    Subclasses implement _build_weight_func() to define how a natural language
    prompt translates into edge weights. The superclass handles heuristic
    computation and pathfinding dispatch (Dijkstra / A*).

    Args:
        graph: An OSMnx/NetworkX MultiDiGraph representing the road network.
    """

    def __init__(self, graph):
        self.graph = graph

    @abstractmethod
    def _build_weight_func(self, prompt: str) -> Callable[[int, int, dict], float]:
        """
        Parse a natural language prompt and return an edge weight function.

        The returned callable must have the signature (u, v, edge_dict) -> float,
        where u and v are node IDs and edge_dict is the edge attribute dictionary.
        Higher weights discourage traversal; lower weights encourage it.

        Args:
            prompt: A natural language navigation request
                    (e.g. "get me there avoiding highways").

        Returns:
            A weight function suitable for nx.dijkstra_path / nx.astar_path.
        """

    def _heuristic(self, n: int, target: int) -> float:
        """
        Great-circle distance heuristic for A* search.

        Computes the straight-line geographic distance (in meters) between two
        nodes using their lat/lon coordinates. This is admissible for road
        networks where edge lengths are geographic distances.

        Args:
            n:      Current node ID.
            target: Goal node ID.

        Returns:
            Distance in meters between the two nodes.
        """
        n_data = self.graph.nodes[n]
        t_data = self.graph.nodes[target]
        return ox.distance.great_circle(n_data['y'], n_data['x'], t_data['y'], t_data['x'])

    def find_route(self, start_node: int, end_node: int, prompt: str,
                   algorithm: Literal["dijkstra", "astar"] = "dijkstra"):
        """
        Find a route between two nodes, weighted by the parsed prompt.

        Delegates prompt interpretation to _build_weight_func(), then runs the
        chosen pathfinding algorithm on the graph.

        Args:
            start_node: Origin node ID in the graph.
            end_node:   Destination node ID in the graph.
            prompt:     Natural language navigation request.
            algorithm:  "dijkstra" (default) or "astar".

        Returns:
            A list of node IDs forming the route, or None if no path exists.
        """
        weight_func = self._build_weight_func(prompt)
        try:
            if algorithm == "astar":
                return nx.astar_path(self.graph, start_node, end_node,
                                     heuristic=self._heuristic, weight=weight_func)
            else:
                return nx.dijkstra_path(self.graph, start_node, end_node,
                                        weight=weight_func)
        except Exception as e:
            print(f"Routing error ({algorithm}): {e}")
            return None
