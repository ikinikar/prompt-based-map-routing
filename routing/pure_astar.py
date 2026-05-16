from typing import Callable

import networkx as nx

from routing.router import Router


class PureAStarRouter(Router):
    """Baseline A* on geographic edge length — ignores the prompt entirely."""

    def _build_weight_func(self, prompt: str) -> Callable:
        return lambda u, v, d: list(d.values())[0].get("length", 1.0)

    def find_route(self, start_node, end_node, prompt, algorithm="astar"):
        try:
            return nx.astar_path(
                self.graph, start_node, end_node,
                heuristic=self._heuristic,
                weight="length",
            )
        except Exception as e:
            print(f"PureAStarRouter error: {e}")
            return None
