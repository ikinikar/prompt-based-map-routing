import re
import networkx as nx
import osmnx as ox
from typing import Literal, List, Tuple, Dict

# ---------------------------------------------------------------------------
# Pattern table: (compiled_regex, category, osm_values, polarity)
#   polarity = "avoid"  → multiply edge weight UP   (penalty)
#   polarity = "prefer" → multiply edge weight DOWN  (reward)
# ---------------------------------------------------------------------------
KEYWORD_RULES: List[Tuple[re.Pattern, str, List[str], str]] = [
    # ── highway type ────────────────────────────────────────────────────────
    (re.compile(r"\b(highway|freeway|interstate|motorway)\b", re.I),
        "highway", ["motorway", "motorway_link"], "prefer"),
    (re.compile(r"\b(avoid|no|without).{0,20}(highway|freeway|interstate|motorway)\b", re.I),
        "highway", ["motorway", "motorway_link"], "avoid"),

    (re.compile(r"\b(on.?ramp|off.?ramp|ramp|link road|slip road)\b", re.I),
        "highway", ["motorway_link", "trunk_link", "primary_link",
                    "secondary_link", "tertiary_link"], "prefer"),
    (re.compile(r"\b(avoid|no).{0,15}(ramp|link)\b", re.I),
        "highway", ["motorway_link", "trunk_link", "primary_link",
                    "secondary_link", "tertiary_link"], "avoid"),

    (re.compile(r"\b(back.?road|side.?road|quiet|residential|neighborhood)\b", re.I),
        "highway", ["residential", "unclassified"], "prefer"),
    (re.compile(r"\b(avoid|no).{0,15}(residential|back.?road)\b", re.I),
        "highway", ["residential", "unclassified"], "avoid"),

    (re.compile(r"\b(main road|main street|primary|arterial)\b", re.I),
        "highway", ["primary", "secondary"], "prefer"),
    (re.compile(r"\b(trunk|route (9|10|20|202|91|90|116|141|66))\b", re.I),
        "highway", ["trunk"], "prefer"),

    # ── access ──────────────────────────────────────────────────────────────
    (re.compile(r"\b(restricted|private|no access|closed)\b", re.I),
        "access", ["no"], "avoid"),
    (re.compile(r"\b(public|open|accessible)\b", re.I),
        "access", ["yes"], "prefer"),

    # ── bridge / tunnel ─────────────────────────────────────────────────────
    (re.compile(r"\b(avoid|no).{0,10}bridge\b", re.I),
        "bridge", ["yes", "viaduct"], "avoid"),
    (re.compile(r"\bbridge\b", re.I),
        "bridge", ["yes", "viaduct"], "prefer"),

    # ── junction ────────────────────────────────────────────────────────────
    (re.compile(r"\b(avoid|no).{0,10}(roundabout|traffic circle)\b", re.I),
        "junction", ["roundabout"], "avoid"),
    (re.compile(r"\b(roundabout|traffic circle)\b", re.I),
        "junction", ["roundabout"], "prefer"),

    # ── speed / road size (continuous → treat as highway-type proxy) ────────
    (re.compile(r"\b(fast|speed|quick|express)\b", re.I),
        "highway", ["motorway", "trunk", "primary"], "prefer"),
    (re.compile(r"\b(slow|scenic|leisurely)\b", re.I),
        "highway", ["residential", "tertiary", "unclassified"], "prefer"),
]

AVOID_MULT  = 25.0   # same scale as your NER weights[0]
PREFER_MULT = 0.04   # same scale as your NER weights[1]


class KeywordRouter:
    def __init__(self, graph, avoid_mult=AVOID_MULT, prefer_mult=PREFER_MULT):
        self.graph = graph
        self.avoid_mult = avoid_mult
        self.prefer_mult = prefer_mult

    def _parse_prompt(self, prompt: str) -> List[Dict]:
        """Return list of active rules: {category, osm_values, polarity}."""
        hits = []
        for pattern, category, values, polarity in KEYWORD_RULES:
            if pattern.search(prompt):
                hits.append({"category": category,
                             "osm_values": values,
                             "polarity": polarity})
        return hits

    def find_route(self, start_node, end_node, prompt: str,
                   algorithm: Literal["dijkstra", "astar"] = "dijkstra"):
        rules = self._parse_prompt(prompt)

        def weight_func(u, v, edge_dict):
            data = list(edge_dict.values())[0] if isinstance(edge_dict, dict) else edge_dict
            base_cost = data.get("length", 1.0)
            if not rules:
                return base_cost

            mult = 1.0
            for rule in rules:
                actual = data.get(rule["category"])
                if actual is None:
                    continue
                current_vals = actual if isinstance(actual, list) else [str(actual)]
                if any(v in rule["osm_values"] for v in current_vals):
                    mult *= self.avoid_mult if rule["polarity"] == "avoid" else self.prefer_mult

            return base_cost * mult

        def heuristic(n, target):
            nd, td = self.graph.nodes[n], self.graph.nodes[target]
            return ox.distance.great_circle(nd["y"], nd["x"], td["y"], td["x"])

        try:
            if algorithm == "astar":
                return nx.astar_path(self.graph, start_node, end_node,
                                     heuristic=heuristic, weight=weight_func)
            else:
                return nx.dijkstra_path(self.graph, start_node, end_node,
                                        weight=weight_func)
        except Exception as e:
            print(f"Routing error ({algorithm}): {e}")
            return None