import re
import networkx as nx
import osmnx as ox
from typing import Literal, List, Tuple, Dict

# ---------------------------------------------------------------------------
# Pattern table: (compiled_regex, category, osm_values, polarity)
#   polarity = "avoid"  → multiply edge weight UP   (penalty)
#   polarity = "prefer" → multiply edge weight DOWN  (reward)
# ---------------------------------------------------------------------------
from typing import Literal

# Speed tiers for maxspeed_imputed (mph)
# Used separately in weight_func since maxspeed is continuous
SPEED_HIGH_THRESHOLD = 45   # motorway/trunk range
SPEED_LOW_THRESHOLD  = 25   # residential range

KEYWORD_RULES = [

    # ── HIGHWAY TYPE ────────────────────────────────────────────────────────

    # Motorway / Interstate
    (re.compile(r"\b(stay off|keep off|avoid|no|without).{0,20}(highway|freeway|interstate|motorway)\b", re.I),
        "highway", ["motorway", "motorway_link"], "avoid"),
    (re.compile(r"\b(highway|freeway|interstate|motorway)\b", re.I),
        "highway", ["motorway", "motorway_link"], "prefer"),

    # Trunk / Major routes
    (re.compile(r"\b(avoid|no).{0,15}(trunk|major route|divided highway)\b", re.I),
        "highway", ["trunk", "trunk_link"], "avoid"),
    (re.compile(r"\b(trunk|major route|divided highway|multi.?lane)\b", re.I),
        "highway", ["trunk", "trunk_link"], "prefer"),

    # Primary / Arterial
    (re.compile(r"\b(avoid|no).{0,15}(main road|primary|arterial)\b", re.I),
        "highway", ["primary", "primary_link"], "avoid"),
    (re.compile(r"\b(main road|primary|arterial|main street)\b", re.I),
        "highway", ["primary", "primary_link"], "prefer"),

    # Secondary
    (re.compile(r"\b(avoid|no).{0,15}(secondary|connector road)\b", re.I),
        "highway", ["secondary", "secondary_link"], "avoid"),
    (re.compile(r"\b(secondary|connector road)\b", re.I),
        "highway", ["secondary", "secondary_link"], "prefer"),

    # Tertiary
    (re.compile(r"\b(avoid|no).{0,15}(tertiary|minor road)\b", re.I),
        "highway", ["tertiary", "tertiary_link"], "avoid"),
    (re.compile(r"\b(tertiary|minor road)\b", re.I),
        "highway", ["tertiary", "tertiary_link"], "prefer"),

    # Residential / Local
    (re.compile(r"\b(stay off|keep off|avoid|no).{0,20}(residential|back.?road|side.?street|local road|neighborhood)\b", re.I),
        "highway", ["residential", "unclassified"], "avoid"),
    (re.compile(r"\b(residential|back.?road|side.?street|local road|neighborhood|quiet road|keep it local)\b", re.I),
        "highway", ["residential", "unclassified"], "prefer"),
    (re.compile(r"\b(stay off|keep off|avoid).{0,15}(busy|main)\b", re.I),
        "highway", ["residential", "unclassified"], "prefer"),

    # Ramps / Links
    (re.compile(r"\b(avoid|no).{0,15}(ramp|on.?ramp|off.?ramp|link road|slip road)\b", re.I),
        "highway", ["motorway_link", "trunk_link", "primary_link", "secondary_link", "tertiary_link"], "avoid"),
    (re.compile(r"\b(ramp|on.?ramp|off.?ramp|link road|slip road)\b", re.I),
        "highway", ["motorway_link", "trunk_link", "primary_link", "secondary_link", "tertiary_link"], "prefer"),

    # Generic fast/slow proxies (highway type as proxy for speed)
    (re.compile(r"\b(fast|quick|express|direct|fastest)\b", re.I),
        "highway", ["motorway", "trunk", "primary"], "prefer"),
    (re.compile(r"\b(slow|scenic|leisurely|no rush|take my time|relaxed)\b", re.I),
        "highway", ["residential", "tertiary", "unclassified"], "prefer"),

    # ── ROUTE REFERENCES (ref tag) ───────────────────────────────────────────

    (re.compile(r"\b(I-?91|interstate\s*91)\b", re.I),
        "ref", ["I 91"], "prefer"),
    (re.compile(r"\b(I-?90|mass\s*(pike|turnpike)|turnpike)\b", re.I),
        "ref", ["I 90"], "prefer"),
    (re.compile(r"\b(I-?391)\b", re.I),
        "ref", ["I 391"], "prefer"),
    (re.compile(r"\b(route\s*9|MA-?9)\b", re.I),
        "ref", ["MA 9"], "prefer"),
    (re.compile(r"\b(route\s*10|MA-?10)\b", re.I),
        "ref", ["MA 10"], "prefer"),
    (re.compile(r"\b(route\s*9\s*(and|&|/)\s*10|MA-?9\s*(and|&|/)\s*MA-?10)\b", re.I),
        "ref", ["MA 9;MA 10"], "prefer"),
    (re.compile(r"\b(route\s*66|MA-?66)\b", re.I),
        "ref", ["MA 66"], "prefer"),
    (re.compile(r"\b(route\s*116|MA-?116)\b", re.I),
        "ref", ["MA 116"], "prefer"),
    (re.compile(r"\b(route\s*116\s*(and|&|/)\s*141|MA-?116\s*(and|&|/)\s*MA-?141)\b", re.I),
        "ref", ["MA 116;MA 141"], "prefer"),
    (re.compile(r"\b(route\s*141|MA-?141)\b", re.I),
        "ref", ["MA 141"], "prefer"),
    (re.compile(r"\b(route\s*187|MA-?187)\b", re.I),
        "ref", ["MA 187"], "prefer"),
    (re.compile(r"\b(US-?\s*5|route\s*5|US\s*route\s*5)\b", re.I),
        "ref", ["US 5"], "prefer"),
    (re.compile(r"\b(US-?\s*5\s*(and|&|/)\s*(US-?\s*)?202)\b", re.I),
        "ref", ["US 5;US 202"], "prefer"),
    (re.compile(r"\b(US-?\s*5\s*(and|&|/)\s*(MA-?\s*)?10)\b", re.I),
        "ref", ["US 5;MA 10"], "prefer"),
    (re.compile(r"\b(US-?\s*20|route\s*20|US\s*route\s*20)\b", re.I),
        "ref", ["US 20"], "prefer"),
    (re.compile(r"\b(US-?\s*20\s*(and|&|/)\s*(US-?\s*)?202)\b", re.I),
        "ref", ["US 20;US 202;MA 10"], "prefer"),
    (re.compile(r"\b(US-?\s*202|route\s*202|US\s*route\s*202)\b", re.I),
        "ref", ["US 202"], "prefer"),
    (re.compile(r"\b((US-?\s*)?202\s*(and|&|/)\s*(MA-?\s*)?10)\b", re.I),
        "ref", ["US 202;MA 10"], "prefer"),

    # ── ACCESS ──────────────────────────────────────────────────────────────

    (re.compile(r"\b(restricted|private|no access|closed|permit only|no.?through.?road|dead.?end)\b", re.I),
        "access", ["no"], "avoid"),
    (re.compile(r"\b(public|open|accessible)\b", re.I),
        "access", ["yes"], "prefer"),

    # ── BRIDGE ──────────────────────────────────────────────────────────────

    (re.compile(r"\b(avoid|no|without).{0,10}(bridge|viaduct|overpass)\b", re.I),
        "bridge", ["yes", "viaduct"], "avoid"),
    (re.compile(r"\b(bridge|viaduct|overpass)\b", re.I),
        "bridge", ["yes", "viaduct"], "prefer"),

    # ── TUNNEL ──────────────────────────────────────────────────────────────

    (re.compile(r"\b(avoid|no|without).{0,10}tunnel\b", re.I),
        "tunnel", ["yes"], "avoid"),
    (re.compile(r"\btunnel\b", re.I),
        "tunnel", ["yes"], "prefer"),

    # ── JUNCTION ────────────────────────────────────────────────────────────

    (re.compile(r"\b(avoid|no).{0,10}(roundabout|traffic circle|rotary)\b", re.I),
        "junction", ["roundabout"], "avoid"),
    (re.compile(r"\b(roundabout|traffic circle|rotary)\b", re.I),
        "junction", ["roundabout"], "prefer"),
    (re.compile(r"\b(avoid|no).{0,10}(jughandle|jug.?handle|indirect.?turn)\b", re.I),
        "junction", ["jughandle"], "avoid"),
    (re.compile(r"\b(jughandle|jug.?handle|indirect.?turn)\b", re.I),
        "junction", ["jughandle"], "prefer"),

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
            if not rules and not speed_rules:
                return base_cost

            mult = 1.0

            for rule in rules:
                if rule["category"] == "maxspeed":
                    # Handle continuous speed separately
                    speed = data.get("maxspeed_imputed")
                    if speed is None:
                        continue
                    speed = float(speed)
                    if rule["osm_values"] == "high" and speed >= SPEED_HIGH_THRESHOLD:
                        mult *= self.prefer_mult
                    elif rule["osm_values"] == "low" and speed <= SPEED_LOW_THRESHOLD:
                        mult *= self.prefer_mult
                else:
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