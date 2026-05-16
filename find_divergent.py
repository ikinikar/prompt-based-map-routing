"""
find_divergent.py
-----------------
Scans the dataset for prompts where the three routers take meaningfully
different paths, then saves folium maps for the top-K most divergent ones.

Divergence metric: mean pairwise Jaccard distance across all router pairs.
  edge_set(route) = {(u,v) for u,v in zip(route, route[1:])}
  jaccard_dist(A, B) = 1 - |A ∩ B| / |A ∪ B|
  score = mean over all pairs of active routers

Usage:
    KMP_DUPLICATE_LIB_OK=TRUE python find_divergent.py --top 5
"""

import argparse
import json
import os
from itertools import combinations

import networkx as nx
import osmnx as ox

from compare_routes import load_resources, compare_routes, ROUTER_STYLES


# ── Divergence helpers ────────────────────────────────────────────────────────

def route_to_edges(route: list) -> set:
    return set(zip(route[:-1], route[1:]))


def jaccard_dist(a: set, b: set) -> float:
    if not a and not b:
        return 0.0
    return 1.0 - len(a & b) / len(a | b)


def divergence_score(routes: dict) -> float:
    """Mean pairwise Jaccard distance across all router pairs with valid routes."""
    valid = {k: route_to_edges(v) for k, v in routes.items() if v}
    if len(valid) < 2:
        return 0.0
    pairs = list(combinations(valid.keys(), 2))
    return sum(jaccard_dist(valid[a], valid[b]) for a, b in pairs) / len(pairs)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=5,
                    help="Number of most-divergent prompts to visualise")
    ap.add_argument("--out-dir", default="divergent_maps",
                    help="Directory to save HTML maps")
    ap.add_argument("--graph",     default="research/pioneer_valley_v2.pkl")
    ap.add_argument("--model",     default="models/adjustor.pt")
    ap.add_argument("--cache-dir", default="models")
    ap.add_argument("--device",    default="cpu")
    ap.add_argument("--dataset",   default="research/synthetic_dataset.jsonl")
    ap.add_argument("--min-score", type=float, default=0.05,
                    help="Ignore routes with divergence below this threshold")
    ap.add_argument("--limit", type=int, default=None,
                    help="Cap the number of prompts scanned (useful for quick runs)")
    ap.add_argument("--only-cities", nargs="+", default=None,
                    help="Restrict to prompts whose start/end match these cities "
                         "(e.g. --only-cities 'Amherst, MA' 'Northampton, MA')")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # ── Load everything once ──────────────────────────────────────────────────
    G, routers = load_resources(
        graph_path=args.graph,
        model_path=args.model,
        cache_dir=args.cache_dir,
        device=args.device,
    )

    # ── Load and optionally filter dataset ───────────────────────────────────
    with open(args.dataset) as f:
        data = [json.loads(l) for l in f]

    if args.only_cities:
        covered = {c.lower() for c in args.only_cities}
        data = [d for d in data
                if {d["start"].lower(), d["end"].lower()} <= covered]

    if args.limit:
        data = data[: args.limit]

    subset = data
    print(f"Scanning {len(subset)} prompts …\n")

    # Geocode start/end once (they're always the same two cities here)
    location_cache = {}
    def get_node(place: str) -> int:
        if place not in location_cache:
            pt = ox.geocoder.geocode(f"{place}, USA")
            location_cache[place] = ox.distance.nearest_nodes(G, X=pt[1], Y=pt[0])
        return location_cache[place]

    # ── Score every prompt ────────────────────────────────────────────────────
    scored = []
    for item in subset:
        s_node = get_node(item["start"])
        e_node = get_node(item["end"])
        if s_node == e_node or not nx.has_path(G, s_node, e_node):
            continue

        routes = {}
        for name, router in routers.items():
            try:
                routes[name] = router.find_route(s_node, e_node, item["prompt"])
            except Exception as e:
                print(f"  [{name}] error: {e}")
                routes[name] = None

        n_valid = sum(1 for r in routes.values() if r)
        if n_valid < len(routers):
            failed = [k for k, v in routes.items() if not v]
            print(f"Skipping (missing routes for: {', '.join(failed)}): {item['prompt'][:60]}")
            continue

        score = divergence_score(routes)
        if score >= args.min_score:
            scored.append((score, item, routes))

    if not scored:
        print("No divergent prompts found above the min-score threshold.")
        return

    scored.sort(key=lambda x: x[0], reverse=True)
    print(f"Found {len(scored)} prompts with divergence ≥ {args.min_score}")
    print(f"Saving top {args.top} maps to '{args.out_dir}/' …\n")

    # ── Print summary table ───────────────────────────────────────────────────
    print(f"{'Rank':<5} {'Score':<7} {'Prompt'}")
    print("─" * 80)
    for rank, (score, item, _) in enumerate(scored[: args.top], 1):
        print(f"{rank:<5} {score:.3f}   {item['prompt'][:70]}")
    print()

    # ── Save maps for top-K ───────────────────────────────────────────────────
    for rank, (score, item, routes) in enumerate(scored[: args.top], 1):
        # Build map directly from pre-computed routes (skip re-running routers)
        out_path = os.path.join(args.out_dir, f"rank{rank:02d}_score{score:.2f}.html")
        _save_map(G, routes, item["prompt"], item["start"], item["end"],
                  score, out_path)
        print(f"  [{rank}] {out_path}")

    print("\nDone.")


def _save_map(G, routes: dict, prompt: str, start: str, end: str,
              score: float, output_path: str):
    """Build a folium map directly from pre-computed routes."""
    import folium
    from compare_routes import _route_length_km, _add_prompt_box

    s_pt = ox.geocoder.geocode(f"{start}, USA")
    e_pt = ox.geocoder.geocode(f"{end}, USA")
    s_node = ox.distance.nearest_nodes(G, X=s_pt[1], Y=s_pt[0])
    e_node = ox.distance.nearest_nodes(G, X=e_pt[1], Y=e_pt[0])

    s_lat, s_lon = G.nodes[s_node]["y"], G.nodes[s_node]["x"]
    e_lat, e_lon = G.nodes[e_node]["y"], G.nodes[e_node]["x"]

    fmap = folium.Map(
        location=[(s_lat + e_lat) / 2, (s_lon + e_lon) / 2],
        zoom_start=13,
        tiles="CartoDB positron",
    )

    folium.Marker([s_lat, s_lon], popup=f"<b>Start</b><br>{start}",
                  tooltip="Start",
                  icon=folium.Icon(color="green", icon="play", prefix="fa")).add_to(fmap)
    folium.Marker([e_lat, e_lon], popup=f"<b>End</b><br>{end}",
                  tooltip="End",
                  icon=folium.Icon(color="red", icon="flag", prefix="fa")).add_to(fmap)

    for name, route in routes.items():
        style = ROUTER_STYLES.get(name, {"color": "gray", "weight": 4, "opacity": 0.8, "dash_array": None})
        group = folium.FeatureGroup(name=name, show=True)
        if route:
            coords = [(G.nodes[n]["y"], G.nodes[n]["x"]) for n in route]
            km = _route_length_km(G, route)
            folium.PolyLine(
                coords,
                color=style["color"], weight=style["weight"], opacity=style["opacity"],
                dash_array=style.get("dash_array"),
                tooltip=f"{name} — {km:.2f} km",
                popup=folium.Popup(
                    f"<b>{name}</b><br>Nodes: {len(route)}<br>"
                    f"Distance: {km:.2f} km<br><i>{prompt}</i>", max_width=300),
            ).add_to(group)
        group.add_to(fmap)

    folium.LayerControl(collapsed=False).add_to(fmap)
    _add_prompt_box(fmap, f"[Divergence: {score:.3f}] {prompt}", start, end)
    from compare_routes import _colorize_layer_control
    _colorize_layer_control(fmap, ROUTER_STYLES)
    fmap.save(output_path)


if __name__ == "__main__":
    main()
