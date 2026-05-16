"""
Route Comparison Visualizer
----------------------------
Compares all six prompt-based routing methods on a given prompt + start/end
locations and overlays the results on an interactive folium map.

Methods compared:
  1. A* (Baseline)     — plain shortest-path A*, prompt ignored
  2. Keyword           — regex rule-based edge weighting
  3. Neural            — trained Adjustor model edge weighting
  4. GLiNER + A*       — NER entity extraction → OSM tag weights, A* search
  5. GLiNER + SMC      — SMC particle filter, GLiNER judges candidates
  6. SMC + LLaMA       — SMC particle filter, local LLaMA judges candidates

Usage:
    python compare_routes.py \\
        --prompt "Avoid highways and take local roads" \\
        --start "Amherst, MA" \\
        --end "Northampton, MA" \\
        --output comparison_map.html
"""

import argparse
import json
import os
import pickle

import folium
import networkx as nx
import osmnx as ox
from sentence_transformers import SentenceTransformer

from routing.adjustor import precompute_tag_embeddings
from routing.keyword_router import KeywordRouter
from routing.neural_router import NeuralEdgeRouter
from routing.pure_astar import PureAStarRouter
from routing.synset import OSMSemanticBridge

try:
    from routing.NER import GlinerAStarRouter
    from routing.smc_router import SMCGLiNERRouter, SMCLLMRouter
    _GLINER_AVAILABLE = True
except ImportError:
    _GLINER_AVAILABLE = False
    GlinerAStarRouter = SMCGLiNERRouter = SMCLLMRouter = None

# ── Tag schema shared across routers ─────────────────────────────────────────

TAG_SCHEMA = {
    "continuous": {
        "maxspeed_imputed": {"min": 10, "max": 65, "unit": "mph"},
        "lanes": {"min": 1, "max": 6},
        "length": {"min": 2, "max": 6845},
    },
    "discrete": {
        "highway": [
            "residential", "trunk", "secondary", "tertiary", "primary",
            "motorway_link", "unclassified", "secondary_link", "motorway",
            "tertiary_link", "primary_link", "trunk_link",
        ],
        "access": ["yes", "no"],
        "bridge": ["yes", "viaduct"],
        "junction": ["roundabout", "jughandle"],
        "ref": ["US 20", "US 5", "MA 9", "MA 10", "MA 66", "MA 187",
                "US 202", "I 91", "I 90", "MA 116"],
    },
}

DISCRETE_KEYS = list(TAG_SCHEMA["discrete"].keys())

# ── Per-method display styles ─────────────────────────────────────────────────

ROUTER_STYLES = {
    "A* (Baseline)": {"color": "#9E9E9E", "weight": 5, "opacity": 0.85, "dash_array": None},   # gray
    "Keyword":       {"color": "#2196F3", "weight": 5, "opacity": 0.85, "dash_array": None},   # blue
    "Neural":        {"color": "#F44336", "weight": 5, "opacity": 0.85, "dash_array": None},   # red
    "GLiNER + A*":   {"color": "#4CAF50", "weight": 5, "opacity": 0.85, "dash_array": None},   # green
    "GLiNER + SMC":  {"color": "#FF9800", "weight": 5, "opacity": 0.85, "dash_array": None},   # orange
    "SMC + LLaMA":   {"color": "#9C27B0", "weight": 5, "opacity": 0.85, "dash_array": None},   # purple
}


# ── Initialization ────────────────────────────────────────────────────────────

def load_resources(
    graph_path: str = "research/pioneer_valley_v2.pkl",
    model_path: str = "models/adjustor.pt",
    cache_dir: str = "models",
    device: str = "cpu",
    llama_model: str = "llama3.2:3b",
    ollama_url: str = "http://localhost:11434/api/generate",
):
    """Load the graph and initialize all six routers. Call once; reuse the result."""
    print("Loading graph …")
    with open(graph_path, "rb") as f:
        G = pickle.load(f)

    print("Loading sentence transformer …")
    st_model = SentenceTransformer("all-MiniLM-L6-v2", device=device)

    print("Building OSM semantic bridge …")
    bridge = OSMSemanticBridge(TAG_SCHEMA, st_model, threshold=0.80)

    cache_file = os.path.join(cache_dir, "tag_embeddings.npy")
    if not os.path.exists(cache_file):
        print("Pre-computing tag embeddings (one-time cost) …")
        precompute_tag_embeddings(G, st_model, DISCRETE_KEYS, cache_dir)
    else:
        print("Tag embeddings cache found.")

    print("Initializing routers …")

    routers = {
        "A* (Baseline)": PureAStarRouter(G),
        "Keyword": KeywordRouter(G),
        "Neural": NeuralEdgeRouter(
            graph=G,
            st_model=st_model,
            model_path=model_path if os.path.exists(model_path) else None,
            cache_dir=cache_dir,
            discrete_keys=DISCRETE_KEYS,
            similarity_threshold=0.15,
            device=device,
        ),
    }

    if _GLINER_AVAILABLE:
        routers["GLiNER + A*"] = GlinerAStarRouter(G, [25.0, 0.01], bridge)
        print("  Initializing GLiNER + SMC router …")
        routers["GLiNER + SMC"] = SMCGLiNERRouter(G, bridge, TAG_SCHEMA, n_particles=50)
        print("  Initializing SMC + LLaMA router …")
        routers["SMC + LLaMA"] = SMCLLMRouter(
            G, bridge, TAG_SCHEMA, n_particles=50,
            llama_model=llama_model, ollama_url=ollama_url,
        )
    else:
        print("  Warning: 'gliner' not installed — GLiNER + A*, GLiNER + SMC, and SMC + LLaMA disabled.")

    print("All resources loaded.\n")
    return G, routers


# ── Core comparison function ──────────────────────────────────────────────────

def compare_routes(
    G,
    routers: dict,
    prompt: str,
    start: str,
    end: str,
    output_path: str = "comparison_map.html",
    gliner_algorithm: str = "astar",  # unused; kept for CLI compatibility
):
    """
    Run all routers on *prompt* between *start* and *end*, then build a folium
    map with one toggleable FeatureGroup per method.

    Returns:
        Path to the saved HTML file.
    """
    print(f"Geocoding: '{start}' → '{end}' …")
    s_pt = ox.geocoder.geocode(f"{start}, USA")
    e_pt = ox.geocoder.geocode(f"{end}, USA")

    s_node = ox.distance.nearest_nodes(G, X=s_pt[1], Y=s_pt[0])
    e_node = ox.distance.nearest_nodes(G, X=e_pt[1], Y=e_pt[0])

    s_lat, s_lon = G.nodes[s_node]["y"], G.nodes[s_node]["x"]
    e_lat, e_lon = G.nodes[e_node]["y"], G.nodes[e_node]["x"]

    print(f"  Start node {s_node}  ({s_lat:.5f}, {s_lon:.5f})")
    print(f"  End   node {e_node}  ({e_lat:.5f}, {e_lon:.5f})")

    if s_node == e_node:
        raise ValueError("Start and end resolve to the same graph node.")
    if not nx.has_path(G, s_node, e_node):
        raise ValueError("No path exists between start and end nodes.")

    # ── Run each method ───────────────────────────────────────────────────────
    routes = {}
    for name, router in routers.items():
        print(f"Running {name} …")
        try:
            route = router.find_route(s_node, e_node, prompt)
            routes[name] = route
            print(f"  {name}: {len(route)} nodes" if route else f"  {name}: no path")
        except Exception as exc:
            print(f"  {name}: ERROR — {exc}")
            routes[name] = None

    # ── Build folium map ──────────────────────────────────────────────────────
    fmap = folium.Map(
        location=[(s_lat + e_lat) / 2, (s_lon + e_lon) / 2],
        zoom_start=13,
        tiles="CartoDB positron",
    )

    folium.Marker(
        location=[s_lat, s_lon],
        popup=f"<b>Start</b><br>{start}",
        tooltip="Start",
        icon=folium.Icon(color="green", icon="play", prefix="fa"),
    ).add_to(fmap)

    folium.Marker(
        location=[e_lat, e_lon],
        popup=f"<b>End</b><br>{end}",
        tooltip="End",
        icon=folium.Icon(color="red", icon="flag", prefix="fa"),
    ).add_to(fmap)

    for name, route in routes.items():
        style = ROUTER_STYLES.get(name, {"color": "gray", "weight": 4, "opacity": 0.8, "dash_array": None})
        group = folium.FeatureGroup(name=name, show=True)

        if route:
            coords = [(G.nodes[n]["y"], G.nodes[n]["x"]) for n in route]
            route_km = _route_length_km(G, route)
            folium.PolyLine(
                locations=coords,
                color=style["color"],
                weight=style["weight"],
                opacity=style["opacity"],
                dash_array=style.get("dash_array"),
                tooltip=f"{name} — {route_km:.2f} km",
                popup=folium.Popup(
                    f"<b>{name}</b><br>"
                    f"Nodes: {len(route)}<br>"
                    f"Distance: {route_km:.2f} km<br>"
                    f"<i>{prompt}</i>",
                    max_width=300,
                ),
            ).add_to(group)

        group.add_to(fmap)

    folium.LayerControl(collapsed=False).add_to(fmap)
    _add_prompt_box(fmap, prompt, start, end)
    _colorize_layer_control(fmap, ROUTER_STYLES)

    fmap.save(output_path)
    print(f"\nMap saved → {output_path}")
    return output_path


# ── Helpers ───────────────────────────────────────────────────────────────────

def _route_length_km(G, route: list) -> float:
    total = 0.0
    for u, v in zip(route[:-1], route[1:]):
        edge_data = G.get_edge_data(u, v)
        if edge_data:
            total += next(iter(edge_data.values())).get("length", 0.0)
    return total / 1000.0


def _colorize_layer_control(fmap, router_styles: dict):
    """Inject JS that prepends a colored SVG swatch to each layer-control label."""
    colors_json = {name: s["color"] for name, s in router_styles.items()}
    js = f"""
    <script>
    window.addEventListener('load', function() {{
        var colors = {colors_json};
        document.querySelectorAll('.leaflet-control-layers-overlays label').forEach(function(label) {{
            var span = label.querySelector('span');
            if (!span) return;
            var name = span.textContent.trim();
            var color = colors[name];
            if (!color) return;
            var swatch = document.createElement('span');
            swatch.innerHTML = '<svg width="22" height="4" style="vertical-align:middle;margin-right:5px"><line x1="0" y1="2" x2="22" y2="2" stroke="' + color + '" stroke-width="3.5" stroke-linecap="round"/></svg>';
            span.insertBefore(swatch, span.firstChild);
        }});
    }});
    </script>
    """
    fmap.get_root().html.add_child(folium.Element(js))


def _add_prompt_box(fmap, prompt: str, start: str, end: str):
    html = f"""
    <div style="
        position: fixed;
        top: 12px; left: 55px;
        z-index: 1000;
        background: rgba(255,255,255,0.93);
        border: 1px solid #ccc;
        border-radius: 6px;
        padding: 10px 14px;
        font-family: sans-serif;
        font-size: 13px;
        max-width: 380px;
        box-shadow: 2px 2px 6px rgba(0,0,0,0.15);
    ">
        <b>Route Comparison</b><br>
        <span style="color:#555">From:</span> {start}<br>
        <span style="color:#555">To:</span> {end}<br>
        <span style="color:#555">Prompt:</span> <i>"{prompt}"</i>
    </div>
    """
    fmap.get_root().html.add_child(folium.Element(html))


# ── CLI entry point ───────────────────────────────────────────────────────────

def _parse_args():
    p = argparse.ArgumentParser(description="Compare all six prompt-based routing methods.")
    p.add_argument("--prompt",   required=True, help="Natural-language navigation prompt")
    p.add_argument("--start",    required=True, help="Start address (geocodable)")
    p.add_argument("--end",      required=True, help="End address (geocodable)")
    p.add_argument("--output",   default="comparison_map.html")
    p.add_argument("--graph",    default="research/pioneer_valley_v2.pkl")
    p.add_argument("--model",    default="models/adjustor.pt")
    p.add_argument("--cache-dir", default="models")
    p.add_argument("--device",   default="cpu")
    p.add_argument("--llama-model", default="llama3.2:3b")
    p.add_argument("--ollama-url",  default="http://localhost:11434/api/generate")
    p.add_argument("--gliner-algorithm", default="astar",   # kept for backward compat
                   choices=["astar", "dijkstra"], help=argparse.SUPPRESS)
    p.add_argument("--batch", default=None,
                   help="JSONL file of {prompt,start,end} entries for batch mode")
    return p.parse_args()


def main():
    args = _parse_args()

    G, routers = load_resources(
        graph_path=args.graph,
        model_path=args.model,
        cache_dir=args.cache_dir,
        device=args.device,
        llama_model=args.llama_model,
        ollama_url=args.ollama_url,
    )

    if args.batch:
        with open(args.batch) as f:
            entries = [json.loads(line) for line in f]
        base, ext = os.path.splitext(args.output)
        for i, entry in enumerate(entries):
            out = f"{base}_{i:03d}{ext}"
            try:
                compare_routes(G, routers,
                               prompt=entry["prompt"],
                               start=entry["start"],
                               end=entry["end"],
                               output_path=out)
            except Exception as exc:
                print(f"  Skipping entry {i}: {exc}")
    else:
        compare_routes(G, routers,
                       prompt=args.prompt,
                       start=args.start,
                       end=args.end,
                       output_path=args.output)


if __name__ == "__main__":
    main()
