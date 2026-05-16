import json
from typing import Callable, List, Optional

import networkx as nx
import numpy as np
import osmnx as ox
import requests
from gliner import GLiNER
from scipy.special import softmax

from routing.router import Router
from routing.synset import OSMSemanticBridge


class SMCRouteGenerator:
    """
    Sequential Monte Carlo route generator using particle filtering.

    Particles move greedily toward the goal while being guided by an anchor
    shortest path and weighted by intent-derived edge multipliers.
    """

    def __init__(self, G, bridge: OSMSemanticBridge, n_particles: int = 50):
        self.G = G
        self.bridge = bridge
        self.n_particles = n_particles

    def _dist(self, node, target) -> float:
        n1, n2 = self.G.nodes[node], self.G.nodes[target]
        dx = (n1["x"] - n2["x"]) * 111000
        dy = (n1["y"] - n2["y"]) * 111000
        return float(np.sqrt(dx * dx + dy * dy))

    def _edge_weight(self, edge_data: dict, intent: dict) -> float:
        mult = 1.0
        for cat in self.bridge.categories:
            val = edge_data.get(cat)
            if not val:
                continue
            val = val[0] if isinstance(val, list) else val
            if val in intent["avoid_tags"].get(cat, []):
                mult *= 0.1
            if val in intent["prefer_tags"].get(cat, []):
                mult *= 10.0
        return mult

    def find_routes(
        self, start_node, end_node, intent: dict, n_candidates: int = 3, max_steps: int = 500
    ) -> List[list]:
        try:
            anchor_nodes = set(nx.astar_path(self.G, start_node, end_node, weight="length"))
        except nx.NetworkXNoPath:
            return []

        # Each particle: [current_node, path, weight, done]
        particles = [[start_node, [start_node], 1.0, False] for _ in range(self.n_particles)]
        completed: dict = {}

        for step in range(max_steps):
            active = [i for i, p in enumerate(particles) if not p[3]]
            if not active or len(completed) >= n_candidates:
                break

            for i in active:
                curr = particles[i][0]
                neighbors = list(self.G.neighbors(curr))
                if not neighbors:
                    particles[i][2] = 0.0
                    continue

                dists = np.array([self._dist(n, end_node) for n in neighbors])
                anchor_bonus = np.array([50.0 if n in anchor_nodes else 0.0 for n in neighbors])
                probs = softmax(-(dists / 100.0) + anchor_bonus / 2.0)

                next_node = np.random.choice(neighbors, p=probs)
                raw = self.G.get_edge_data(curr, next_node)
                data = list(raw.values())[0] if isinstance(raw, dict) else raw

                particles[i][2] *= self._edge_weight(data, intent)
                particles[i][0] = next_node
                particles[i][1].append(next_node)

                if next_node == end_node:
                    particles[i][3] = True
                    completed[tuple(particles[i][1])] = particles[i][2]

            # Resample every 20 steps
            if step % 20 == 0:
                w = np.array([p[2] for p in particles])
                total = w.sum()
                if total > 0:
                    w /= total
                    idx = np.random.choice(self.n_particles, size=self.n_particles, p=w)
                    particles = [
                        [particles[j][0], list(particles[j][1]), 1.0, particles[j][3]]
                        for j in idx
                    ]

        if completed:
            ranked = sorted(completed.items(), key=lambda x: x[1], reverse=True)
            return [list(path) for path, _ in ranked[:n_candidates]]

        # No particles reached the goal — fall back to the anchor shortest path
        print("  SMC: no particles completed, using anchor A* path as fallback")
        return [anchor_path]


class _GLiNERExtractor:
    """GLiNER-based intent extraction shared by both SMC router variants."""

    def __init__(self, model_id: str = "urchade/gliner_medium-v2.1"):
        self.model = GLiNER.from_pretrained(model_id)
        self.labels = ["avoid_feature", "preferred_feature"]

    def extract(self, prompt: str, tag_schema: dict) -> dict:
        entities = self.model.predict_entities(prompt, self.labels, threshold=0.55)
        intent: dict = {"avoid_tags": {}, "prefer_tags": {}}
        for ent in entities:
            text = ent["text"].lower()
            bucket = "avoid_tags" if ent["label"] == "avoid_feature" else "prefer_tags"
            for cat, values in tag_schema["discrete"].items():
                for val in values:
                    if val in text or text in val:
                        intent[bucket].setdefault(cat, set()).add(val)
        return {k: {c: list(vs) for c, vs in v.items()} for k, v in intent.items()}


class SMCGLiNERRouter(Router):
    """
    SMC particle-filter candidate generation with GLiNER as the final judge.

    The judge scores candidates by counting preferred/avoided tag matches
    across the edges of each route, then returns the highest-scoring one.
    """

    def __init__(
        self,
        graph,
        bridge: OSMSemanticBridge,
        tag_schema: dict,
        n_particles: int = 50,
        gliner_model_id: str = "urchade/gliner_medium-v2.1",
    ):
        super().__init__(graph)
        self.tag_schema = tag_schema
        self.extractor = _GLiNERExtractor(gliner_model_id)
        self.smc = SMCRouteGenerator(graph, bridge, n_particles=n_particles)

    def _build_weight_func(self, prompt: str) -> Callable:
        raise NotImplementedError

    def _judge(self, prompt: str, candidates: List[list], intent: dict) -> Optional[list]:
        if not candidates:
            return None
        if len(candidates) == 1:
            return candidates[0]

        prefer_vals = [v for vals in intent["prefer_tags"].values() for v in vals]
        avoid_vals  = [v for vals in intent["avoid_tags"].values()  for v in vals]

        best, best_score = candidates[0], float("-inf")
        for path in candidates:
            tags, dist = set(), 0.0
            for u, v in zip(path[:-1], path[1:]):
                data = list(self.graph.get_edge_data(u, v).values())[0]
                dist += data.get("length", 0.0)
                hw = data.get("highway", "")
                tags.update(hw if isinstance(hw, list) else [hw])

            desc = " ".join(tags).lower()
            score = (sum(10 for t in prefer_vals if t in desc)
                     - sum(20 for a in avoid_vals if a in desc)
                     - dist / 1000.0)
            if score > best_score:
                best_score, best = score, path
        return best

    def find_route(self, start_node, end_node, prompt, algorithm=None):
        try:
            intent = self.extractor.extract(prompt, self.tag_schema)
            candidates = self.smc.find_routes(start_node, end_node, intent)
            return self._judge(prompt, candidates, intent)
        except Exception as e:
            print(f"SMCGLiNERRouter error: {e}")
            return None


class SMCLLMRouter(Router):
    """
    SMC particle-filter candidate generation with a local LLaMA model as judge.

    LLaMA runs via Ollama at localhost:11434. If Ollama is unreachable the
    router falls back to returning the first (highest-weight) SMC candidate.
    """

    def __init__(
        self,
        graph,
        bridge: OSMSemanticBridge,
        tag_schema: dict,
        n_particles: int = 50,
        gliner_model_id: str = "urchade/gliner_medium-v2.1",
        llama_model: str = "llama3.2:3b",
        ollama_url: str = "http://localhost:11434/api/generate",
    ):
        super().__init__(graph)
        self.tag_schema = tag_schema
        self.schema_keys = list(tag_schema["discrete"].keys())
        self.extractor = _GLiNERExtractor(gliner_model_id)
        self.smc = SMCRouteGenerator(graph, bridge, n_particles=n_particles)
        self.llama_model = llama_model
        self.ollama_url = ollama_url

    def _build_weight_func(self, prompt: str) -> Callable:
        raise NotImplementedError

    def _llm_judge(self, prompt: str, candidates: List[list], intent: dict) -> Optional[list]:
        if not candidates:
            return None
        if len(candidates) == 1:
            return candidates[0]

        summaries = []
        for i, path in enumerate(candidates):
            dist, path_tags = 0.0, {k: set() for k in self.schema_keys}
            for u, v in zip(path[:-1], path[1:]):
                data = list(self.graph.get_edge_data(u, v).values())[0]
                dist += data.get("length", 0.0)
                for k in self.schema_keys:
                    val = data.get(k)
                    if val:
                        (path_tags[k].update(val) if isinstance(val, list) else path_tags[k].add(val))
            tag_str = ", ".join(f"{k}: {list(v)}" for k, v in path_tags.items() if v)
            summaries.append(f"ID {i} | Distance: {dist/1000:.2f}km | Tags: {tag_str}")

        llm_prompt = (
            f"USER: {prompt}\n"
            f"AVOID: {json.dumps(intent['avoid_tags'])}\n"
            f"PREFER: {json.dumps(intent['prefer_tags'])}\n\n"
            "CANDIDATES:\n" + "\n".join(summaries) + "\n\n"
            "TASK: Return the ID of the best candidate in JSON format."
        )

        try:
            resp = requests.post(
                self.ollama_url,
                json={
                    "model": self.llama_model,
                    "prompt": llm_prompt,
                    "stream": False,
                    "format": {
                        "type": "object",
                        "properties": {"selected_index": {"type": "integer"}},
                        "required": ["selected_index"],
                    },
                    "options": {"temperature": 0, "num_predict": 20},
                },
                timeout=30,
            )
            idx = json.loads(resp.json().get("response", "{}")).get("selected_index", 0)
            return candidates[max(0, min(idx, len(candidates) - 1))]
        except Exception as e:
            print(f"LLM judge error (falling back to top SMC candidate): {e}")
            return candidates[0]

    def find_route(self, start_node, end_node, prompt, algorithm=None):
        try:
            intent = self.extractor.extract(prompt, self.tag_schema)
            candidates = self.smc.find_routes(start_node, end_node, intent)
            return self._llm_judge(prompt, candidates, intent)
        except Exception as e:
            print(f"SMCLLMRouter error: {e}")
            return None
