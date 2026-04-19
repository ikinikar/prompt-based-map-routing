import networkx as nx
import numpy as np
import torch
from typing import Dict, List
from bert_score import score
from sentence_transformers import SentenceTransformer, util

class prompt_based_route_evaluator:
    def __init__(self, graph, prompts: List[str], routes: List[List[int]], device: str = "cpu"):
        self.graph = graph
        self.prompts = prompts
        self.routes = routes
        # SWAP: Using a larger, QA-tuned model for independent verification
        self.st_evaluator = SentenceTransformer('sentence-transformers/multi-qa-mpnet-base-dot-v1', device=device)

    def check_path_validity(self, route: List[int]) -> bool:
        if not route or len(route) < 2: return False
        return all(self.graph.has_edge(u, v) for u, v in zip(route[:-1], route[1:]))

    def deviation_penalty(self, route: List[int]) -> float:
        if not self.check_path_validity(route): return 10.0
        start, end = route[0], route[-1]
        try:
            ideal_len = nx.shortest_path_length(self.graph, start, end, weight='length')
            actual_len = sum(self.graph[u][v][0].get('length', 1.0) for u, v in zip(route[:-1], route[1:]))
            return actual_len / ideal_len if ideal_len > 0 else 1.0
        except: return 10.0

    def constraint_satisfaction(self, prompt: str, route: List[int]) -> float:
        words = [w.lower() for w in prompt.split() if len(w) > 3]
        if not words: return 1.0
        
        path_tags = []
        for u, v in zip(route[:-1], route[1:]):
            data = self.graph.get_edge_data(u, v)[0]
            for key in ['name', 'highway', 'amenity', 'shop', 'leisure', 'landuse']:
                val = data.get(key)
                if val: path_tags.extend(val if isinstance(val, list) else [str(val)])
        
        if not path_tags: return 0.0
        
        c_embs = self.st_evaluator.encode(words, convert_to_tensor=True)
        t_embs = self.st_evaluator.encode(list(set(path_tags)), convert_to_tensor=True)
        scores = util.cos_sim(c_embs, t_embs)
        return torch.sum(torch.max(scores, dim=1)[0] > 0.60).item() / len(words)

    def evaluate_method(self) -> Dict:
        return {
            "avg_path_validity": np.mean([self.check_path_validity(r) for r in self.routes]),
            "avg_deviation_penalty": np.mean([self.deviation_penalty(r) for r in self.routes]),
            "avg_constraint_satisfaction": np.mean([self.constraint_satisfaction(p, r) for p, r in zip(self.prompts, self.routes)])
        }