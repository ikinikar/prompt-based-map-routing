import numpy as np
import torch
import networkx as nx
from typing import List, Dict
from sentence_transformers import SentenceTransformer, util
from bert_score import score as bert_score_fn
import nltk
from nltk.corpus import stopwords
import spacy
import re
from routing.synset import OSMSemanticBridge

class prompt_based_route_evaluator:
    def __init__(self, graph, prompts: List[str], routes: List[List[int]], bridge, device: str = "cpu"):
        self.graph = graph
        self.prompts = prompts
        self.routes = routes
        self.bridge = bridge
        self.device = device
        
        # 1. Critic Model: Independent from Actor's MiniLM
        self.st_evaluator = SentenceTransformer('sentence-transformers/multi-qa-mpnet-base-dot-v1', device=device)
        
        # 2. NLP Infrastructure
        self.nlp = spacy.load("en_core_web_sm")
            
        self.stop_words = set(stopwords.words('english'))
        
        # 3. Pre-cached Polarity Anchor
        self.avoid_concept = self.st_evaluator.encode(
            "avoid exclude shun shunned away from bypass miss without", 
            convert_to_tensor=True
        )

    def check_path_validity(self, route: List[int]) -> bool:
        """Verifies if the path is physically traversable in the graph."""
        if not route or len(route) < 2: return False
        return all(self.graph.has_edge(u, v) for u, v in zip(route[:-1], route[1:]))

    def deviation_penalty(self, route: List[int]) -> float:
        """Calculates length ratio between generated route and shortest physical path."""
        if not self.check_path_validity(route): return 10.0
        start, end = route[0], route[-1]
        try:
            # ideal_len uses pure physical length
            ideal_len = nx.shortest_path_length(self.graph, start, end, weight='length')
            # actual_len sums physical length of the generated route
            actual_len = 0
            for u, v in zip(route[:-1], route[1:]):
                edge_data = self.graph.get_edge_data(u, v)
                data = list(edge_data.values())[0] if isinstance(edge_data, dict) else edge_data
                actual_len += data.get('length', 1.0)
            
            return actual_len / ideal_len if ideal_len > 0 else 1.0
        except: 
            return 10.0

    def _get_path_tags_list(self, route: List[int]) -> List[str]:
        """Extracts unique discrete tags based on the bridge's dynamic schema."""
        tags = set()
        schema_keys = self.bridge.categories
        for u, v in zip(route[:-1], route[1:]):
            edge_data = self.graph.get_edge_data(u, v)
            if not edge_data: continue
            data = list(edge_data.values())[0] if isinstance(edge_data, dict) else edge_data
            
            for key in schema_keys:
                val = data.get(key)
                if val:
                    if isinstance(val, list):
                        tags.update([str(v).lower().replace('_', ' ') for v in val])
                    else:
                        tags.add(str(val).lower().replace('_', ' '))
        return list(tags)

    def _get_path_metadata_string(self, route: List[int]) -> str:
        """Flattens all relevant tags (schema + names) into a semantic sentence."""
        # 1. Start with schema-grounded tags
        all_tags = self._get_path_tags_list(route)
        
        # 2. Augment with names and descriptive tags for BERTScore global vibe
        for u, v in zip(route[:-1], route[1:]):
            edge_data = self.graph.get_edge_data(u, v)
            data = list(edge_data.values())[0] if isinstance(edge_data, dict) else edge_data
            for key in ['name', 'amenity', 'shop', 'leisure', 'landuse']:
                val = data.get(key)
                if val:
                    if isinstance(val, list):
                        all_tags.extend([str(v).lower().replace('_', ' ') for v in val])
                    else:
                        all_tags.append(str(val).lower().replace('_', ' '))
        
        return " ".join(list(set(all_tags))).lower()

    def semantic_alignment_bertscore(self) -> float:
        """Calculates global F1 alignment between prompt and path metadata."""
        if not self.routes: return 0.0
        candidates = [self._get_path_metadata_string(r) for r in self.routes]
        references = self.prompts
        
        _, _, f1 = bert_score_fn(candidates, references, lang="en", device=self.device, verbose=False)
        return f1.mean().item()

    def constraint_satisfaction(self, prompt: str, route: List[int]) -> float:
        """Fine-grained satisfaction using dependency logic and schema synsets."""
        doc = self.nlp(prompt.lower())
        path_tags = self._get_path_tags_list(route)
        path_metadata_str = self._get_path_metadata_string(route)
        if not path_metadata_str: return 0.0

        semantic_constraints = []
        for chunk in doc.noun_chunks:
            clean_text = " ".join(re.findall(r"[\w-]+", chunk.text))
            if all(w in self.stop_words for w in clean_text.split()):
                continue

            # Check for linguistic negation ('no', 'not')
            is_negated = any(t.dep_ == "neg" or t.lemma_ in ["no", "not"] for t in chunk)
            
            # Check for semantic negation in head verb (e.g., 'avoid')
            head_verb = chunk.root.head
            if head_verb.pos_ == "VERB":
                verb_emb = self.st_evaluator.encode(head_verb.lemma_, convert_to_tensor=True)
                if torch.cosine_similarity(verb_emb.unsqueeze(0), self.avoid_concept.unsqueeze(0)) > 0.6:
                    is_negated = True

            semantic_constraints.append({"text": clean_text, "negated": is_negated})

        if not semantic_constraints: return 1.0

        total_satisfaction = 0
        for c in semantic_constraints:
            # 1. Symbolic: Check synset mapping from the bridge
            synset_map = self.bridge.get_osm_synsets(c['text'])
            target_tags = [val for values in synset_map.values() for val in values]
            has_hard_match = any(tag in path_tags for tag in target_tags)
            
            # 2. Neural: Cosine similarity fallback
            c_emb = self.st_evaluator.encode(c['text'], convert_to_tensor=True)
            p_emb = self.st_evaluator.encode(path_metadata_str, convert_to_tensor=True)
            similarity = float(torch.cosine_similarity(c_emb.unsqueeze(0), p_emb.unsqueeze(0)).item())

            current_score = 1.0 if has_hard_match else similarity

            # 3. Polarity Logic
            if c['negated']:
                total_satisfaction += (1.0 - current_score)
            else:
                total_satisfaction += current_score

        return total_satisfaction / len(semantic_constraints)

    def evaluate_method(self) -> Dict:
        """Aggregates all metrics for the batch."""
        if not self.routes: return {"error": "No routes to evaluate"}

        # BERTScore is computed batch-wise for GPU efficiency
        avg_semantic_align = self.semantic_alignment_bertscore()
        
        return {
            "avg_path_validity": float(np.mean([self.check_path_validity(r) for r in self.routes])),
            "avg_deviation_penalty": float(np.mean([self.deviation_penalty(r) for r in self.routes])),
            "avg_constraint_satisfaction": float(np.mean([self.constraint_satisfaction(p, r) for p, r in zip(self.prompts, self.routes)])),
            "avg_semantic_alignment_f1": avg_semantic_align
        }