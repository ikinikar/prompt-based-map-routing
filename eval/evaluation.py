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

class prompt_based_route_evaluator:
    def __init__(self, graph, prompts: List[str], routes: List[List[int]], device: str = "cpu"):
        self.graph = graph
        self.prompts = prompts
        self.routes = routes
        self.device = device
        # Critic Model: Independent from the Actor (MiniLM)
        self.st_evaluator = SentenceTransformer('sentence-transformers/multi-qa-mpnet-base-dot-v1', device=device)

    def check_path_validity(self, route: List[int]) -> bool:
        """Verifies if the path is physically traversable in the graph."""
        if not route or len(route) < 2: return False
        return all(self.graph.has_edge(u, v) for u, v in zip(route[:-1], route[1:]))

    def deviation_penalty(self, route: List[int]) -> float:
        """Calculates how much longer the route is compared to the shortest physical path."""
        if not self.check_path_validity(route): return 10.0
        start, end = route[0], route[-1]
        try:
            ideal_len = nx.shortest_path_length(self.graph, start, end, weight='length')
            actual_len = sum(self.graph[u][v][0].get('length', 1.0) for u, v in zip(route[:-1], route[1:]))
            return actual_len / ideal_len if ideal_len > 0 else 1.0
        except: 
            return 10.0

    def _get_path_metadata_string(self, route: List[int]) -> str:
        """Flattens unique path tags into a 'semantic sentence' for BERTScore."""
        all_tags = []
        for u, v in zip(route[:-1], route[1:]):
            data = self.graph.get_edge_data(u, v)[0]
            for key in ['name', 'highway', 'amenity', 'shop', 'leisure', 'landuse']:
                val = data.get(key)
                if val:
                    if isinstance(val, list): 
                        all_tags.extend([str(v).replace('_', ' ') for v in val])
                    else: 
                        all_tags.append(str(val).replace('_', ' '))
        
        # Use set to keep the 'sentence' concise but representative
        return " ".join(list(set(all_tags))).lower()

    def semantic_alignment_bertscore(self) -> float:
        """
        Calculates BERTScore (F1) between the prompt and the path metadata.
        This captures the global 'vibe' match.
        """
        if not self.routes: return 0.0
        
        # Candidate: The flattened tags of the generated route
        candidates = [self._get_path_metadata_string(r) for r in self.routes]
        # Reference: The original user prompt
        references = self.prompts
        
        # bert_score returns P, R, and F1 tensors
        _, _, f1 = bert_score_fn(candidates, references, lang="en", device=self.device, verbose=False)
        return f1.mean().item()

    def constraint_satisfaction(self, prompt: str, route: List[int]) -> float:
        nlp = spacy.load("en_core_web_sm")
        doc = nlp(prompt.lower())
        path_metadata = self._get_path_metadata_string(route)
        if not path_metadata: return 0.0

        # 1. Instruction Embeddings
        # We compare verbs to these "Anchor Concepts" to determine polarity
        avoid_concept = self.st_evaluator.encode("avoid exclude shun shunned away from", convert_to_tensor=True)
        
        semantic_constraints = []
        
        # 2. Extract Noun Chunks (Grammatically intact)
        for chunk in doc.noun_chunks:
            # Punctuation cleaning while preserving geographic IDs (e.g., 'I-91')
            clean_text = " ".join(re.findall(r"[\w-]+", chunk.text))
            
            # STOPWORD REMOVAL (Performed AFTER chunking to maintain structure)
            # Only discard the chunk if it contains *only* stopwords
            stop_words = set(stopwords.words('english'))
            if all(w in stop_words for w in clean_text.split()):
                continue

            # 3. Dynamic Polarity Check (No hardcoding)
            # Check A: Linguistic Negation (e.g., "no", "not", "never")
            is_negated = any(t.dep_ == "neg" or t.lemma_ in ["no", "not"] for t in chunk)
            
            # Check B: Semantic Negation via the Head Verb
            head_verb = chunk.root.head
            if head_verb.pos_ == "VERB":
                verb_emb = self.st_evaluator.encode(head_verb.lemma_, convert_to_tensor=True)
                # If verb is semantically closer to "avoid" than "take", it's negated
                if torch.cosine_similarity(verb_emb.unsqueeze(0), avoid_concept.unsqueeze(0)) > 0.6:
                    is_negated = True

            semantic_constraints.append({
                "text": clean_text,
                "negated": is_negated
            })

        if not semantic_constraints:
            return 1.0

        # 4. Evaluation
        total_satisfaction = 0
        for c in semantic_constraints:
            c_emb = self.st_evaluator.encode(c['text'], convert_to_tensor=True)
            p_emb = self.st_evaluator.encode(path_metadata, convert_to_tensor=True)
            
            similarity = float(torch.cosine_similarity(c_emb.unsqueeze(0), p_emb.unsqueeze(0)).item())
            
            # Logical Satisfaction
            # If negated, we want LOW similarity. If not, we want HIGH similarity.
            total_satisfaction += (1.0 - similarity) if c['negated'] else similarity

        return total_satisfaction / len(semantic_constraints)

    def evaluate_method(self) -> Dict:
        """Runs the full suite of metrics across the generated routes."""
        if not self.routes:
            return {"error": "No routes to evaluate"}

        # Run semantic alignment across the whole batch
        avg_semantic_align = self.semantic_alignment_bertscore()
        
        return {
            "avg_path_validity": np.mean([self.check_path_validity(r) for r in self.routes]),
            "avg_deviation_penalty": np.mean([self.deviation_penalty(r) for r in self.routes]),
            "avg_constraint_satisfaction": np.mean([self.constraint_satisfaction(p, r) for p, r in zip(self.prompts, self.routes)]),
            "avg_semantic_alignment_f1": avg_semantic_align
        }