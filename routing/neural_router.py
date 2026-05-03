from typing import Callable, List

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from routing.router import Router
from routing.adjustor import (
    Adjustor, ADJUSTMENT_CLASSES,
    precompute_tag_embeddings, load_tag_embeddings,
)


class NeuralEdgeRouter(Router):
    """
    Neural router that uses a trained Adjustor model to predict per-edge
    weight multipliers conditioned on a natural language prompt.

    Args:
        graph:                OSMnx/NetworkX MultiDiGraph.
        st_model:             SentenceTransformer instance (frozen encoder).
        model_path:           Path to trained Adjustor checkpoint (.pt).
                              Pass None to use a randomly-initialized model.
        cache_dir:            Directory containing tag_embeddings.npy and edge_index.pkl.
        discrete_keys:        List of discrete tag keys used for embedding.
        similarity_threshold: Cosine similarity floor for pre-filtering (0.0 to disable).
        device:               'cpu' or 'cuda'.
    """

    def __init__(
        self,
        graph,
        st_model: SentenceTransformer,
        model_path: str,
        cache_dir: str,
        discrete_keys: List[str],
        similarity_threshold: float = 0.15,
        device: str = "cpu",
    ):
        super().__init__(graph)
        self.st_model = st_model
        self.device = device
        self.similarity_threshold = similarity_threshold
        self.discrete_keys = discrete_keys

        try:
            self.tag_embeddings, self.edge_index = load_tag_embeddings(cache_dir)
        except FileNotFoundError:
            print("Tag embedding cache not found. Computing...")
            self.tag_embeddings, self.edge_index = precompute_tag_embeddings(
                graph, st_model, discrete_keys, cache_dir
            )

        self.tag_emb_tensor = torch.tensor(
            self.tag_embeddings, dtype=torch.float32, device=device
        )

        self.adjustor = Adjustor()
        if model_path is not None:
            self.adjustor.load_state_dict(
                torch.load(model_path, map_location=device, weights_only=True)
            )
        else:
            print("WARNING: No model checkpoint provided. Using randomly-initialized Adjustor.")
        self.adjustor.to(device)
        self.adjustor.eval()

        self._class_to_mult = [ADJUSTMENT_CLASSES[i] for i in range(len(ADJUSTMENT_CLASSES))]

    def _encode_prompt_tokens(self, prompt: str) -> torch.Tensor:
        encoded = self.st_model.tokenize([prompt])
        encoded = {k: v.to(self.device) for k, v in encoded.items() if isinstance(v, torch.Tensor)}
        with torch.no_grad():
            output = self.st_model[0].auto_model(**encoded)
        token_embeddings = output.last_hidden_state[0]
        # Strip [CLS] and [SEP]
        return token_embeddings[1:-1]

    def _build_weight_func(self, prompt: str) -> Callable[[int, int, dict], float]:
        prompt_tokens = self._encode_prompt_tokens(prompt)
        prompt_centroid = prompt_tokens.mean(dim=0)

        if self.similarity_threshold > 0:
            cos_sim = torch.nn.functional.cosine_similarity(
                self.tag_emb_tensor, prompt_centroid.unsqueeze(0), dim=1
            )
            candidate_mask = cos_sim >= self.similarity_threshold
            candidate_indices = torch.where(candidate_mask)[0]
        else:
            candidate_indices = torch.arange(len(self.edge_index), device=self.device)

        multipliers = np.ones(len(self.edge_index), dtype=np.float32)

        if len(candidate_indices) > 0:
            candidate_tag_embs = self.tag_emb_tensor[candidate_indices]
            type_flags = torch.zeros(
                len(candidate_indices), 1, dtype=torch.float32, device=self.device
            )

            with torch.no_grad():
                logits = self.adjustor(prompt_tokens, candidate_tag_embs, type_flags)
                predicted_classes = logits.argmax(dim=-1).cpu().numpy()

            candidate_idx_np = candidate_indices.cpu().numpy()
            for i, cls_idx in enumerate(predicted_classes):
                multipliers[candidate_idx_np[i]] = self._class_to_mult[cls_idx]

        edge_multipliers = {}
        for i, (u, v, k) in enumerate(self.edge_index):
            if multipliers[i] != 1.0:
                edge_multipliers[(u, v, k)] = float(multipliers[i])

        def weight_func(u, v, edge_dict):
            data = list(edge_dict.values())[0] if isinstance(edge_dict, dict) else edge_dict
            base_cost = data.get("length", 1.0)
            key = next(iter(edge_dict)) if isinstance(edge_dict, dict) else 0
            mult = edge_multipliers.get((u, v, key), 1.0)
            return base_cost * mult

        return weight_func
