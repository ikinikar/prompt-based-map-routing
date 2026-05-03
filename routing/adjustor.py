import os
import pickle
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer


ADJUSTMENT_CLASSES = {0: 1.0, 1: 1.4, 2: 2.2, 3: 0.6, 4: 0.2}
NUM_CLASSES = 5
EMBED_DIM = 384
ATTN_DIM = 64


class Adjustor(nn.Module):

    def __init__(self, embed_dim: int = EMBED_DIM, attn_dim: int = ATTN_DIM,
                 num_classes: int = NUM_CLASSES):
        super().__init__()
        self.W_q = nn.Linear(embed_dim, attn_dim)
        self.W_k = nn.Linear(embed_dim, attn_dim)
        self.W_v = nn.Linear(embed_dim, attn_dim)
        self.scale = attn_dim ** 0.5

        mlp_input_dim = attn_dim + embed_dim + 1
        self.mlp = nn.Sequential(
            nn.Linear(mlp_input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes),
        )

    def forward(self, prompt_tokens: torch.Tensor, tag_embs: torch.Tensor,
                type_flags: torch.Tensor) -> torch.Tensor:
        Q = self.W_q(tag_embs)
        K = self.W_k(prompt_tokens)
        V = self.W_v(prompt_tokens)

        scores = torch.matmul(Q, K.T) / self.scale
        weights = F.softmax(scores, dim=-1)
        attended = torch.matmul(weights, V)

        features = torch.cat([attended, tag_embs, type_flags], dim=-1)
        return self.mlp(features)

    def predict(self, prompt_tokens: torch.Tensor, tag_embs: torch.Tensor,
                type_flags: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            logits = self.forward(prompt_tokens, tag_embs, type_flags)
            return logits.argmax(dim=-1)


def build_tag_string(edge_data: dict, discrete_keys: List[str]) -> str:
    parts = []
    for key in discrete_keys:
        val = edge_data.get(key)
        if val is not None:
            if isinstance(val, list):
                parts.extend(str(v).lower().replace('_', ' ') for v in val)
            else:
                parts.append(str(val).lower().replace('_', ' '))
    return " ".join(parts) if parts else "unknown"


def precompute_tag_embeddings(
    graph,
    st_model: SentenceTransformer,
    discrete_keys: List[str],
    cache_dir: str,
    batch_size: int = 512,
) -> Tuple[np.ndarray, List[Tuple[int, int, int]]]:
    os.makedirs(cache_dir, exist_ok=True)
    emb_path = os.path.join(cache_dir, "tag_embeddings.npy")
    idx_path = os.path.join(cache_dir, "edge_index.pkl")

    edge_index = []
    tag_strings = []
    for u, v, key, data in graph.edges(keys=True, data=True):
        edge_index.append((u, v, key))
        tag_strings.append(build_tag_string(data, discrete_keys))

    print(f"Encoding {len(tag_strings)} edge tag strings...")
    embeddings = st_model.encode(
        tag_strings, batch_size=batch_size,
        show_progress_bar=True, convert_to_numpy=True,
    )

    np.save(emb_path, embeddings)
    with open(idx_path, "wb") as f:
        pickle.dump(edge_index, f)

    print(f"Saved tag embeddings: {emb_path} (shape {embeddings.shape})")
    print(f"Saved edge index: {idx_path} ({len(edge_index)} entries)")
    return embeddings, edge_index


def load_tag_embeddings(cache_dir: str) -> Tuple[np.ndarray, List[Tuple[int, int, int]]]:
    emb_path = os.path.join(cache_dir, "tag_embeddings.npy")
    idx_path = os.path.join(cache_dir, "edge_index.pkl")

    embeddings = np.load(emb_path)
    with open(idx_path, "rb") as f:
        edge_index = pickle.load(f)

    print(f"Loaded tag embeddings: shape {embeddings.shape}")
    return embeddings, edge_index
