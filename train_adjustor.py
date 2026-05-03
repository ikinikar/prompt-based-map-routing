import argparse
import json
import os
import pickle
import random
import re
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer

from routing.adjustor import (
    Adjustor, build_tag_string,
    precompute_tag_embeddings, load_tag_embeddings,
)

NEGATION_RE = re.compile(
    r"\b(avoid|no|without|stay off|keep off|don't|do not|never)\b",
    re.IGNORECASE,
)

DISCRETE_KEYS = ["highway", "access", "bridge", "junction", "ref"]


def detect_negation(prompt: str) -> bool:
    return bool(NEGATION_RE.search(prompt))


def derive_label(edge_data: dict, constraints: dict, is_avoidance: bool) -> int:
    """
    Derive an adjustment class for one edge given a prompt's constraints.

    Returns:
        0 = unchanged, 1 = +40%, 2 = +120%, 3 = -40%, 4 = -80%
    """
    discrete_constraints = {
        k: v for k, v in constraints.items()
        if k in DISCRETE_KEYS and isinstance(v, str)
    }

    if not discrete_constraints:
        return 0

    match_count = 0
    total = len(discrete_constraints)

    for key, target_val in discrete_constraints.items():
        actual = edge_data.get(key)
        if actual is not None:
            actual_vals = actual if isinstance(actual, list) else [str(actual)]
            if target_val in actual_vals:
                match_count += 1

    if match_count == 0:
        if is_avoidance:
            return 0
        return 0

    match_ratio = match_count / total

    if is_avoidance:
        # Edge matches what the user wants to avoid
        if match_ratio >= 0.5:
            return 2  # strong penalty (+120%)
        else:
            return 1  # mild penalty (+40%)
    else:
        # Edge matches what the user prefers
        if match_ratio >= 0.5:
            return 4  # strong reward (-80%)
        else:
            return 3  # mild reward (-40%)


def build_tag_index(graph, edge_index):
    """Build an inverted index: (tag_key, tag_value) -> set of edge indices."""
    index = {}
    for i, (u, v, k) in enumerate(edge_index):
        data = graph.edges[u, v, k]
        for key in DISCRETE_KEYS:
            val = data.get(key)
            if val is not None:
                vals = val if isinstance(val, list) else [str(val)]
                for v_str in vals:
                    pair = (key, v_str)
                    if pair not in index:
                        index[pair] = set()
                    index[pair].add(i)
    return index


def find_matching_edges(tag_index, constraints):
    """Return indices of edges that match at least one discrete constraint."""
    discrete_constraints = {
        k: v for k, v in constraints.items()
        if k in DISCRETE_KEYS and isinstance(v, str)
    }
    if not discrete_constraints:
        return []

    matching = set()
    for key, target_val in discrete_constraints.items():
        hits = tag_index.get((key, target_val), set())
        matching.update(hits)
    return list(matching)


def encode_prompt_tokens(st_model, prompt, device):
    encoded = st_model.tokenize([prompt])
    encoded = {k: v.to(device) for k, v in encoded.items() if isinstance(v, torch.Tensor)}
    with torch.no_grad():
        output = st_model[0].auto_model(**encoded)
    token_embeddings = output.last_hidden_state[0]
    return token_embeddings[1:-1]


def main():
    parser = argparse.ArgumentParser(description="Train the Adjustor model")
    parser.add_argument("--graph", default="research/pioneer_valley_v2.pkl")
    parser.add_argument("--dataset", default="research/synthetic_dataset.jsonl")
    parser.add_argument("--cache-dir", default="models")
    parser.add_argument("--output", default="models/adjustor.pt")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-edges", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    print("Loading graph...")
    with open(args.graph, "rb") as f:
        graph = pickle.load(f)

    print("Loading dataset...")
    with open(args.dataset, "r") as f:
        data = [json.loads(line) for line in f]

    print("Loading SentenceTransformer...")
    st_model = SentenceTransformer("all-MiniLM-L6-v2", device=args.device)

    # Pre-compute or load tag embeddings
    emb_path = os.path.join(args.cache_dir, "tag_embeddings.npy")
    if os.path.exists(emb_path):
        tag_embeddings, edge_index = load_tag_embeddings(args.cache_dir)
    else:
        tag_embeddings, edge_index = precompute_tag_embeddings(
            graph, st_model, DISCRETE_KEYS, args.cache_dir
        )
    tag_emb_tensor = torch.tensor(tag_embeddings, dtype=torch.float32, device=args.device)

    num_edges = len(edge_index)
    all_edge_indices = list(range(num_edges))

    # Build inverted index once, then do O(1) lookups per prompt
    print("Building inverted tag index...")
    tag_index = build_tag_index(graph, edge_index)
    print(f"Indexed {len(tag_index)} (key, value) pairs")

    usable = []
    print("Finding matching edges per prompt...")
    for item in data:
        constraints = item["constraints"]
        matching = find_matching_edges(tag_index, constraints)
        if matching:
            usable.append({"item": item, "matching": matching})
    print(f"Usable prompts: {len(usable)} / {len(data)}")

    # Split by prompt
    random.shuffle(usable)
    split_idx = int(len(usable) * (1 - args.val_split))
    train_set = usable[:split_idx]
    val_set = usable[split_idx:]
    print(f"Train prompts: {len(train_set)}, Val prompts: {len(val_set)}")

    # Initialize model
    adjustor = Adjustor()
    adjustor.to(args.device)
    adjustor.train()

    optimizer = torch.optim.AdamW(adjustor.parameters(), lr=args.lr)

    # Estimate class weights from a sample
    label_counts = Counter()
    sample_size = min(200, len(train_set))
    for entry in train_set[:sample_size]:
        item = entry["item"]
        is_avoid = detect_negation(item["prompt"])
        half = args.batch_edges // 2
        pos_sample = random.choices(entry["matching"], k=min(half, len(entry["matching"])))
        neg_sample = random.choices(all_edge_indices, k=half)
        for idx in pos_sample + neg_sample:
            u, v, k = edge_index[idx]
            label = derive_label(graph.edges[u, v, k], item["constraints"], is_avoid)
            label_counts[label] += 1

    total_labels = sum(label_counts.values())
    class_weights = torch.ones(5, device=args.device)
    for cls, count in label_counts.items():
        if count > 0:
            class_weights[cls] = total_labels / (5 * count)
    print(f"Class weights: {class_weights.tolist()}")

    criterion = nn.CrossEntropyLoss(weight=class_weights)

    best_val_loss = float("inf")
    epochs_no_improve = 0

    for epoch in range(args.epochs):
        adjustor.train()
        random.shuffle(train_set)
        train_loss_sum = 0.0
        train_steps = 0

        for entry in train_set:
            item = entry["item"]
            prompt = item["prompt"]
            constraints = item["constraints"]
            is_avoid = detect_negation(prompt)

            prompt_tokens = encode_prompt_tokens(st_model, prompt, args.device)

            half = args.batch_edges // 2
            pos_sample = random.choices(entry["matching"], k=min(half, len(entry["matching"])))
            neg_count = args.batch_edges - len(pos_sample)
            neg_sample = random.choices(all_edge_indices, k=neg_count)
            batch_indices = pos_sample + neg_sample

            labels = []
            for idx in batch_indices:
                u, v, k = edge_index[idx]
                label = derive_label(graph.edges[u, v, k], constraints, is_avoid)
                labels.append(label)

            batch_tag_embs = tag_emb_tensor[batch_indices]
            type_flags = torch.zeros(len(batch_indices), 1, dtype=torch.float32, device=args.device)
            label_tensor = torch.tensor(labels, dtype=torch.long, device=args.device)

            logits = adjustor(prompt_tokens, batch_tag_embs, type_flags)
            loss = criterion(logits, label_tensor)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item()
            train_steps += 1

        avg_train_loss = train_loss_sum / max(train_steps, 1)

        # Validation
        adjustor.eval()
        val_loss_sum = 0.0
        val_steps = 0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for entry in val_set:
                item = entry["item"]
                prompt = item["prompt"]
                constraints = item["constraints"]
                is_avoid = detect_negation(prompt)

                prompt_tokens = encode_prompt_tokens(st_model, prompt, args.device)

                half = args.batch_edges // 2
                pos_sample = random.choices(entry["matching"], k=min(half, len(entry["matching"])))
                neg_count = args.batch_edges - len(pos_sample)
                neg_sample = random.choices(all_edge_indices, k=neg_count)
                batch_indices = pos_sample + neg_sample

                labels = []
                for idx in batch_indices:
                    u, v, k = edge_index[idx]
                    label = derive_label(graph.edges[u, v, k], constraints, is_avoid)
                    labels.append(label)

                batch_tag_embs = tag_emb_tensor[batch_indices]
                type_flags = torch.zeros(len(batch_indices), 1, dtype=torch.float32, device=args.device)
                label_tensor = torch.tensor(labels, dtype=torch.long, device=args.device)

                logits = adjustor(prompt_tokens, batch_tag_embs, type_flags)
                loss = criterion(logits, label_tensor)

                val_loss_sum += loss.item()
                val_steps += 1

                preds = logits.argmax(dim=-1)
                val_correct += (preds == label_tensor).sum().item()
                val_total += len(label_tensor)

        avg_val_loss = val_loss_sum / max(val_steps, 1)
        val_acc = val_correct / max(val_total, 1)

        print(
            f"Epoch {epoch+1:3d}/{args.epochs} | "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {avg_val_loss:.4f} | "
            f"Val Acc: {val_acc:.4f}"
        )

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            os.makedirs(os.path.dirname(args.output), exist_ok=True)
            torch.save(adjustor.state_dict(), args.output)
            print(f"  -> Saved best model to {args.output}")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= args.patience:
                print(f"Early stopping after {epoch+1} epochs (patience={args.patience})")
                break

    print("Training complete.")


if __name__ == "__main__":
    main()
