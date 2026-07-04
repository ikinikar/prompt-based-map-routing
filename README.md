# Prompt-Based Route Mapping

This project compares six methods for translating a natural-language prompt into a valid route on a real road network, ranging from a plain shortest-path baseline to a cross-attention model that directly adjusts edge weights to a particle-based search to generate candidate paths evaluated by an LLM. 

Ongoing developments are focused on engineering a new method that uses GRPO to fine-tune an LLM to generate weights for A* search, refining the synthetic data, and organizing the repo.

## Overview

Given a prompt such as: "Can you take me from Amherst to Northampton via the on/off ramp?", a prompt-based routing system must:

1. Extract the start and end locations.
2. Identify the constraints implied by the prompt.
3. Produce a route through a real road graph that satisfies those constraints as well as possible, while staying reasonably close to the shortest path.

The road graph is built from OpenStreetMap data for the Pioneer Valley region of Massachusetts (Amherst, Northampton, Springfield, Holyoke, Greenfield, and Westfield), and all methods share a common graph representation and evaluation suite so that results are directly comparable.

## Research Questions

1. What is the best way to navigate routes based on a prompt that contains constraints?
2. Can neurosymbolic approaches for prompt-based map routing significantly improve upon traditional search?
3. Can resource-constrained prompt-based map routing approaches work as well as resource-intensive approaches?

## Data

### Graph

The graph was constructed with the OSMnx Python package, pulling data from OpenStreetMap for the Pioneer Valley towns and their connected roads.

| Property | Value |
|---|---|
| Nodes | 3400 |
| Edges | 8850 |
| Cities covered | 6 (Amherst, Northampton, Springfield, Holyoke, Greenfield, Westfield) |

Edges carry four numerical tags (max speed, number of lanes, length, width) and five discrete tags (highway type, accessibility, bridge status, junction type, road name). Missing max speed values were imputed using standardized Massachusetts speed limits based on highway type.

### Prompts

A synthetic dataset of prompts was generated using the open-source `llama3.1-8b` model via the Cerebras API. Each prompt encodes 1 to 3 constraints on a route between a start and end location in the Pioneer Valley.

| Property | Value |
|---|---|
| Total prompts | 5753 |
| Constraints per prompt | 1 to 3 |
| Sentences per prompt | 1 to 2 |

The dataset is stored as `synthetic_dataset.jsonl` under `research/`, with each line following this schema:

| Field | Type | Description |
|---|---|---|
| `id` | int | Line number of the data entry |
| `start` | str | Start node of the query |
| `end` | str | End node of the query |
| `prompt` | str | Natural language navigation request |
| `constraints` | json | Edge constraints implied by the prompt |
| `sampled_context` | json | The tag context originally sampled to generate the prompt |

Care was taken to avoid evaluating any method that shares its underlying model with the one used to generate the data.

## Shared Components

All methods below build on a few shared pieces:

- **OSMSemanticBridge (synset class)**: groups OSMnx tags into synsets when their cosine similarity (via `all-MiniLM-L6-v2`) exceeds 0.8, so that semantically equivalent tags (for example, "motorway" and "highway") are not penalized as mismatches.
- **GLiNER-medium-v2.1**: a 209M-parameter named entity recognition model from Fastino Labs used by several methods to extract constraints from prompts. It runs on a CPU with sub-100ms latency and produces deterministic outputs.
- **Sequential Monte Carlo (SMC)**: a particle-based search that starts with 100 particles at the origin node and proposes moves at each step, biased toward progress along a naive A* route. Particles are resampled every 20 steps based on accumulated path weight, and the top 3 paths by weight are returned after 500 steps.

## Methods

| Method | Description |
|---|---|
| Pure A* | Shortest path search using Great-circle distance as the heuristic. No constraint handling. |
| Keyword Matching + A* | Regex rules map phrases to OSM tag categories and polarity (prefer/avoid), which adjust edge weights before running A*. |
| GLiNER + A* | GLiNER extracts "road_type_to_avoid" and "amenity_required" constraints, which are mapped through the synset class into edge weight adjustments, then A* runs on the reweighted graph. |
| Neural Edge Editing | A cross-attention plus MLP classifier predicts one of five weight adjustment classes for each edge/node, based on the prompt embedding and tag embeddings. |
| GLiNER + SMC | GLiNER extracts "avoid_feature" and "preferred_feature" constraints to weight SMC's stochastic resampling. Final candidate paths are judged by keyword matching alone. |
| SMC Pipeline + Llama Judge | Same as GLiNER + SMC, but the final judging step is replaced with Llama 3.2-3B (served via Ollama), using constrained JSON-mode decoding with a 20-token limit to select the best of three candidate paths. |

### Neural Edge Editing adjustment classes

| Class | Effect |
|---|---|
| 0 | Leave edge weight unchanged |
| 1 | Increase edge weight by 40% |
| 2 | Increase edge weight by 120% |
| 3 | Decrease edge weight by 40% |
| 4 | Decrease edge weight by 80% |

Training labels for this model are heuristically generated from the match ratio between prompt-derived constraint tags and edge tags, combined with a detected preference or avoidance intent.

## Custom Evaluation Suite

| Metric | Definition |
|---|---|
| Path Validity | Whether the path is a valid, connected route on the graph between the specified start and end nodes |
| Deviation Penalty | Ratio of chosen path length to shortest possible path length (always greater than 1; lower is more efficient) |
| Constraint Satisfaction | Average satisfaction of prompt constraints, matched via the synset class with a cosine similarity fallback using `multi-qa-mpnet-base-dot-v1` (higher is better) |
| Semantic Alignment | BERTScore F1 between the prompt and the collated tags along the path (higher is better) |

Constraint extraction for ground truth uses negation marking and part-of-speech tagging, kept independent from the extraction methods used inside the routing pipelines themselves to avoid circularity.

## Results

All metrics were averaged across 100 out-of-distribution prompts.

| Method | Path Validity | Deviation Penalty (lower is better) | Constraint Satisfaction (higher is better) | Semantic Alignment (higher is better) |
|---|---|---|---|---|
| Pure A* | 1.0000 | 1.0006 | 0.4215 | 0.7935 |
| Keyword + A* | 1.0000 | 1.1606 | 0.4454 | 0.7923 |
| GLiNER + A* | 1.0000 | 1.0028 | 0.4245 | 0.7914 |
| Neural Edge Editing | 1.0000 | 1.1199 | 0.4468 | 0.7910 |
| GLiNER + SMC | 1.0000 | 1.4678 | 0.4420 | 0.7917 |
| SMC Pipeline + Llama Judge | 1.0000 | 1.9722 | 0.4669 | 0.7947 |

All methods produced valid paths in every case, and semantic alignment scores were close enough across methods that they were not treated as a primary point of comparison. The main tradeoff observed is between deviation penalty and constraint satisfaction: methods that deviate further from the shortest path tend to satisfy more constraints.

Using the Pareto front over deviation penalty and constraint satisfaction, four methods are non-dominated and considered optimal: Pure A*, GLiNER + A*, Neural Edge Editing, and SMC Pipeline + Llama Judge. Keyword + A* is strictly dominated by Neural Edge Editing, and GLiNER + SMC is strictly dominated by both Neural Edge Editing and Keyword + A*.

## Repository Structure

```
.
├── GRPO-Tuned-LLM-A-star.ipynb
├── README.md
├── SMC-GLiNER-router.ipynb
├── SMC-LLM-router.ipynb
├── cache/                          # cached intermediate results
├── chart.ipynb
├── eval
│   └── evaluation.py               # evaluation suite (path validity, deviation penalty, constraint satisfaction, semantic alignment)
├── keyword-test.ipynb
├── models
│   ├── adjustor.pt                 # trained Neural Edge Editing model weights
│   └── edge_index.pkl
├── ner-A-star-test.ipynb
├── neural-test.ipynb
├── pure-A-star.ipynb
├── research
│   ├── cache/
│   ├── osmnx_test.ipynb
│   ├── pioneer_valley.pkl          # cached graph
│   ├── pioneer_valley_v2.pkl       # cached graph, revised
│   ├── synthetic_data.ipynb        # dataset generation and preprocessing
│   └── synthetic_dataset.jsonl     # 5753 synthetic prompts
├── routing
│   ├── NER.py                      # GLiNER-based constraint extraction
│   ├── adjustor.py                 # Neural Edge Editing model definition
│   ├── keyword_router.py           # regex-based keyword matching router
│   ├── neural_router.py            # Neural Edge Editing routing pipeline
│   ├── router.py                   # shared routing utilities (A*, SMC)
│   └── synset.py                   # OSMSemanticBridge synset class
├── tradeoff.png                    # deviation penalty vs constraint satisfaction plot
└── train_adjustor.py                # training script for Neural Edge Editing
```

## Future Work

- Increase prompt variety and diversity, clean up any problematic prompts
- Add queries requiring geospatial awareness, such as proximity to a landmark not directly on the graph
- Quantify the resource usage of each method directly, since resource efficiency was a research question but was not measured explicitly

## Authors

Ishan Kinikar, Archimedes Li, Vinayak Rao
