# Temporal Subgraph Contrastive Learning for Orthrus

## Goal

Add a temporal contrastive objective on top of Orthrus by aligning node embeddings
between two consecutive time-window subgraphs `G_t` and `G_{t+1}`.

The objective is auxiliary. The main Orthrus task remains edge-type prediction.

## Motivation

Orthrus already trains on time-windowed provenance subgraphs. Consecutive windows
contain overlapping entities whose behavior should evolve smoothly under normal
conditions. A temporal contrastive loss can regularize the encoder so that:

- shared nodes keep stable representations across adjacent windows
- abnormal temporal drift becomes easier for the main detection objective to expose
- the encoder learns continuity without replacing Orthrus' anomaly score definition

## MVP Design

This repository currently trains one graph at a time. To avoid rewriting the
data-loading pipeline, the first implementation uses a cached previous window:

1. Train on the current subgraph `G_t` with the standard Orthrus loss.
2. Cache the current node embeddings and `original_n_id`.
3. When `G_{t+1}` arrives, match shared nodes between the cached state and the
   current state.
4. Apply an InfoNCE loss over matched node pairs.

This is intentionally asymmetric:

- gradients flow through the current window encoder output
- the previous window embeddings are detached from the computation graph

That keeps the implementation simple and compatible with the existing training
loop, while still enforcing temporal consistency.

## Positive and Negative Pairs

- Positive pairs: the same original node ID appearing in both `G_t` and `G_{t+1}`
- Negative pairs: other matched nodes in the same minibatch pair

The first version only uses shared node IDs. It does not attempt graph-level
alignment or hard-negative mining yet.

## Loss

For aligned embeddings `h_i^t` and `h_i^{t+1}`:

1. Project both through a small MLP head.
2. L2-normalize projected embeddings.
3. Compute a symmetric InfoNCE loss over the similarity matrix.

Total loss:

`L = L_edge_type + lambda * L_temporal_contrastive`

Recommended initial values:

- `lambda = 0.1`
- `temperature = 0.2`
- `projection_dim = node_out_dim`
- `warmup_epochs = 2`

## Implementation Notes

### Config

Add `training.temporal_contrastive`:

- `enabled`
- `loss_weight`
- `temperature`
- `projection_dim`
- `warmup_epochs`
- `max_pairs_per_batch`

### Model

Add:

- a temporal projection head
- a helper to package temporal state from the current graph
- a helper to compute temporal InfoNCE from current and cached previous states

### Training Loop

For each dataset sequence inside an epoch:

- reset `prev_temporal_state = None`
- after each graph forward pass, compute temporal loss against the previous state
- update the cached state with the current graph

The cache is reset between datasets and epochs.

## Why This MVP First

This version is the lowest-risk path because it:

- does not change batching or dataset interfaces
- preserves current Orthrus training and inference behavior
- introduces only one extra cached state
- lets us measure whether temporal consistency helps before adding more machinery

## Planned Next Steps

If this improves validation behavior, the next upgrades should be:

1. Pair-aware dataloading that explicitly yields `(G_t, G_{t+1})`
2. Hard negatives, especially same-type different-ID nodes
3. Multi-window positives such as `(t-1, t)` and `(t, t+1)`
4. Optional temporal drift scoring during inference
