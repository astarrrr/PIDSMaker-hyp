# HT-ORTHRUS Design

## Overview

This document summarizes the design of the proposed HT-ORTHRUS model built on top of ORTHRUS.

HT-ORTHRUS extends ORTHRUS along two main axes:

1. Hyperbolic structural encoding for hierarchical provenance propagation.
2. Anchor-centered tri-temporal contrastive learning with adaptive temporal weighting.

The target remains the original ORTHRUS task: edge-type prediction on temporal provenance subgraphs.

## Motivation

ORTHRUS already operates on time-windowed provenance subgraphs and uses a graph encoder plus an edge-type prediction objective. This is effective, but it has two limitations:

1. Euclidean message passing is not ideal for tree-like or hierarchical attack propagation.
2. Adjacent windows are processed mostly independently, so temporal continuity is only weakly modeled.

HT-ORTHRUS addresses these issues by combining hyperbolic geometry and temporal consistency regularization.

## High-Level Architecture

For three consecutive subgraphs `G_{t-1}`, `G_t`, and `G_{t+1}`:

1. Each graph is encoded by a shared hyperbolic graph encoder.
2. Node embeddings are produced in Lorentz space.
3. Embeddings are mapped back to Euclidean tangent space for the ORTHRUS decoder and temporal projector.
4. The main edge-type prediction loss is computed on the current graph.
5. A tri-temporal contrastive loss is computed with `t` as the anchor once `G_{t+1}` is available.

Overall objective:

`L = L_edge + lambda * L_tri-temp`

## Module 1: Hyperbolic Structural Encoder

### Goal

Replace the Euclidean graph attention encoder with a Lorentz hyperbolic encoder so the model can better represent:

- hierarchical propagation
- tree-like attack paths
- non-Euclidean structural growth in provenance graphs

### Design Choice

Instead of using a naive hyperbolic distance softmax attention, HT-ORTHRUS follows the Lorentz-style design used in the reference hyperbolic transformer implementation:

- Euclidean input features are mapped into Lorentz space
- query, key, and value projections are performed in hyperbolic geometry
- attention scores use Lorentz inner products
- aggregation is performed in tangent space and projected back to Lorentz space
- the final node embedding is mapped back to Euclidean tangent space with `logmap0`

### Encoder Pipeline

For node feature `x_i`:

1. `x_i` is projected into Lorentz space:
   `z_i^(0) = HypLinear(x_i)`
2. Hyperbolic attention layers propagate information over graph edges.
3. The resulting Lorentz embedding is mapped back:
   `h_i = log_0(z_i)`
4. `h_i` is used by the ORTHRUS decoder.

### Attention Form

For edge `(j -> i)`:

- `q_i = W_Q^H(z_i)`
- `k_j = W_K^H(z_j)`
- `v_j = W_V^H(z_j)`

Attention score:

`e_ij = (2 + 2 <q_i, k_j>_L) / tau + beta^T phi(edge_ij)`

Normalized attention:

`alpha_ij = softmax_j(e_ij)`

Aggregation:

`m_i = HyperbolicAggregate({v_j}, {alpha_ij})`

This preserves edge-feature influence, which is important because ORTHRUS is trained for edge-type prediction.

## Module 2: Anchor-Centered Tri-Temporal Contrastive Learning

### Goal

Encourage node embeddings to evolve smoothly across adjacent windows while keeping the existing ORTHRUS training pipeline mostly intact.

### Why Anchor-Centered

The current training loop processes one graph at a time in sequence. That means `G_t` and `G_{t-1}` are naturally available together, but `G_{t+1}` is only available one step later.

Therefore, HT-ORTHRUS uses an anchor-centered delayed computation:

- when `G_{t+1}` arrives, the model computes the temporal loss centered on `G_t`
- this uses cached states for `G_{t-1}` and `G_t`

### Positive Pairs

Positive pairs are shared original node IDs across adjacent windows:

- `(t-1, t)`
- `(t, t+1)`

If node `v` appears in both windows, then:

- `(h_v^{t-1}, h_v^t)` is a positive pair
- `(h_v^t, h_v^{t+1})` is a positive pair

Negatives are other matched nodes in the same batch pair, as in standard InfoNCE.

### Projection and Similarity

For Euclidean embeddings `h_v^t`, a projector `g(.)` is applied:

`u_v^t = normalize(g(h_v^t))`

For one window pair `(a, b)`, the pairwise similarity matrix is:

`S = U^a (U^b)^T / T`

and the loss is symmetric InfoNCE:

`L_{a,b} = 1/2 * (CE(S, I) + CE(S^T, I))`

### Tri-Temporal Loss

For center window `t`:

`L_tri-temp^(t) = w_{t-1,t} * L_{t-1,t} + w_{t,t+1} * L_{t,t+1}`

where the weights are adaptive and normalized.

## Module 3: Adaptive Temporal Weighting

### Goal

Not all adjacent windows are equally reliable. Some windows share many nodes and reflect stable behavior; others have weak overlap and are noisier.

HT-ORTHRUS uses two weighting levels:

1. Window-level consistency weight
2. Node-level stability weight

### Window-Level Weight

For two adjacent windows:

`a_{t-1,t} = |V_{t-1} ∩ V_t| / |V_{t-1} ∪ V_t|`

`a_{t,t+1} = |V_t ∩ V_{t+1}| / |V_t ∪ V_{t+1}|`

To avoid vanishing temporal regularization, a floor is applied:

`a' = max(a, epsilon)`

Then the two pair weights are normalized:

`w_{t-1,t} = a'_{t-1,t} / (a'_{t-1,t} + a'_{t,t+1})`

`w_{t,t+1} = a'_{t,t+1} / (a'_{t-1,t} + a'_{t,t+1})`

### Node-Level Weight

For node `v`, let `c_v` be the number of windows in `{t-1, t, t+1}` in which it appears.

`b_v = c_v / 3`

To avoid suppressing unstable but potentially important nodes too aggressively, a floor-smoothed version is used:

`b'_v = gamma + (1 - gamma) * b_v`

### Final Weighted Temporal Loss

For center time `t`:

`L_tri-temp^(t) = sum_{v in P_{t-1,t}} w_{t-1,t} * b'_v * l(v; t-1, t) + sum_{v in P_{t,t+1}} w_{t,t+1} * b'_v * l(v; t, t+1)`

where `l(v; a, b)` denotes the node-level symmetric InfoNCE contribution.

## Training Procedure

The training loop processes graphs sequentially.

### State Cache

Maintain a queue of length two:

- `S_{t-1}`
- `S_t`

Each state stores:

- `original_n_id`
- node embeddings for temporal contrastive learning

### Online Training Logic

For each graph `G_t`:

1. Encode `G_t`
2. Compute `L_edge` on the current graph
3. If only one previous state exists, compute a bi-temporal fallback loss
4. If two previous states exist, compute the tri-temporal loss centered on the middle graph
5. Update the cache
6. Backpropagate `L_edge + lambda * L_tri-temp`

This gives a practical approximation of tri-temporal learning without rewriting the data loader into explicit graph triplets.

## Why This Design Fits ORTHRUS

This design was chosen to fit the current PIDSMaker ORTHRUS implementation with minimal disruption:

- the main task stays edge-type prediction
- decoders remain Euclidean
- temporal learning remains auxiliary
- sequential graph loading is preserved
- tri-temporal learning is implemented through cached states instead of changing the batching interface

## Recommended Ablation Study

To validate each contribution, compare:

1. ORTHRUS
2. ORTHRUS + bi-temporal contrastive
3. ORTHRUS + hyperbolic encoder
4. ORTHRUS + hyperbolic encoder + tri-temporal contrastive
5. ORTHRUS + hyperbolic encoder + tri-temporal contrastive + adaptive weighting

Recommended sensitivity analyses:

- curvature parameter `k`
- temporal loss weight `lambda`
- temperature `T`
- number of hyperbolic attention heads

## Practical Notes

### Decoder Space

The hyperbolic encoder should not directly feed Lorentz embeddings into the ORTHRUS decoder.

Instead:

- message passing happens in Lorentz space
- final node representations are mapped back with `logmap0`
- edge decoding and contrastive projection operate in Euclidean tangent space

This keeps the implementation compatible with the current ORTHRUS objective stack.

### Engineering Interpretation

In implementation terms, HT-ORTHRUS is:

- ORTHRUS with `tgn + hyperbolic_transformer`
- plus temporal contrastive state caching over three windows
- plus adaptive weighting for temporal consistency loss

## Summary

HT-ORTHRUS combines:

1. Hyperbolic structural encoding to model hierarchical provenance propagation.
2. Anchor-centered tri-temporal contrastive learning to model temporal continuity.
3. Adaptive temporal weighting to emphasize reliable windows and stable nodes.

The method preserves ORTHRUS' original downstream task while strengthening both structural representation and temporal consistency.
