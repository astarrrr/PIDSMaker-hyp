# Hyperbolic Transformer Reference

Reference source for later implementation work:
- `/home/astar/projects/hyperbolicTransformer-master`
- Main reusable module folder: `/home/astar/projects/hyperbolicTransformer-master/Hypformer`

This note summarizes the external project's hyperbolic implementation in a way that is useful for integrating a hyperbolic temporal encoder into PIDSMaker.

## Stable Takeaways

The reference project is centered on a fully hyperbolic Transformer called `HypFormer`. Its implementation is built around the Lorentz model of hyperbolic space, not a Poincare-ball-first design.

The reusable pieces are:
- `Hypformer/hypformer.py`: model and attention implementation
- `Hypformer/manifolds/lorentz.py`: manifold wrapper on top of `geoopt.Lorentz`
- `Hypformer/manifolds/lorentz_math.py`: low-level Lorentz operations
- `Hypformer/manifolds/hyp_layer.py`: hyperbolic linear/norm/activation/dropout/classifier and optimizer split

## Geometry Choice

The project uses the Lorentz/hyperboloid model:
- manifold class: `Lorentz`
- curvature parameter: `k`
- optional learnable curvature exists in the manifold wrapper, but examples mainly pass fixed `k_in` and `k_out`

Important implementation convention:
- hyperbolic points are stored as `[time_like, spatial...]`
- many layer operations act only on the spatial part `x[..., 1:]`
- the time-like coordinate is recomputed afterward as:

```python
x_time = ((x_space ** 2).sum(dim=-1, keepdims=True) + k).sqrt()
```

This is a very important design decision. The code avoids many explicit tangent-space residual formulas by updating the spatial part and then re-projecting to the Lorentz manifold.

## Core Attention Design

The attention block is implemented in `TransConvLayer` inside `Hypformer/hypformer.py`.

Two attention types are supported:
- `full`: dense all-pairs attention
- `linear_focused`: kernelized linear attention

### Full Hyperbolic Attention

The attention score is built from Lorentz cross-inner product:

```python
att_weight = 2 + 2 * manifold.cinner(q, k)
```

Then:
- divide by scale
- add bias
- apply softmax
- aggregate values using `manifold.mid_point`

Interpretation:
- the reference does not use a standard Euclidean dot-product attention
- it uses Lorentz geometry directly for pair scoring
- weighted aggregation is done by hyperbolic midpoint/barycenter instead of Euclidean weighted sum

### Linear Attention

The linear variant does not stay purely geometric in every micro-step:
- it extracts the spatial coordinates only
- applies a positive kernelization based on `relu(x) + eps`
- uses a focus-power transform `fp`
- computes linear attention in a Euclidean-style way on spatial coordinates
- reconstructs the Lorentz time-like coordinate at the end

This is a pragmatic implementation, not a fully symbolic Riemannian derivation at every line.

For PIDSMaker, this is useful because it shows a practical compromise:
- keep manifold-aware representation globally
- allow local attention computations to use simpler spatial-coordinate operations when needed

## Hyperbolic Building Blocks

Implemented in `Hypformer/manifolds/hyp_layer.py`.

### `HypLinear`

Behavior:
- if input is Euclidean, prepend one coordinate and map with `expmap0`
- then apply a standard `nn.Linear`
- reconstruct the time-like Lorentz coordinate
- optionally rescale to another manifold curvature

Practical meaning:
- this is not Mobius matrix multiplication
- it is a Lorentz-coordinate linear layer with reprojection
- easier to implement and debug than a fully intrinsic alternative

This style is likely the best first step for PIDSMaker.

### `HypLayerNorm`

Behavior:
- normalize only the spatial coordinates
- recompute the time-like coordinate

### `HypActivation`

Behavior:
- apply activation only to spatial coordinates
- recompute the time-like coordinate

### `HypDropout`

Behavior:
- dropout only on spatial coordinates
- recompute the time-like coordinate

### `HypCLS`

Behavior:
- stores class prototypes as `geoopt.ManifoldParameter`
- classification score is based on Lorentz inner-product-derived distance

This is useful if PIDSMaker later wants a fully hyperbolic decoder, but it is not required for the first encoder migration.

## Residual and Fusion Strategy

The project repeatedly uses:
- `manifold.mid_point(torch.stack((x, residual), dim=1))`

This appears in:
- positional encoding fusion
- residual connections between Transformer layers
- graph/Transformer branch fusion in larger variants

This is one of the most reusable ideas for PIDSMaker.

Instead of Euclidean `x + residual`, the reference uses hyperbolic midpoint as the residual merge operator.

## Euclidean vs Hyperbolic Decoder Boundary

The model explicitly supports two output styles:
- `decoder_type = euc`
- `decoder_type = hyp`

For `euc`:
- final hyperbolic features are mapped back with `logmap0`
- then a normal Euclidean linear layer is used

For `hyp`:
- the decoder stays hyperbolic using `HypLinear` / `HypCLS`

This is highly relevant for PIDSMaker.

Recommended migration order:
1. Keep the new encoder hyperbolic internally.
2. Map back to Euclidean before existing edge decoder and temporal contrastive head.
3. Only later consider a fully hyperbolic decoder and hyperbolic temporal contrastive loss.

## Optimizer Strategy

The reference separates parameters into:
- Euclidean parameters
- hyperbolic manifold parameters (`ManifoldParameter`)

Then it uses:
- Euclidean optimizer: `Adam` or `SGD`
- Hyperbolic optimizer: `RiemannianAdam` or `RiemannianSGD`

This is implemented in `Optimizer` inside `hyp_layer.py`.

Important caveat:
- only parameters explicitly stored as `ManifoldParameter` go through the Riemannian optimizer
- ordinary tensors inside Lorentz-coordinate layers still use the Euclidean optimizer

For PIDSMaker, the first version may not need a split optimizer if we do not introduce manifold-valued trainable prototypes/embeddings. But if we add hyperbolic classifier prototypes or learnable manifold embeddings later, we should copy this pattern.

## What Is Reusable For PIDSMaker

Directly reusable ideas:
- Lorentz manifold wrapper with `expmap0`, `logmap0`, `cinner`, `mid_point`
- hyperbolic residual merge via `mid_point`
- hyperbolic layernorm/activation/dropout pattern on spatial coordinates
- staged output boundary: hyperbolic encoder, Euclidean decoder

Reusable with adaptation:
- `TransConvLayer` attention logic
- linear-focused attention
- hyperbolic classifier layer
- split optimizer

Not directly reusable as-is:
- full all-pairs attention over all nodes
- dataset/training scripts
- graph-level task assumptions from their benchmarks

PIDSMaker uses sampled temporal subgraphs and edge-level objectives, so attention must likely be localized to the sampled subgraph instead of dense all-node attention.

## Best-Fit Adaptation For PIDSMaker

The safest adaptation path is:

1. Keep `TGNEncoder` as the temporal shell.
2. Replace the current Euclidean `GraphAttentionEmbedding` base encoder with a new hyperbolic encoder.
3. Use Lorentz manifold operations internally.
4. Return Euclidean node embeddings by applying `logmap0(...)[..., 1:]` before handing outputs to the existing edge-type decoder.

In other words:
- temporal batching, neighbor loading, and TGN graph construction stay the same
- only the inner message-passing / attention encoder becomes hyperbolic

## Suggested First Implementation Scope

For PIDSMaker, first implementation should include:
- `Lorentz` manifold wrapper
- `HypLinear`
- `HypLayerNorm`
- `HypActivation`
- `HypDropout`
- a localized `HyperbolicTemporalTransformerLayer`
- Euclidean output projection via `logmap0`

It should not initially include:
- fully hyperbolic edge decoder
- hyperbolic tri-temporal contrastive loss
- dense all-pairs attention across the whole graph
- manifold parameter split optimizer unless needed

## Specific Design Choices Worth Copying

Copy:
- Lorentz model instead of Poincare for first implementation
- spatial-coordinate-only normalization/activation/dropout
- midpoint residual fusion
- Euclidean decoder boundary

Do not copy blindly:
- dense full attention over all nodes
- benchmark-specific graph branch fusion
- assumption that sequence length is moderate enough for full pairwise attention

## Dependency Note

The reference implementation depends on `geoopt`.

If PIDSMaker adopts this design, the expected extra dependency is:

```bash
pip install geoopt
```

## Recommended Next Step In PIDSMaker

When implementing a hyperbolic encoder in PIDSMaker:
- create a new encoder file rather than editing the current `graph_attention.py` in place
- preserve the current encoder interface: input tensors in, return `{\"h\": ...}`
- first return Euclidean embeddings after `logmap0`
- only after the baseline works, evaluate whether temporal contrastive loss should also move into hyperbolic space

## Minimal Reference Pointers

Use these files first when implementing:
- `/home/astar/projects/hyperbolicTransformer-master/Hypformer/hypformer.py`
- `/home/astar/projects/hyperbolicTransformer-master/Hypformer/manifolds/hyp_layer.py`
- `/home/astar/projects/hyperbolicTransformer-master/Hypformer/manifolds/lorentz.py`

Ignore the benchmark scripts and dataset code until integration work is complete.
