# Early Detection Requirements

  ## Goal

  Implement an early-detection-oriented model variant in `/home/astar/projects/PIDSMaker-hyp` on a new
  git branch:

  - Branch: `feature/early-detection`
  - Main direction: dual-branch fusion model
  - Do not focus on loss redesign for now

  ## Confirmed Constraints

  - Training is performed on benign / normal data only.
  - Therefore, training objectives that directly upweight early malicious events are not suitable for
  the first version.
  - The first iteration should avoid changing too many variables at once.
  - The purpose is to build a model contribution, not just tune parameters like `tgn_neighbor_size`.

  ## Current Understanding Of Orthrus In This Repo

  - `orthrus` uses `tgn` together with `graph_attention`.
  - It also uses `tgn_last_neighbor`.
  - In the current config, `use_memory: False`, so the TGN path is relatively lightweight and not a
  full memory-heavy temporal setup.

  ## Chosen Direction

  Start with a dual-branch encoder for early detection:

  - `instant branch`
    - Focus on current-batch / local immediate evidence
    - Should not depend on expanded TGN historical context
  - `context branch`
    - Reuse the existing Orthrus-style `TGNEncoder` path
    - Preserve current short-term historical modeling
  - `fusion`
    - Fuse the two branch outputs into a final node representation
    - First version should use a simple gate, not a complex attention mechanism

  ## First Version: Minimal Viable Model

  ### Context Branch

  Reuse the existing Orthrus temporal-context path:

  - existing `TGNEncoder`
  - existing `graph_attention` downstream structure

  This branch should remain as close as possible to the current implementation so that experimental
  differences mainly come from the new instant branch and fusion.

  ### Instant Branch

  Build a lightweight encoder that only uses the current batch graph.

  Preferred first version:

  - a lightweight 1-layer graph encoder on the original current-batch graph
  - keep it simple
  - avoid adding new memory/state behavior

  Fallback minimal version if integration is difficult:

  - an MLP over current node features only

  ### Fusion

  Use node-level gated fusion:

  `h = g * h_inst + (1 - g) * h_ctx`

  Where:

  - `h_inst` is the instant-branch node embedding
  - `h_ctx` is the context-branch node embedding
  - `g` is produced by a small gate network from `[h_inst, h_ctx]`

  The fused output should still expose:

  - `h`
  - `h_src`
  - `h_dst`

  so that existing downstream decoders/objectives can be reused with minimal changes.

  ## Recommended Implementation Scope

  First implementation should only introduce the structural change below:

  - add instant branch
  - add context branch wrapper
  - add gated fusion

  Avoid in the first pass:

  - attention-based fusion
  - uncertainty-aware fusion
  - early-weighted loss
  - memory redesign
  - extra auxiliary losses
  - major evaluation redesign

  ## Evaluation Strategy For First Pass

  At minimum, compare:

  1. `instant only`
  2. `context only` (current Orthrus baseline)
  3. `fusion`

  Primary question:

  - Does fusion improve early-detection behavior, especially under smaller historical context budgets?

  ## Early Detection Motivation To Preserve

  The core argument is:

  - early attack stages often do not yet have enough accumulated context
  - but they may already exhibit immediate local anomalies
  - a single context-heavy branch may miss these weak early signals
  - separating immediate evidence and short-term context may improve early detection

  ## Likely Files To Modify In `/home/astar/projects/PIDSMaker-hyp`

  These are the expected touch points based on the current repo structure:

  - `pidsmaker/encoders/early_fusion_encoder.py`
    - new wrapper encoder
  - `pidsmaker/encoders/instant_encoder.py`
    - new lightweight current-batch encoder
  - `pidsmaker/encoders/__init__.py`
    - export new encoders
  - `pidsmaker/factory.py`
    - register / construct the new dual-branch model
  - `config/orthrus_early_fusion.yml`
    - new config for experiments

  ## Practical Guidance

  - Keep the first version easy to ablate.
  - Reuse as much of the current Orthrus pipeline as possible.
  - Make the branch outputs align at node level before fusion.
  - Prefer a minimal working prototype over a highly optimized design.

  ## What Not To Claim Yet

  Do not frame the first version as:

  - a new loss design
  - a malicious-label-aware training method
  - a fully online streaming system

  For now, frame it as:

  - an early-detection-oriented dual-branch encoder architecture for benign-only training settings