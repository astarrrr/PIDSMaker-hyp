"""PIDS Model combining encoder with multiple training objectives.

The Model class orchestrates encoder execution and applies multiple objectives
(reconstruction, prediction, contrastive learning) for joint training. Supports
few-shot learning mode and MC Dropout uncertainty quantification.
"""

import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from pidsmaker.encoders import TGNEncoder
from pidsmaker.experiments.uncertainty import activate_dropout_inference


class Model(nn.Module):
    """Main PIDS model combining graph encoder with training objectives.

    Attributes:
        encoder: Neural network encoder (SAGE, GAT, TGN, etc.)
        objectives: List of training objectives (reconstruction, prediction, etc.)
        objective_few_shot: Few-shot detection objective (optional)
        device: PyTorch device (cuda/cpu)
        few_shot_mode: Whether currently in few-shot fine-tuning mode
    """

    def __init__(
        self,
        encoder: nn.Module,
        objectives: list[nn.Module],
        objective_few_shot: nn.Module,
        device,
        is_running_mc_dropout,
        use_few_shot,
        freeze_encoder,
        node_out_dim,
        temporal_contrastive_cfg,
    ):
        super(Model, self).__init__()

        self.encoder = encoder
        self.objectives = nn.ModuleList(objectives)
        self.device = device
        self.is_running_mc_dropout = is_running_mc_dropout

        self.objective_few_shot = objective_few_shot
        self.use_few_shot = use_few_shot
        self.few_shot_mode = False
        self.freeze_encoder = freeze_encoder
        self.temporal_contrastive_cfg = temporal_contrastive_cfg
        self.temporal_contrastive_enabled = temporal_contrastive_cfg.enabled

        if self.temporal_contrastive_enabled:
            proj_dim = temporal_contrastive_cfg.projection_dim
            self.temporal_projector = nn.Sequential(
                nn.Linear(node_out_dim, proj_dim),
                nn.ReLU(),
                nn.Linear(proj_dim, proj_dim),
            )
        else:
            self.temporal_projector = None

    def embed(self, batch, inference=False, **kwargs):
        """Generate node embeddings for batch using encoder.

        Args:
            batch: Data batch with edge_index, node features, timestamps, etc.
            inference: If True, run in inference mode (no gradients)
            **kwargs: Additional arguments passed to encoder

        Returns:
            tuple: (h, h_src, h_dst) where:
                - h: All node embeddings (N, d) or tuple of (src_nodes, dst_nodes)
                - h_src: Source node embeddings for edges (E, d)
                - h_dst: Destination node embeddings for edges (E, d)
        """
        train_mode = not inference
        edge_index = batch.edge_index
        with torch.set_grad_enabled(train_mode):
            res = self.encoder(
                edge_index=edge_index,
                t=batch.t,
                x_src=batch.x_src,
                x_dst=batch.x_dst,
                msg=batch.msg,
                edge_feats=getattr(batch, "edge_feats", None),
                inference=inference,
                edge_types=batch.edge_type,
                node_type_src=batch.node_type_src,
                node_type_dst=batch.node_type_dst,
                batch=batch,
                # Reindexing attr
                x=getattr(batch, "x", None),
                original_n_id=getattr(batch, "original_n_id", None),
                node_type=getattr(batch, "node_type", None),
                node_type_argmax=getattr(batch, "node_type_argmax", None),
            )
        h, h_src, h_dst = self.gather_h(batch, res)
        return h, h_src, h_dst

    def _embed_nodes(self, batch, inference):
        """Runs the encoder, optionally applying input masking for reconstruction objectives."""
        train_mode = not inference
        mask_nodes = None
        x_for_encoding = getattr(batch, "x", None)

        if train_mode and x_for_encoding is not None:
            # Check if we have a GMAEFeatReconstruction objective
            for objective in self.objectives:
                # ValidationWrapper hides the underlying objective's methods (like mask_input). 
                # We need to access the inner objective to find and call mask_input if it exists.
                # TODO: make ValidationWrapper transparently proxy method calls or use a more generic unwrapping approach.
                actual_objective = objective
                if hasattr(objective, "objective"):
                    actual_objective = objective.objective

                if hasattr(actual_objective, "mask_input"):
                    x_for_encoding, mask_nodes = actual_objective.mask_input(x_for_encoding)
                    break  # Only mask once

        with torch.set_grad_enabled(train_mode):
            # Pass masked input to encoder if masking was applied
            if mask_nodes is not None and hasattr(batch, "x") and x_for_encoding is not None:
                # Temporarily replace batch.x with masked version
                original_x = batch.x
                batch.x = x_for_encoding
                h, h_src, h_dst = self.embed(batch, inference=inference)
                batch.x = original_x  # Restore original for objectives
            else:
                h, h_src, h_dst = self.embed(batch, inference=inference)

        return h, h_src, h_dst, mask_nodes

    def forward(self, batch, inference=False, validation=False):
        """Forward pass: embed nodes and compute loss/scores across all objectives.

        Args:
            batch: Data batch with graph structure and features
            inference: If True, return anomaly scores; if False, return training loss
            validation: If True, compute validation metrics

        Returns:
            dict: Contains 'loss' key with:
                - Training mode: scalar loss (sum of all objective losses)
                - Inference mode: per-edge anomaly scores (E,)
        """
        h, h_src, h_dst, mask_nodes = self._embed_nodes(batch, inference)
        train_mode = not inference

        with torch.set_grad_enabled(train_mode):
            # Train mode: loss | Inference mode: scores
            loss_or_scores = None
            temporal_state = self._get_temporal_state(batch, h)

            for objective in self.objectives:
                results = objective(
                    h_src=h_src,  # shape (E, d)
                    h_dst=h_dst,  # shape (E, d)
                    h=h,  # shape (N, d)
                    edge_index=batch.edge_index,
                    edge_type=batch.edge_type,
                    y_edge=batch.y,
                    inference=inference,
                    x=getattr(batch, "x", None),
                    node_type=getattr(batch, "node_type", None),
                    node_type_src=batch.node_type_src,
                    node_type_dst=batch.node_type_dst,
                    validation=validation,
                    batch=batch,
                    mask_nodes=mask_nodes,  # Pass mask_nodes to objectives
                )
                loss = results["loss"]

                if loss_or_scores is None:
                    loss_or_scores = (
                        torch.zeros(1)
                        if train_mode
                        else torch.zeros(loss.shape[0], dtype=torch.float)
                    ).to(batch.edge_index.device)

                if loss.numel() != loss_or_scores.numel():
                    raise TypeError(
                        f"Shapes of loss/score do not match ({loss.numel()} vs {loss_or_scores.numel()})"
                    )
                loss_or_scores = loss_or_scores + loss

            results["loss"] = loss_or_scores
            if temporal_state is not None:
                results["temporal_state"] = temporal_state
            return results

    def _get_temporal_state(self, batch, h):
        if not self.temporal_contrastive_enabled or not isinstance(h, torch.Tensor):
            return None

        node_ids = getattr(batch, "original_n_id", None)
        if node_ids is None or h.shape[0] != node_ids.shape[0]:
            return None

        return {
            "node_ids": node_ids,
            "node_embeddings": h,
        }

    def detach_temporal_state(self, temporal_state):
        if temporal_state is None:
            return None

        return {
            "node_ids": temporal_state["node_ids"].detach(),
            "node_embeddings": temporal_state["node_embeddings"].detach(),
        }

    def _align_temporal_states(self, source_state, target_state):
        if source_state is None or target_state is None:
            return None

        source_ids = source_state["node_ids"]
        target_ids = target_state["node_ids"]
        shared_mask = torch.isin(source_ids, target_ids)
        if not torch.any(shared_mask):
            return None

        source_idx = torch.nonzero(shared_mask, as_tuple=False).squeeze(-1)
        shared_ids = source_ids[source_idx]

        target_sorted_ids, target_perm = torch.sort(target_ids)
        target_idx = target_perm[torch.searchsorted(target_sorted_ids, shared_ids)]

        max_pairs = self.temporal_contrastive_cfg.max_pairs_per_batch
        if max_pairs > 0 and source_idx.numel() > max_pairs:
            perm = torch.randperm(source_idx.numel(), device=source_idx.device)[:max_pairs]
            source_idx = source_idx[perm]
            target_idx = target_idx[perm]
            shared_ids = shared_ids[perm]

        return {
            "shared_ids": shared_ids,
            "source_h": source_state["node_embeddings"][source_idx],
            "target_h": target_state["node_embeddings"][target_idx].detach(),
        }

    def _compute_pair_temporal_loss(self, pair_state, node_weights=None):
        if pair_state is None:
            return None

        source_z = self.temporal_projector(pair_state["source_h"])
        target_z = self.temporal_projector(pair_state["target_h"])

        source_z = F.normalize(source_z, p=2, dim=-1)
        target_z = F.normalize(target_z, p=2, dim=-1)

        logits = torch.matmul(source_z, target_z.T) / self.temporal_contrastive_cfg.temperature
        labels = torch.arange(logits.shape[0], device=logits.device)

        loss_fwd = F.cross_entropy(logits, labels, reduction="none")
        loss_bwd = F.cross_entropy(logits.T, labels, reduction="none")
        loss = 0.5 * (loss_fwd + loss_bwd)

        if node_weights is not None:
            norm = node_weights.sum().clamp_min(1e-6)
            return (loss * node_weights).sum() / norm
        return loss.mean()

    def _window_overlap_weight(self, left_state, right_state):
        if left_state is None or right_state is None:
            return 0.0

        left_ids = left_state["node_ids"]
        right_ids = right_state["node_ids"]
        inter = torch.isin(left_ids, right_ids).sum().item()
        union = torch.unique(torch.cat([left_ids, right_ids])).numel()
        if union == 0:
            return 0.0
        return inter / union

    def _node_stability_weights(self, pair_ids, states):
        if pair_ids is None or pair_ids.numel() == 0:
            return None

        gamma = getattr(self.temporal_contrastive_cfg, "node_weight_floor", 0.5)
        counts = torch.zeros_like(pair_ids, dtype=torch.float)
        for state in states:
            counts = counts + torch.isin(pair_ids, state["node_ids"]).to(torch.float)
        base = counts / max(len(states), 1)
        return gamma + (1.0 - gamma) * base

    def compute_temporal_contrastive_loss(self, current_state, previous_state):
        if (
            not self.temporal_contrastive_enabled
            or current_state is None
            or previous_state is None
        ):
            return torch.zeros(1, device=self.device).squeeze()

        pair_state = self._align_temporal_states(current_state, previous_state)
        if pair_state is None:
            return current_state["node_embeddings"].new_zeros(())
        return self._compute_pair_temporal_loss(pair_state)

    def compute_tri_temporal_contrastive_loss(
        self, previous_previous_state, previous_state, current_state
    ):
        if (
            not self.temporal_contrastive_enabled
            or previous_previous_state is None
            or previous_state is None
            or current_state is None
        ):
            return torch.zeros(1, device=self.device).squeeze()

        prev_pair = self._align_temporal_states(previous_state, previous_previous_state)
        next_pair = self._align_temporal_states(previous_state, current_state)
        if prev_pair is None and next_pair is None:
            return previous_state["node_embeddings"].new_zeros(())

        eps = getattr(self.temporal_contrastive_cfg, "window_weight_floor", 0.1)
        pair_weights = []
        pair_losses = []

        if prev_pair is not None:
            prev_weight = max(
                self._window_overlap_weight(previous_previous_state, previous_state), eps
            )
            prev_node_weights = self._node_stability_weights(
                prev_pair["shared_ids"],
                [previous_previous_state, previous_state, current_state],
            )
            pair_weights.append(prev_weight)
            pair_losses.append(self._compute_pair_temporal_loss(prev_pair, prev_node_weights))

        if next_pair is not None:
            next_weight = max(self._window_overlap_weight(previous_state, current_state), eps)
            next_node_weights = self._node_stability_weights(
                next_pair["shared_ids"],
                [previous_previous_state, previous_state, current_state],
            )
            pair_weights.append(next_weight)
            pair_losses.append(self._compute_pair_temporal_loss(next_pair, next_node_weights))

        weight_tensor = previous_state["node_embeddings"].new_tensor(pair_weights)
        weight_tensor = weight_tensor / weight_tensor.sum().clamp_min(1e-6)
        return sum(weight * loss for weight, loss in zip(weight_tensor, pair_losses))

    def get_val_ap(self):
        """Get average validation score across all objectives.

        Returns:
            float: Mean validation score (average precision)
        """
        return np.mean([d.get_val_score() for d in self.objectives])

    def to_device(self, device):
        """Move model and associated components to specified device.

        Handles special device migration for TGN memory and graph reindexer.

        Args:
            device: Target PyTorch device

        Returns:
            Model: Self for chaining
        """
        if self.device == device:
            return self

        for objective in self.objectives:
            objective.graph_reindexer.to(device)

        if isinstance(self.encoder, TGNEncoder):
            self.encoder.to_device(device)

        self.device = device
        return self.to(device)

    def eval(self):
        """Set model to evaluation mode.

        Overrides default eval() to keep dropout active for MC Dropout uncertainty.
        """
        super().eval()

        if self.is_running_mc_dropout:
            activate_dropout_inference(self)

    def gather_h(self, batch, res):
        """Extract source and destination node embeddings from encoder output.

        Handles different encoder output formats:
        - Single tensor h (N, d): index by edge_index to get h_src, h_dst
        - Tuple (h_src_nodes, h_dst_nodes): separate embeddings for src/dst nodes
        - Pre-computed (h_src, h_dst): already indexed for edges

        Args:
            batch: Data batch with edge_index
            res: Encoder output dict with 'h', optionally 'h_src', 'h_dst'

        Returns:
            tuple: (h, h_src, h_dst) node embeddings
        """
        h = res["h"]
        h_src = res.get("h_src", None)
        h_dst = res.get("h_dst", None)

        if None in [h_src, h_dst]:
            if isinstance(h, torch.Tensor):
                # h is a single tensor with node embeddings - index by edge_index
                h_src, h_dst = h[batch.edge_index[0]], h[batch.edge_index[1]]
            elif isinstance(h, (tuple, list)):
                # h is (h_src_nodes, h_dst_nodes) with separate node embeddings - index each
                h_src, h_dst = h[0][batch.edge_index[0]], h[1][batch.edge_index[1]]
            else:
                h_src, h_dst = h

        return h, h_src, h_dst

    def to_fine_tuning(self, do: bool):
        """Switch between self-supervised pretraining and few-shot fine-tuning.

        When entering few-shot mode:
        - Optionally freezes encoder weights
        - Replaces pretraining objectives with few-shot detection objective

        When exiting few-shot mode:
        - Unfreezes encoder
        - Restores pretraining objectives

        Args:
            do: True to enter few-shot mode, False to exit
        """
        if not self.use_few_shot:
            return
        if do and not self.few_shot_mode:
            if self.freeze_encoder:
                self.encoder.eval()
                for param in self.encoder.parameters():  # freeze the encoder
                    param.requires_grad = False

            # the objective is replaced by a copy of the objective_few_shot + the old objective is saved for later switch
            ssl_objective = (
                self.objectives
            )  # switch the pretext objective and fine-tuning objective
            self.objectives = copy.deepcopy(self.objective_few_shot)
            self.ssl_objective = ssl_objective
            self.few_shot_mode = True

        if not do and self.few_shot_mode:
            self.encoder.train()
            for param in self.encoder.parameters():
                param.requires_grad = True

            # the ssl objective is set back
            self.objectives = self.ssl_objective
            self.few_shot_mode = False

    def reset_state(self):
        """Reset encoder state (e.g., TGN memory) between evaluation windows."""
        if hasattr(self.encoder, "reset_state"):
            self.encoder.reset_state()
