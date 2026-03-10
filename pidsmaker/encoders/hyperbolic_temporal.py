"""Hyperbolic temporal encoder following HTGN-style design."""

from collections import defaultdict, deque

import torch
import torch.nn as nn
from torch_geometric.nn import SAGEConv


class HyperbolicTemporalEncoder(nn.Module):
    """HGNN + HGRU + HTA + HTC in a practical HCC-style implementation."""

    def __init__(
        self,
        in_dim,
        hid_dim,
        out_dim,
        activation,
        dropout,
        num_layers,
        num_nodes,
        c=1.0,
        window_size=3,
        htc_weight=0.01,
        attention_temperature=1.0,
    ):
        super().__init__()
        self.activation = activation
        self.dropout = nn.Dropout(dropout)
        self.curvature = float(c)
        self.window_size = int(window_size)
        self.htc_weight = float(htc_weight)
        self.attention_temperature = float(attention_temperature)
        self.num_nodes = int(num_nodes)

        self.time_encoder = nn.Linear(1, in_dim)

        # Use a stable spatial aggregator; temporal attention is handled by HTA.
        self.hgnn_layers = nn.ModuleList()
        current_dim = in_dim
        for _ in range(num_layers - 1):
            self.hgnn_layers.append(SAGEConv(current_dim, hid_dim, normalize=False))
            current_dim = hid_dim
        self.hgnn_layers.append(SAGEConv(current_dim, out_dim, normalize=False))

        # HGRU in tangent space.
        self.hgru = nn.GRUCell(out_dim, out_dim)
        self.init_hidden = nn.Parameter(torch.zeros(out_dim))

        # HTA projections.
        self.attn_q = nn.Linear(out_dim, out_dim, bias=False)
        self.attn_k = nn.Linear(out_dim, out_dim, bias=False)
        self.attn_v = nn.Linear(out_dim, out_dim, bias=False)

        self.register_buffer("global_state", torch.zeros(self.num_nodes, out_dim))
        self._node_history = defaultdict(lambda: deque(maxlen=max(self.window_size, 1)))

    def reset_state(self):
        self.global_state.zero_()
        self._node_history.clear()

    def _project_to_ball(self, x):
        c = torch.as_tensor(self.curvature, device=x.device, dtype=x.dtype)
        sqrt_c = torch.sqrt(torch.clamp(c, min=1e-8))
        max_norm = (1.0 - 1e-5) / sqrt_c
        norm = torch.norm(x, dim=-1, keepdim=True).clamp(min=1e-12)
        scale = torch.clamp(max_norm / norm, max=1.0)
        return x * scale

    def _artanh(self, x):
        x = torch.clamp(x, min=-1.0 + 1e-6, max=1.0 - 1e-6)
        return 0.5 * (torch.log1p(x) - torch.log1p(-x))

    def _expmap0(self, v):
        c = torch.as_tensor(self.curvature, device=v.device, dtype=v.dtype)
        sqrt_c = torch.sqrt(torch.clamp(c, min=1e-8))
        norm = torch.norm(v, dim=-1, keepdim=True).clamp(min=1e-12)
        scaled = sqrt_c * norm
        out = torch.tanh(scaled) * v / scaled
        return self._project_to_ball(out)

    def _logmap0(self, y):
        y = self._project_to_ball(y)
        c = torch.as_tensor(self.curvature, device=y.device, dtype=y.dtype)
        sqrt_c = torch.sqrt(torch.clamp(c, min=1e-8))
        norm = torch.norm(y, dim=-1, keepdim=True).clamp(min=1e-12)
        scaled = sqrt_c * norm
        out = self._artanh(scaled) * y / scaled
        return out

    def _compute_node_time(self, edge_index, t, num_nodes):
        src, dst = edge_index[0], edge_index[1]
        node_time = torch.zeros(num_nodes, dtype=t.dtype, device=t.device)
        counts = torch.zeros(num_nodes, dtype=t.dtype, device=t.device)
        node_time.index_add_(0, src, t)
        node_time.index_add_(0, dst, t)
        counts.index_add_(0, src, torch.ones_like(t))
        counts.index_add_(0, dst, torch.ones_like(t))
        return node_time / torch.clamp(counts, min=1.0)

    def _time_encoding(self, node_time, dtype):
        encoded = torch.cos(self.time_encoder(node_time.unsqueeze(-1).to(dtype)))
        return encoded

    def _history_attention(self, query_tan, global_ids):
        histories = []
        max_len = 0
        for node_id in global_ids.tolist():
            hist = list(self._node_history[int(node_id)])
            histories.append(hist)
            max_len = max(max_len, len(hist))

        if max_len == 0:
            return None

        batch_size, hid_dim = query_tan.shape
        hist_hyp = torch.zeros(
            batch_size,
            max_len,
            hid_dim,
            device=query_tan.device,
            dtype=query_tan.dtype,
        )
        valid_mask = torch.zeros(batch_size, max_len, device=query_tan.device, dtype=torch.bool)

        for i, hist in enumerate(histories):
            if not hist:
                continue
            stacked = torch.stack([h.to(query_tan.device, query_tan.dtype) for h in hist], dim=0)
            hist_hyp[i, : stacked.shape[0]] = stacked
            valid_mask[i, : stacked.shape[0]] = True

        hist_tan = self._logmap0(hist_hyp)

        q = self.attn_q(query_tan).unsqueeze(1)
        k = self.attn_k(hist_tan)
        v = self.attn_v(hist_tan)

        score = (k * q).sum(dim=-1) / (
            query_tan.shape[-1] ** 0.5 * max(self.attention_temperature, 1e-6)
        )
        has_history = valid_mask.any(dim=-1, keepdim=True)
        safe_score = score.masked_fill(~valid_mask, float("-inf"))
        safe_score = torch.where(has_history, safe_score, torch.zeros_like(safe_score))
        alpha = torch.softmax(safe_score, dim=-1)
        alpha = torch.where(has_history, alpha, torch.zeros_like(alpha)).unsqueeze(-1)
        return torch.sum(alpha * v, dim=1)

    def _hyperbolic_distance(self, x, y):
        c = torch.as_tensor(self.curvature, device=x.device, dtype=x.dtype)
        x_sq = torch.sum(x * x, dim=-1)
        y_sq = torch.sum(y * y, dim=-1)
        diff_sq = torch.sum((x - y) * (x - y), dim=-1)
        denom = torch.clamp((1 - c * x_sq) * (1 - c * y_sq), min=1e-8)
        z = 1 + 2 * c * diff_sq / denom
        z = torch.clamp(z, min=1.0 + 1e-6)
        return torch.acosh(z) / torch.sqrt(torch.clamp(c, min=1e-8))

    def forward(self, x=None, edge_index=None, t=None, batch=None, original_n_id=None, **kwargs):
        if x is None and batch is not None:
            x = getattr(batch, "x", None)
        if x is None:
            raise ValueError("HyperbolicTemporalEncoder requires node features `x`.")
        if edge_index is None:
            raise ValueError("HyperbolicTemporalEncoder requires `edge_index`.")
        if original_n_id is None and batch is not None:
            original_n_id = getattr(batch, "original_n_id", None)
        if original_n_id is None:
            raise ValueError("HyperbolicTemporalEncoder requires `original_n_id` for global alignment.")
        global_ids = original_n_id.long().to(self.global_state.device)

        # Inject temporal information with a TGN-style encoding instead of a decay gate.
        if t is None:
            x = x + self._time_encoding(torch.zeros(x.shape[0], device=x.device), x.dtype)
        else:
            node_time = self._compute_node_time(edge_index, t.to(x.dtype), x.shape[0])
            x = x + self._time_encoding(node_time, x.dtype)

        # HGNN in tangent space.
        for conv in self.hgnn_layers[:-1]:
            x = conv(x, edge_index)
            x = self.activation(x)
            x = self.dropout(x)
        h_tan_now = self.hgnn_layers[-1](x, edge_index)

        prev_hyp = self.global_state[global_ids].to(h_tan_now.device, h_tan_now.dtype)
        prev_tan = self._logmap0(prev_hyp)

        # HTA over node-aligned historical states.
        hist_ctx_tan = self._history_attention(h_tan_now, global_ids)
        if hist_ctx_tan is None:
            hist_ctx_tan = torch.zeros_like(h_tan_now)

        # HGRU update with node-aligned previous state.
        h_tan = self.hgru(h_tan_now, prev_tan + hist_ctx_tan)
        h_hyp = self._expmap0(h_tan)

        # HTC: smooth temporal drift between consecutive states for the same nodes.
        aux_loss = None
        if self.htc_weight > 0.0:
            has_prev = torch.any(prev_hyp.abs().sum(dim=-1) > 0)
            if has_prev:
                distance = self._hyperbolic_distance(h_hyp, prev_hyp)
                valid_prev = (prev_hyp.abs().sum(dim=-1) > 0).to(distance.dtype)
                aux_loss = self.htc_weight * ((distance * valid_prev).sum() / torch.clamp(valid_prev.sum(), min=1.0))

        self.global_state[global_ids] = h_hyp.detach().to(self.global_state.device, self.global_state.dtype)
        for node_id, state in zip(global_ids.tolist(), h_hyp.detach()):
            self._node_history[int(node_id)].append(
                state.to(self.global_state.device, self.global_state.dtype).clone()
            )
        return {"h": h_hyp, "aux_loss": aux_loss}
