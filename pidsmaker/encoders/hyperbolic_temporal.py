"""Hyperbolic temporal encoder following HTGN-style design."""

from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv


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

        self.time_beta_raw = nn.Parameter(torch.tensor(0.0))
        self.time_linear = nn.Linear(1, in_dim)

        # HGNN in tangent space (HCC view: manifold ops approximated via tangent transforms).
        self.hgnn_layers = nn.ModuleList()
        current_dim = in_dim
        for _ in range(num_layers - 1):
            self.hgnn_layers.append(GATConv(current_dim, hid_dim, heads=1, concat=False))
            current_dim = hid_dim
        self.hgnn_layers.append(GATConv(current_dim, out_dim, heads=1, concat=False))

        # HGRU in tangent space.
        self.hgru = nn.GRUCell(out_dim, out_dim)
        self.init_hidden = nn.Parameter(torch.zeros(out_dim))

        # HTA projections.
        self.attn_q = nn.Linear(out_dim, out_dim, bias=False)
        self.attn_k = nn.Linear(out_dim, out_dim, bias=False)
        self.attn_v = nn.Linear(out_dim, out_dim, bias=False)

        self._history = deque(maxlen=max(self.window_size, 1))

    def reset_state(self):
        self._history.clear()

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

    def _temporal_gate(self, node_time):
        if node_time.numel() == 0:
            return node_time
        max_t = node_time.max()
        delta_t = max_t - node_time
        scale = torch.clamp(delta_t.mean(), min=1e-6)
        beta = F.softplus(self.time_beta_raw) + 1e-6
        return torch.exp(-(beta * delta_t / scale))

    def _history_attention(self, query_tan):
        if len(self._history) == 0:
            return None

        keys_tan = torch.stack([self._logmap0(h) for h in self._history], dim=0)  # (W, d)
        q = self.attn_q(query_tan)  # (d,)
        k = self.attn_k(keys_tan)  # (W, d)
        v = self.attn_v(keys_tan)  # (W, d)

        score = torch.matmul(k, q) / (
            (query_tan.shape[-1] ** 0.5) * max(self.attention_temperature, 1e-6)
        )
        alpha = torch.softmax(score, dim=0).unsqueeze(-1)
        return torch.sum(alpha * v, dim=0)

    def forward(self, x=None, edge_index=None, t=None, batch=None, **kwargs):
        if x is None and batch is not None:
            x = getattr(batch, "x", None)
        if x is None:
            raise ValueError("HyperbolicTemporalEncoder requires node features `x`.")
        if edge_index is None:
            raise ValueError("HyperbolicTemporalEncoder requires `edge_index`.")

        # Temporal decay feature injection.
        if t is None:
            gate = torch.ones(x.shape[0], device=x.device, dtype=x.dtype)
        else:
            node_time = self._compute_node_time(edge_index, t.to(x.dtype), x.shape[0])
            gate = self._temporal_gate(node_time).to(x.dtype)
        x = x + self.time_linear(gate.unsqueeze(-1))

        # HGNN in tangent space.
        for conv in self.hgnn_layers[:-1]:
            x = conv(x, edge_index)
            x = self.activation(x)
            x = self.dropout(x)
        h_tan_now = self.hgnn_layers[-1](x, edge_index)

        # HTA over historical manifold states (mapped to tangent).
        query_tan = h_tan_now.mean(dim=0)
        hist_ctx_tan = self._history_attention(query_tan)

        # HGRU update.
        if hist_ctx_tan is None:
            hist_ctx_tan = self.init_hidden.to(h_tan_now.device, h_tan_now.dtype)
        hist_ctx_tan = hist_ctx_tan.unsqueeze(0).expand(h_tan_now.shape[0], -1)
        h_tan = self.hgru(h_tan_now, hist_ctx_tan)

        # Back to hyperbolic manifold.
        h_hyp = self._expmap0(h_tan)
        pooled_hyp = self._expmap0(h_tan.mean(dim=0, keepdim=True)).squeeze(0)

        # HTC: smooth temporal drift between consecutive pooled states.
        aux_loss = None
        if len(self._history) > 0 and self.htc_weight > 0.0:
            prev_tan = self._logmap0(self._history[-1].unsqueeze(0)).squeeze(0)
            curr_tan = self._logmap0(pooled_hyp.unsqueeze(0)).squeeze(0)
            aux_loss = self.htc_weight * F.mse_loss(curr_tan, prev_tan)

        self._history.append(pooled_hyp.detach())
        return {"h": h_hyp, "aux_loss": aux_loss}
