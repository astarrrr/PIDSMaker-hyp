import math

import geoopt
import torch
import torch.nn as nn
from torch_geometric.utils import softmax


def lorentz_inner(x, y):
    return -(x[..., :1] * y[..., :1]).sum(dim=-1) + (x[..., 1:] * y[..., 1:]).sum(dim=-1)


class HypLinear(nn.Module):
    """Lorentz linear layer adapted from the reference hyperbolic transformer."""

    def __init__(self, manifold, in_features, out_features):
        super().__init__()
        self.manifold = manifold
        self.linear = nn.Linear(in_features + 1, out_features)
        nn.init.xavier_uniform_(self.linear.weight, gain=math.sqrt(2))
        nn.init.zeros_(self.linear.bias)

    def forward(self, x, x_manifold="hyp"):
        if x_manifold != "hyp":
            tangent = torch.cat([torch.zeros_like(x[..., :1]), x], dim=-1)
            x = self.manifold.expmap0(tangent)

        x_space = self.linear(x)
        x_time = ((x_space**2).sum(dim=-1, keepdim=True) + self.manifold.k).sqrt()
        return torch.cat([x_time, x_space], dim=-1)


class HypLayerNorm(nn.Module):
    def __init__(self, manifold, hidden_dim):
        super().__init__()
        self.manifold = manifold
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        x_space = self.layer_norm(x[..., 1:])
        x_time = ((x_space**2).sum(dim=-1, keepdim=True) + self.manifold.k).sqrt()
        return torch.cat([x_time, x_space], dim=-1)


class HypActivation(nn.Module):
    def __init__(self, manifold, activation):
        super().__init__()
        self.manifold = manifold
        self.activation = activation

    def forward(self, x):
        x_space = self.activation(x[..., 1:])
        x_time = ((x_space**2).sum(dim=-1, keepdim=True) + self.manifold.k).sqrt()
        return torch.cat([x_time, x_space], dim=-1)


class HypDropout(nn.Module):
    def __init__(self, manifold, dropout):
        super().__init__()
        self.manifold = manifold
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x_space = self.dropout(x[..., 1:])
        x_time = ((x_space**2).sum(dim=-1, keepdim=True) + self.manifold.k).sqrt()
        return torch.cat([x_time, x_space], dim=-1)


class HyperbolicAttentionLayer(nn.Module):
    """Edge-aware Lorentz attention with tangent-space aggregation."""

    def __init__(self, manifold, hidden_dim, num_heads, dropout, edge_dim, flow):
        super().__init__()
        self.manifold = manifold
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.flow = flow

        self.q_proj = nn.ModuleList([HypLinear(manifold, hidden_dim, hidden_dim) for _ in range(num_heads)])
        self.k_proj = nn.ModuleList([HypLinear(manifold, hidden_dim, hidden_dim) for _ in range(num_heads)])
        self.v_proj = nn.ModuleList([HypLinear(manifold, hidden_dim, hidden_dim) for _ in range(num_heads)])

        self.edge_bias = nn.Linear(edge_dim, num_heads, bias=False) if edge_dim else None
        self.scale = math.sqrt(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def _aggregate(self, values, alpha, target_index, num_nodes):
        tangent = self.manifold.logmap0(values)
        weighted = tangent[..., 1:] * alpha.unsqueeze(-1)
        agg_space = weighted.new_zeros((num_nodes, weighted.shape[-1]))
        agg_space.index_add_(0, target_index, weighted)
        agg_tangent = torch.cat([torch.zeros_like(agg_space[:, :1]), agg_space], dim=-1)
        return self.manifold.expmap0(agg_tangent)

    def forward(self, x, edge_index, edge_feats=None):
        if edge_index.numel() == 0:
            return x

        src, dst = edge_index
        if self.flow == "target_to_source":
            src, dst = dst, src

        head_outputs = []
        edge_bias = self.edge_bias(edge_feats) if self.edge_bias is not None and edge_feats is not None else None

        for head in range(self.num_heads):
            q = self.q_proj[head](x)
            k = self.k_proj[head](x)
            v = self.v_proj[head](x)

            logits = (2 + 2 * lorentz_inner(q[dst], k[src])) / self.scale
            if edge_bias is not None:
                logits = logits + edge_bias[:, head]

            alpha = softmax(logits, dst)
            alpha = self.dropout(alpha)
            head_outputs.append(self._aggregate(v[src], alpha, dst, x.shape[0]))

        stacked = torch.stack(head_outputs, dim=1)
        tangent = self.manifold.logmap0(stacked)[..., 1:]
        mean_space = tangent.mean(dim=1)
        mean_tangent = torch.cat([torch.zeros_like(mean_space[:, :1]), mean_space], dim=-1)
        return self.manifold.expmap0(mean_tangent)


class HyperbolicTransformerEmbedding(nn.Module):
    """Lorentz hyperbolic encoder that returns Euclidean embeddings for Orthrus decoders."""

    def __init__(
        self,
        in_dim,
        hid_dim,
        out_dim,
        edge_dim,
        dropout,
        activation,
        trans_num_heads,
        trans_num_layers,
        k,
        use_bn,
        use_residual,
        flow="source_to_target",
        **kwargs,
    ):
        super().__init__()
        self.manifold = geoopt.Lorentz(k=k)
        self.input_proj = HypLinear(self.manifold, in_dim, hid_dim)
        self.layers = nn.ModuleList(
            [
                HyperbolicAttentionLayer(
                    manifold=self.manifold,
                    hidden_dim=hid_dim,
                    num_heads=trans_num_heads,
                    dropout=dropout,
                    edge_dim=edge_dim,
                    flow=flow,
                )
                for _ in range(trans_num_layers)
            ]
        )
        self.norms = nn.ModuleList([HypLayerNorm(self.manifold, hid_dim) for _ in range(trans_num_layers)])
        self.use_bn = use_bn
        self.use_residual = use_residual
        self.activation = HypActivation(self.manifold, activation)
        self.dropout = HypDropout(self.manifold, dropout)
        self.output_proj = nn.Linear(hid_dim, out_dim)

    def _mix_residual(self, x, residual):
        x_tan = self.manifold.logmap0(x)[..., 1:]
        r_tan = self.manifold.logmap0(residual)[..., 1:]
        mix_tan = 0.5 * (x_tan + r_tan)
        mix = torch.cat([torch.zeros_like(mix_tan[:, :1]), mix_tan], dim=-1)
        return self.manifold.expmap0(mix)

    def forward(self, x, edge_index, edge_feats=None, **kwargs):
        h_hyp = self.input_proj(x, x_manifold="euc")
        for layer, norm in zip(self.layers, self.norms):
            residual = h_hyp
            h_hyp = layer(h_hyp, edge_index, edge_feats=edge_feats)
            if self.use_residual:
                h_hyp = self._mix_residual(h_hyp, residual)
            if self.use_bn:
                h_hyp = norm(h_hyp)
            h_hyp = self.activation(h_hyp)
            h_hyp = self.dropout(h_hyp)

        h_euc = self.manifold.logmap0(h_hyp)[..., 1:]
        h_out = self.output_proj(h_euc)
        return {"h": h_out}
