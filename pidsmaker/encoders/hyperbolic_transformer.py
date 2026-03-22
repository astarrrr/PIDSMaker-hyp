import torch
import torch.nn as nn
import torch.nn.functional as F
from geoopt import Lorentz
from torch_geometric.utils import scatter, softmax


class HyperbolicLorentz(Lorentz):
    """Minimal Lorentz wrapper with helpers used by the local transformer."""

    def cinner(self, x, y):
        x = x.clone()
        x.narrow(-1, 0, 1).mul_(-1)
        return (x * y).sum(dim=-1)

    def weighted_midpoint(self, x, index, weights, dim_size):
        weighted = x * weights.unsqueeze(-1)
        ave = scatter(weighted, index, dim=0, dim_size=dim_size, reduce="sum")
        denom = (-self.cinner(ave, ave)).abs().clamp_min(1e-8).sqrt().unsqueeze(-1)
        return self.k.sqrt() * ave / denom

    def midpoint(self, x, dim=-2):
        ave = x.mean(dim=dim)
        denom = (-self.cinner(ave, ave)).abs().clamp_min(1e-8).sqrt().unsqueeze(-1)
        return self.k.sqrt() * ave / denom


class HypLinear(nn.Module):
    """Lorentz linear layer following the reference project's coordinate style."""

    def __init__(self, manifold, in_features, out_features, bias=True):
        super().__init__()
        self.manifold = manifold
        self.in_features = in_features
        self.out_features = out_features
        self.linear = nn.Linear(in_features + 1, out_features, bias=bias)

    def forward(self, x, x_manifold="hyp"):
        if x_manifold != "hyp":
            zeros = torch.zeros_like(x[..., :1])
            x = torch.cat([zeros, x], dim=-1)
            x = self.manifold.expmap0(x)

        x_space = self.linear(x)
        x_time = ((x_space**2).sum(dim=-1, keepdim=True) + self.manifold.k).sqrt()
        return torch.cat([x_time, x_space], dim=-1)


class HypLayerNorm(nn.Module):
    def __init__(self, manifold, in_features):
        super().__init__()
        self.manifold = manifold
        self.norm = nn.LayerNorm(in_features)

    def forward(self, x):
        x_space = self.norm(x[..., 1:])
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


class HyperbolicTransformerLayer(nn.Module):
    """Local graph attention layer in Lorentz space over edge_index neighborhoods."""

    def __init__(self, manifold, hidden_dim, num_heads, dropout, edge_dim=None):
        super().__init__()
        self.manifold = manifold
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")

        self.q_proj = nn.ModuleList(
            [HypLinear(manifold, hidden_dim, self.head_dim) for _ in range(num_heads)]
        )
        self.k_proj = nn.ModuleList(
            [HypLinear(manifold, hidden_dim, self.head_dim) for _ in range(num_heads)]
        )
        self.v_proj = nn.ModuleList(
            [HypLinear(manifold, hidden_dim, self.head_dim) for _ in range(num_heads)]
        )
        self.out_proj = HypLinear(manifold, hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.edge_bias = nn.Linear(edge_dim, num_heads, bias=False) if edge_dim else None

    def forward(self, x, edge_index, edge_feats=None):
        src, dst = edge_index
        num_nodes = x.shape[0]

        q = torch.stack([proj(x) for proj in self.q_proj], dim=1)
        k = torch.stack([proj(x) for proj in self.k_proj], dim=1)
        v = torch.stack([proj(x) for proj in self.v_proj], dim=1)

        scores = 2 + 2 * self.manifold.cinner(q[dst], k[src])
        scores = scores / (self.head_dim**0.5)
        if edge_feats is not None and self.edge_bias is not None:
            scores = scores + self.edge_bias(edge_feats)

        attn = softmax(scores, dst, num_nodes=num_nodes)
        attn = self.dropout(attn)
        out = self.manifold.weighted_midpoint(v[src], dst, attn, dim_size=num_nodes)
        out_space = out[..., 1:].reshape(num_nodes, self.hidden_dim)
        out_time = ((out_space**2).sum(dim=-1, keepdim=True) + self.manifold.k).sqrt()
        out = torch.cat([out_time, out_space], dim=-1)
        return self.out_proj(out)


class HyperbolicTemporalTransformer(nn.Module):
    """Hyperbolic local transformer that returns Euclidean embeddings by default."""

    def __init__(
        self,
        in_dim,
        hid_dim,
        out_dim,
        edge_dim,
        dropout,
        activation,
        num_heads,
        num_layers,
        curvature,
        project_back_to_euclidean=True,
    ):
        super().__init__()
        self.manifold = HyperbolicLorentz(k=curvature)
        self.project_back_to_euclidean = project_back_to_euclidean

        self.input_proj = HypLinear(self.manifold, in_dim, hid_dim)
        self.input_norm = HypLayerNorm(self.manifold, hid_dim)
        self.activation = HypActivation(self.manifold, activation)
        self.dropout = HypDropout(self.manifold, dropout)

        self.layers = nn.ModuleList(
            [
                HyperbolicTransformerLayer(
                    manifold=self.manifold,
                    hidden_dim=hid_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                    edge_dim=edge_dim,
                )
                for _ in range(num_layers)
            ]
        )
        self.norms = nn.ModuleList([HypLayerNorm(self.manifold, hid_dim) for _ in range(num_layers)])
        self.output_proj = HypLinear(self.manifold, hid_dim, out_dim)

    def _midpoint_residual(self, x, residual):
        return self.manifold.midpoint(torch.stack([x, residual], dim=1), dim=1)

    def forward(self, x, edge_index, edge_feats=None, **kwargs):
        x = self.input_proj(x, x_manifold="euc")
        x = self.input_norm(x)
        x = self.activation(x)
        x = self.dropout(x)

        for layer, norm in zip(self.layers, self.norms):
            residual = x
            x = layer(x, edge_index, edge_feats=edge_feats)
            x = self._midpoint_residual(x, residual)
            x = norm(x)

        x = self.output_proj(x)
        if self.project_back_to_euclidean:
            x = self.manifold.logmap0(x)[..., 1:]
        return {"h": x}
