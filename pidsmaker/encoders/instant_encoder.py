import torch.nn as nn

from .graph_attention import GraphAttentionEmbedding
from .linear_encoder import LinearEncoder


class InstantEncoder(nn.Module):
    def __init__(
        self,
        in_dim,
        hid_dim,
        out_dim,
        dropout,
        activation,
        edge_dim,
        method="graph_attention",
        use_edge_features=True,
    ):
        super().__init__()
        self.method = method
        self.use_edge_features = use_edge_features

        if method == "graph_attention":
            self.encoder = GraphAttentionEmbedding(
                in_dim=in_dim,
                hid_dim=hid_dim,
                out_dim=out_dim,
                edge_dim=edge_dim if use_edge_features else None,
                activation=activation,
                dropout=dropout,
                num_heads=1,
                concat=False,
                num_layers=1,
            )
        elif method == "mlp":
            self.encoder = LinearEncoder(in_dim=in_dim, out_dim=out_dim, dropout=dropout)
        else:
            raise ValueError(f"Invalid instant branch encoder {method}")

    def forward(self, x, edge_index, edge_feats=None, **kwargs):
        edge_feats = edge_feats if self.use_edge_features else None
        if self.method == "graph_attention":
            return self.encoder(x=x, edge_index=edge_index, edge_feats=edge_feats, **kwargs)
        return self.encoder(x=x, edge_index=edge_index, edge_feats=edge_feats, **kwargs)
