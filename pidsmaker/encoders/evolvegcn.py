"""EvolveGCN-O encoder.

This implementation keeps the layer weights as recurrent states and evolves them
across temporal graph windows. The recurrent state is detached after each
forward pass so it can be used with the framework's per-window optimization
loop without backpropagating through the full training history.
"""

import torch
import torch.nn as nn
from torch_geometric.utils import add_remaining_self_loops, degree


class _EvolveGCNOLayer(nn.Module):
    def __init__(self, in_dim, out_dim, activation, dropout):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.activation = activation
        self.dropout = nn.Dropout(dropout)

        self.initial_weight = nn.Parameter(torch.empty(in_dim, out_dim))
        nn.init.xavier_uniform_(self.initial_weight)

        flat_dim = in_dim * out_dim
        self.recurrent = nn.GRUCell(flat_dim, flat_dim)
        self._weight_state = None

    def reset_state(self):
        self._weight_state = None

    def _evolve_weight(self, device):
        prev_weight = self._weight_state
        if prev_weight is None:
            prev_weight = self.initial_weight
        else:
            prev_weight = prev_weight.to(device)

        prev_flat = prev_weight.reshape(1, -1)
        next_flat = self.recurrent(prev_flat, prev_flat)
        next_weight = next_flat.view(self.in_dim, self.out_dim)

        # Keep temporal state across windows while avoiding graph retention.
        self._weight_state = next_weight.detach()
        return next_weight

    def forward(self, x, edge_index):
        num_nodes = x.size(0)
        weight = self._evolve_weight(x.device)
        edge_index, _ = add_remaining_self_loops(edge_index, num_nodes=num_nodes)
        row, col = edge_index

        deg = degree(col, num_nodes=num_nodes, dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt.masked_fill_(torch.isinf(deg_inv_sqrt), 0)
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        x = x @ weight
        out = x.new_zeros((num_nodes, self.out_dim))
        out.index_add_(0, col, x[row] * norm.unsqueeze(-1))
        out = self.activation(out)
        out = self.dropout(out)
        return out


class EvolveGCNO(nn.Module):
    """EvolveGCN-O encoder with recurrently updated GCN weights."""

    def __init__(self, in_dim, hid_dim, out_dim, activation, dropout, num_layers):
        super().__init__()
        self.layers = nn.ModuleList()

        current_dim = in_dim
        for _ in range(num_layers - 1):
            self.layers.append(
                _EvolveGCNOLayer(
                    in_dim=current_dim,
                    out_dim=hid_dim,
                    activation=activation,
                    dropout=dropout,
                )
            )
            current_dim = hid_dim

        self.layers.append(
            _EvolveGCNOLayer(
                in_dim=current_dim,
                out_dim=out_dim,
                activation=nn.Identity(),
                dropout=0.0,
            )
        )

    def reset_state(self):
        for layer in self.layers:
            layer.reset_state()

    def forward(self, x, edge_index, **kwargs):
        for layer in self.layers:
            x = layer(x, edge_index)
        return {"h": x}
