import torch
import torch.nn as nn

from .tgn_encoder import TGNEncoder


class EarlyFusionEncoder(nn.Module):
    def __init__(
        self,
        context_encoder,
        instant_encoder,
        out_dim,
        mode="fusion",
        gate_hidden_dim=None,
    ):
        super().__init__()
        self.context_encoder = context_encoder
        self.instant_encoder = instant_encoder
        self.mode = mode

        gate_hidden_dim = gate_hidden_dim or out_dim
        self.gate = nn.Sequential(
            nn.Linear(out_dim * 2, gate_hidden_dim),
            nn.ReLU(),
            nn.Linear(gate_hidden_dim, out_dim),
            nn.Sigmoid(),
        )

    def forward(self, batch, inference=False, **kwargs):
        current_edge_index = self._get_current_batch_edge_index(batch)
        context_res = self.context_encoder(batch=batch, inference=inference, **kwargs)
        instant_res = self.instant_encoder(
            x=batch.x,
            edge_index=current_edge_index,
            edge_feats=getattr(batch, "edge_feats", None),
        )

        h_ctx = context_res["h"]
        h_inst = instant_res["h"]

        if isinstance(h_ctx, (tuple, list)) or isinstance(h_inst, (tuple, list)):
            raise TypeError("EarlyFusionEncoder expects tensor node embeddings for both branches.")

        if self.mode == "context_only":
            h = h_ctx
        elif self.mode == "instant_only":
            h = h_inst
        elif self.mode == "fusion":
            gate = self.gate(torch.cat([h_inst, h_ctx], dim=-1))
            h = gate * h_inst + (1.0 - gate) * h_ctx
        else:
            raise ValueError(f"Invalid early fusion mode {self.mode}")

        max_edge_index = int(current_edge_index.max().item()) if current_edge_index.numel() > 0 else -1
        if max_edge_index >= h.size(0):
            raise IndexError(
                f"Current-batch edge index expects {max_edge_index + 1} nodes, but fused "
                f"representation only has {h.size(0)} nodes."
            )

        h_src = h[current_edge_index[0]]
        h_dst = h[current_edge_index[1]]
        return {"h": h, "h_src": h_src, "h_dst": h_dst}

    def reset_state(self):
        if hasattr(self.context_encoder, "reset_state"):
            self.context_encoder.reset_state()

    def to_device(self, device):
        if hasattr(self.context_encoder, "to_device"):
            self.context_encoder.to_device(device)
        return self

    def get_tgn_encoder(self):
        if isinstance(self.context_encoder, TGNEncoder):
            return self.context_encoder
        if hasattr(self.context_encoder, "get_tgn_encoder"):
            return self.context_encoder.get_tgn_encoder()
        return None

    def _get_current_batch_edge_index(self, batch):
        if hasattr(batch, "reindexed_edge_index"):
            return batch.reindexed_edge_index

        if hasattr(batch, "original_n_id"):
            assoc = torch.empty(
                int(batch.original_n_id.max().item()) + 1,
                dtype=torch.long,
                device=batch.original_n_id.device,
            )
            assoc[batch.original_n_id] = torch.arange(
                batch.original_n_id.size(0), device=batch.original_n_id.device
            )
            return assoc[batch.edge_index]

        return batch.edge_index
