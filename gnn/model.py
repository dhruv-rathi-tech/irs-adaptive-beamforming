"""
Phase 3 -- Noise-aware GNN model.

------------------------------------------------------------------
WHY THIS ARCHITECTURE
------------------------------------------------------------------
Each graph (see gnn/dataset.py) has Q=M=64 nodes, one per IRS phase
measurement, FULLY CONNECTED (every node can attend to every other
node's measurement). This is deliberately simple:

  - No PyTorch Geometric dependency (kept out for disk-space reasons
    in this environment -- see Stage 2 report). A fully-connected graph
    with a fixed, small, identical node count every sample is easy to
    express as plain batched tensor ops.
  - Message passing = one round of "every node looks at the (mean of)
    every other node's features, transformed by a shared MLP" -- this
    is a standard, minimal Graph Convolution / GraphSAGE-style mean
    aggregator. It mirrors what the classical LS step already does
    (jointly combining all Q phase measurements to solve for C_k), but
    lets the network learn a NONLINEAR, noise-robust combination
    instead of a fixed linear least-squares solve.
  - Readout: mean-pool over the 64 nodes -> MLP head -> flattened
    (2*M*N,) real vector = the predicted C_hat (real part + imag part).

------------------------------------------------------------------
ARCHITECTURE
------------------------------------------------------------------
Input:  (B, Q=64, F=145) node features
  1. Node encoder:      Linear(F -> H) + ReLU                (per-node)
  2. GraphConv block x2: h_i' = ReLU( W_self h_i + W_nbr * mean_j(h_j) )
     (mean over ALL other nodes j != i -- the fully-connected graph)
  3. Mean-pool over nodes -> (B, H)
  4. Readout MLP:  H -> H -> 2*M*N                            (graph-level)

H (hidden width) and number of GraphConv blocks are kept small
(H=128, 2 blocks) -- this is an academic-review prototype, not a
production model; a small model trains fast and is easy to explain.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class GraphConvBlock(nn.Module):
    """
    One round of mean-aggregation message passing over a fully-connected
    graph (no explicit adjacency needed -- every node aggregates every
    OTHER node, computed via a masked mean for numerical exactness).
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.w_self = nn.Linear(hidden_dim, hidden_dim)
        self.w_nbr = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        # h: (B, Q, H)
        Q = h.shape[1]
        total = h.sum(dim=1, keepdim=True)          # (B, 1, H)
        # mean over all OTHER nodes (exclude self): (total - h_i) / (Q - 1)
        nbr_mean = (total - h) / (Q - 1)             # (B, Q, H)
        out = self.w_self(h) + self.w_nbr(nbr_mean)  # (B, Q, H)
        out = torch.relu(out)
        return self.norm(out + h)  # residual connection + LayerNorm


class NoiseAwareGNN(nn.Module):
    """
    `residual=True` (default, and the formulation that actually works --
    see gnn/dataset.py ARCHITECTURE REFORMULATION note): the network
    predicts a CORRECTION to the LS point-estimate rather than C_k from
    scratch. forward() then takes an additional `c_hat_ls` argument and
    returns c_hat_ls + predicted_correction.

    `residual=False` keeps the original from-scratch prediction mode
    (retained for comparison / documentation of what was tried first).
    """

    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int = 128,
                 n_blocks: int = 2, residual: bool = True):
        super().__init__()
        self.residual = residual
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
        )
        self.blocks = nn.ModuleList([GraphConvBlock(hidden_dim) for _ in range(n_blocks)])
        self.readout = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, node_feats: torch.Tensor, c_hat_ls: torch.Tensor = None) -> torch.Tensor:
        """
        node_feats: (B, Q, F)
        c_hat_ls:   (B, out_dim), required if self.residual is True
        returns:    (B, out_dim)  predicted normalized C_k (real+imag flattened)
        """
        h = self.encoder(node_feats)         # (B, Q, H)
        for block in self.blocks:
            h = block(h)
        pooled = h.mean(dim=1)               # (B, H) -- graph-level readout
        out = self.readout(pooled)           # (B, out_dim)
        if self.residual:
            assert c_hat_ls is not None, "residual=True requires c_hat_ls at forward time"
            return c_hat_ls + out
        return out


if __name__ == "__main__":
    from config.system_config import SystemConfig

    cfg = SystemConfig()
    F = 2 * cfg.M + 2 * cfg.N + 1
    out_dim = 2 * cfg.M * cfg.N

    # residual=True (default) -- requires c_hat_ls at forward time
    model = NoiseAwareGNN(in_dim=F, out_dim=out_dim)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"in_dim={F}, out_dim={out_dim}, total parameters={n_params:,}")

    dummy_feats = torch.randn(3, cfg.M, F)       # batch of 3 graphs
    dummy_c_hat = torch.randn(3, out_dim) * 0.5  # stand-in LS estimate
    out = model(dummy_feats, c_hat_ls=dummy_c_hat)
    print("output shape:", out.shape, "expected (3,", out_dim, ")")
    assert out.shape == (3, out_dim)
    assert torch.isfinite(out).all()
    print("residual-mode forward pass OK, output finite.")

    # Check: with a freshly-initialized (untrained) head, output should be
    # CLOSE to c_hat_ls (small random correction) -- confirms the residual
    # connection is wired correctly (identity-like at init, not overwritten).
    diff = (out - dummy_c_hat).abs().mean().item()
    print(f"mean |output - c_hat_ls| at init: {diff:.4f} (should be small, "
          f"i.e. model starts close to the LS estimate)")

    # non-residual mode still available for reference/comparison
    model_scratch = NoiseAwareGNN(in_dim=F, out_dim=out_dim, residual=False)
    out2 = model_scratch(dummy_feats)
    assert out2.shape == (3, out_dim)
    print("non-residual (from-scratch) mode also OK, output finite:",
          bool(torch.isfinite(out2).all()))
