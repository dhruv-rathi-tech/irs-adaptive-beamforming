"""
Phase 3 -- Deep Beamforming Network for IRS-Assisted Wireless Communications.

Formulation (Candidate A):
  Input:  LS-estimated cascaded channel C_hat_LS in C^(K, M, N) and noise floor.
  Output: IRS phase vector theta_ML in C^(M,) with strictly |theta_m| = 1,
          and active beamformer W_ML in C^(N, K) with ||W_ML||_F^2 = P_max.

Architecture:
  - IRS element-wise representation: M=64 nodes, each node m has features from
    all K users across all N antennas: [Re(C[:, m, :]), Im(C[:, m, :]), power, noise]
  - Multi-layer Message-Passing / Global Context pooling across the M elements.
  - Phase head: outputs (u_m, v_m), theta_m = (u_m + j*v_m) / sqrt(u_m^2 + v_m^2)
  - Power/Regularization head: outputs RZF regularization alpha and per-user power allocation p.
  - Differentiable downstream RZF beamformer construction and true sum-rate calculation.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ElementBlock(nn.Module):
    """
    Message-passing block across the M=64 IRS elements.
    Combines self features with global pooled context across elements.
    """
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_global = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, M, H)
        h_glob = x.mean(dim=1, keepdim=True)  # (B, 1, H)
        h = F.relu(self.fc1(x) + self.fc_global(h_glob))
        h = self.fc2(h)
        return self.norm(x + h)


class DeepBeamformingNet(nn.Module):
    def __init__(self, M: int = 64, N: int = 8, K: int = 4, hidden_dim: int = 128, n_blocks: int = 3):
        super().__init__()
        self.M = M
        self.N = N
        self.K = K
        self.hidden_dim = hidden_dim

        # Input feature dimension per IRS element m:
        # Re and Im of C_hat[k, m, n] for all k in 1..K, n in 1..N: 2 * K * N = 64
        # Plus per-element power (1) + noise level (1) = 66
        in_dim = 2 * K * N + 2

        self.encoder = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        self.blocks = nn.ModuleList([ElementBlock(hidden_dim) for _ in range(n_blocks)])

        # Head 1: per-element phase vector (u_m, v_m) -> unit modulus phase
        self.phase_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
        )

        # Head 2: global context -> per-user power allocation p and RZF regularization alpha
        self.global_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, K + 1),
        )

    def forward_phase(self, c_hat_ls: torch.Tensor, noise_dbm: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        c_hat_ls: (B, K, M, N) complex64
        noise_dbm: (B,) or (B, 1) float32
        Returns:
          theta_ml: (B, M) complex64, unit modulus
          p_alloc:  (B, K) float32, sum(p) == 1
          alpha:    (B, 1) float32, regularization > 0
        """
        B = c_hat_ls.shape[0]
        if noise_dbm.ndim == 1:
            noise_dbm = noise_dbm.unsqueeze(1)  # (B, 1)

        # Normalize C_hat entries to O(1) using standard scale 1e-5
        c_scaled = c_hat_ls / 1e-5

        # Reshape to (B, M, 2*K*N): per element m, gather all k, n
        c_perm = c_scaled.permute(0, 2, 1, 3).reshape(B, self.M, self.K * self.N)
        re_part = c_perm.real
        im_part = c_perm.imag
        elem_power = (re_part ** 2 + im_part ** 2).mean(dim=-1, keepdim=True)

        noise_norm = ((noise_dbm + 85.0) / 15.0).unsqueeze(1).expand(B, self.M, 1)

        node_feats = torch.cat([re_part, im_part, elem_power, noise_norm], dim=-1)  # (B, M, in_dim)

        h = self.encoder(node_feats)
        for block in self.blocks:
            h = block(h)

        # 1. Phase output: (B, M, 2)
        uv = self.phase_head(h)
        u = uv[..., 0]
        v = uv[..., 1]
        norm = torch.sqrt(u ** 2 + v ** 2 + 1e-8)
        theta_real = u / norm
        theta_imag = v / norm
        theta_ml = torch.complex(theta_real, theta_imag)  # (B, M), |theta_m| == 1.0

        # 2. Global head for power allocation and regularization
        h_glob = h.mean(dim=1)  # (B, hidden_dim)
        glob_out = self.global_head(h_glob)  # (B, K + 1)
        p_logits = glob_out[:, :self.K]
        p_alloc = F.softmax(p_logits, dim=-1)  # (B, K), sums to 1

        alpha_raw = glob_out[:, self.K:]  # (B, 1)
        alpha = F.softplus(alpha_raw) + 1e-6  # positive regularization

        return theta_ml, p_alloc, alpha

    def construct_beamformer(self, c_hat_ls: torch.Tensor, theta_ml: torch.Tensor,
                             p_alloc: torch.Tensor, alpha: torch.Tensor,
                             P_max: float = 10.0) -> torch.Tensor:
        """
        Construct RZF beamformer W_ML in C^(B, N, K) satisfying ||W_ML||_F^2 = P_max.
        Effective channel for user k: h_eff_k = theta_ml^T @ C_hat_k
        """
        B, K, M, N = c_hat_ls.shape
        # Compute H_eff: (B, K, N)
        H_eff = torch.einsum('bm,bkmn->bkn', theta_ml, c_hat_ls)  # (B, K, N)

        # RZF: W_unnorm = H_eff^H (H_eff H_eff^H + alpha * I_K)^(-1)
        H_eff_H = H_eff.conj().transpose(-2, -1)  # (B, N, K)
        G_eff = torch.bmm(H_eff, H_eff_H)        # (B, K, K)

        I_K = torch.eye(K, device=c_hat_ls.device, dtype=c_hat_ls.dtype).unsqueeze(0).expand(B, K, K)
        reg_matrix = G_eff + alpha.unsqueeze(-1) * I_K  # (B, K, K)

        inv_reg = torch.linalg.inv(reg_matrix)  # (B, K, K)
        W_unnorm = torch.bmm(H_eff_H, inv_reg)   # (B, N, K)

        # Normalize each user's beamforming column to unit norm
        col_norms = torch.sqrt(torch.sum(W_unnorm.real ** 2 + W_unnorm.imag ** 2, dim=1, keepdim=True) + 1e-12)
        W_dir = W_unnorm / col_norms  # (B, N, K) with unit column norm

        # Scale by per-user power allocation: p_alloc sums to 1, total power = P_max
        p_weights = torch.sqrt(P_max * p_alloc).unsqueeze(1)  # (B, 1, K)
        W_ml = W_dir * p_weights  # (B, N, K)

        return W_ml

    def forward(self, c_hat_ls: torch.Tensor, noise_dbm: torch.Tensor, P_max: float = 10.0) -> tuple[torch.Tensor, torch.Tensor]:
        """
        End-to-end inference pass.
        Returns:
          theta_ml: (B, M) complex64
          W_ml:     (B, N, K) complex64
        """
        theta_ml, p_alloc, alpha = self.forward_phase(c_hat_ls, noise_dbm)
        W_ml = self.construct_beamformer(c_hat_ls, theta_ml, p_alloc, alpha, P_max=P_max)
        return theta_ml, W_ml


def compute_true_sum_rate_torch(H_true: torch.Tensor, G_true: torch.Tensor,
                                theta: torch.Tensor, W: torch.Tensor,
                                sigma_watts: float) -> torch.Tensor:
    """
    Differentiable true sum-rate calculation in PyTorch.
    H_true: (B, M, K) complex64
    G_true: (B, M, N) complex64
    theta:  (B, M) complex64, unit modulus
    W:      (B, N, K) complex64, transmit beamformer
    sigma_watts: data-phase receiver noise power (scalar float)

    H_all_true(k, :) = H_true(:, k)' * diag(conj(theta)) * G_true
    """
    B, M, K = H_true.shape
    N = G_true.shape[-1]

    # Effective true channel: H_all_true: (B, K, N)
    H_all_true = torch.einsum('bmk,bm,bmn->bkn', H_true.conj(), theta.conj(), G_true)  # (B, K, N)

    # Received signal matrix across users:
    # S_mat[b, k, j] = H_all_true[b, k, :] @ W[b, :, j]
    S_mat = torch.bmm(H_all_true, W)  # (B, K, K)
    S_power = S_mat.real ** 2 + S_mat.imag ** 2  # (B, K, K)

    # Desired signal power for user k: S_power[:, k, k]
    desired_power = torch.diagonal(S_power, dim1=1, dim2=2)  # (B, K)

    # Total received power for user k: sum_j S_power[:, k, j]
    total_power = torch.sum(S_power, dim=2)  # (B, K)
    interf_power = total_power - desired_power  # (B, K)

    sinr = desired_power / (interf_power + sigma_watts)  # (B, K)
    user_rates = torch.log2(1.0 + sinr)                  # (B, K)
    sum_rate = torch.sum(user_rates, dim=1)              # (B,)

    return sum_rate
