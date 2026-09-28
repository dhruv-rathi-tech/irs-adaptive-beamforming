"""
Phase 3 -- GNN dataset / graph construction.

Builds one graph PER USER PER CHANNEL REALIZATION from the exact same
multi-phase pilot pipeline already used by the LS baseline
(simulation/multi_phase_estimation.py), so the GNN and the classical
baseline consume IDENTICAL underlying noisy observations -- only the
estimator differs.

------------------------------------------------------------------
WHAT THE GNN SEES (per graph)
------------------------------------------------------------------
During pilot training, Q = M = 64 distinct random IRS phase
configurations theta_q are used. For each one, the classical pipeline
already computes a per-phase LS estimate of the effective channel:

    h_eff_hat_stack[q, k, :]  (shape (N,), complex)

for every user k. This is exactly the noisy, per-measurement quantity
that the *second* LS step (Theta_stack @ C_k = h_eff_hat_stack) turns
into the final estimate C_hat[k]. Our GNN replaces that second LS step
with a learned, noise-aware graph aggregation.

Per user k, we build a graph with:
  - Q = M = 64 nodes, one per IRS phase configuration.
  - Node features (real-valued, dim = 2M + 2N + 1 = 145):
        [Re(theta_q) (M,), Im(theta_q) (M,),
         Re(h_eff_hat_q) (N,), Im(h_eff_hat_q) (N,),
         snr_db (1,) ]
    snr_db is broadcast (same scalar on every node of this graph) so the
    network can explicitly condition on the noise level -- this is what
    makes training "noise-aware" rather than fixed-SNR.
  - Fully connected adjacency (all 64x64 pairs, including self-loops).
    64 nodes -> a complete graph is cheap and lets the network jointly
    reason across all phase measurements, mirroring what the joint LS
    solve already does.
  - Target: C_true[k] (M, N) complex, flattened to a real vector of
    length 2*M*N = 1024 (real part then imaginary part).

------------------------------------------------------------------
NOISE-AWARE TRAINING
------------------------------------------------------------------
Each sample (each call to `make_graph_sample`) draws its own
noise_power_dbm from a configurable range (default: uniform over the
SNR sweep already used in Phase 2, -100 to -70 dBm) instead of a fixed
value. This directly implements the "noise-aware" requirement: the
model is exposed to many noise conditions during training, not one.

------------------------------------------------------------------
REUSE, NOT REWRITE
------------------------------------------------------------------
This module calls simulation.channel and simulation.multi_phase_estimation
UNCHANGED. No Phase 1/2 file is modified.
"""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

from config.system_config import SystemConfig
from simulation.channel import generate_bs_irs_channel, generate_irs_user_channels, path_loss_linear
from simulation.multi_phase_estimation import multi_phase_pilot_estimate_C


def channel_magnitude_scale(cfg: SystemConfig) -> float:
    """
    Analytic (not data-fit) magnitude scale of a C_k = diag(conj(h_k)) @ G
    entry, derived directly from the physical channel model in
    simulation/channel.py:
        |G entry|  ~ gain_linear * sqrt(PL_BI)
        |H entry|  ~ gain_linear * sqrt(PL_IU)
        |C_k entry| ~ |H entry| * |G entry| ~ gain_linear^2 * sqrt(PL_BI * PL_IU)
    Uses the midpoint IRS-user distance for PL_IU since d_IU is randomized
    per user. This constant is FIXED (computed once from SystemConfig, not
    fit to any particular batch/dataset), so normalizing by it does not
    leak per-sample target statistics into the model's inputs.
    h_eff_hat entries share this same scale (h_eff = h^H Phi G is a sum of
    M such products divided by ~sqrt(M) in expectation after averaging,
    same order of magnitude for our purposes -- both quantities are
    normalized by the identical constant below, which is what matters for
    numerical conditioning).
    """
    gain_linear = 10 ** (cfg.element_gain_db / 20.0)
    pl_BI = path_loss_linear(np.array([cfg.d_BI]), cfg.C0_dB, cfg.d0, cfg.alpha_BI)[0]
    d_IU_mid = (cfg.d_IU_min + cfg.d_IU_max) / 2.0
    pl_IU_mid = path_loss_linear(np.array([d_IU_mid]), cfg.C0_dB, cfg.d0, cfg.alpha_IU)[0]
    return float(gain_linear ** 2 * np.sqrt(pl_BI * pl_IU_mid))


# Fixed normalization constants (derived once from the physical model above,
# NOT fit per-batch/per-sample -- see channel_magnitude_scale docstring).
# Rounded to a clean value close to the analytic estimate (~8.2e-6).
CHANNEL_SCALE = 1e-5
SNR_DBM_LO, SNR_DBM_HI = -100.0, -70.0  # matches the Phase 2 sweep range


def _build_h_eff_hat_stack(G, H_true, cfg: SystemConfig, rng, noise_power_dbm: float):
    """
    Re-derive h_eff_hat_stack (Q, K, N) and thetas (Q, M) exactly as
    multi_phase_pilot_estimate_C does internally, but expose the
    intermediate per-phase LS estimate (that function only returns the
    FINAL C_hat, not this intermediate) -- needed as GNN node features.

    This duplicates ~15 lines of arithmetic from multi_phase_estimation.py
    rather than modifying that file. Kept algebraically IDENTICAL (same
    RNG call order) so C_true / recovery_err from the original function
    remain valid references for direct comparison.
    """
    from simulation.pilot import random_irs_phase, orthogonal_pilot_matrix, dbm_to_watts, compute_received_snr_db

    M, N, K = cfg.M, cfg.N, cfg.K
    tau = cfg.tau
    Q = M

    X_p = orthogonal_pilot_matrix(N, tau, rng)
    sigma_n2 = dbm_to_watts(noise_power_dbm)

    thetas = np.zeros((Q, M), dtype=np.complex128)
    h_eff_hat_stack = np.zeros((Q, K, N), dtype=np.complex128)
    sig_power_accum = np.zeros(K)

    for q in range(Q):
        theta_q = random_irs_phase(M, rng)
        thetas[q] = theta_q
        Phi = np.diag(theta_q)

        h_eff_true = np.zeros((K, N), dtype=np.complex128)
        for k in range(K):
            h_eff_true[k, :] = np.conj(H_true[:, k]) @ Phi @ G

        Y_clean = h_eff_true @ X_p
        noise = np.sqrt(sigma_n2 / 2) * (
            rng.standard_normal((K, tau)) + 1j * rng.standard_normal((K, tau))
        )
        Y_noisy = Y_clean + noise
        sig_power_accum += np.mean(np.abs(Y_clean) ** 2, axis=1)

        h_eff_hat_stack[q] = (N / tau) * (Y_noisy @ X_p.conj().T)

    avg_received_snr_db = compute_received_snr_db(sig_power_accum / Q, sigma_n2)
    C_true = [np.diag(np.conj(H_true[:, k])) @ G for k in range(K)]

    return thetas, h_eff_hat_stack, avg_received_snr_db, C_true


def make_graph_sample(cfg: SystemConfig, rng: np.random.Generator, noise_power_dbm: float):
    """
    Build ONE channel realization and return graphs for ALL K users.

    Returns:
      node_feats : (K, Q, F) float32   F = 2M + 2N + 1
      targets    : (K, 2*M*N) float32  (real part then imag part, flattened) -- normalized C_true
      snr_db     : (K,) float32        emergent per-user received SNR (for logging)
      C_hat_ls   : (K, 2*M*N) float32  normalized LS point-estimate (the SAME estimate the
                   classical Phase 2 baseline uses) -- exposed so the GNN can be trained as
                   a RESIDUAL CORRECTION on top of LS instead of predicting C_true from
                   scratch. See ARCHITECTURE REFORMULATION note below.

    ------------------------------------------------------------------
    ARCHITECTURE REFORMULATION (Stage 3 debugging finding)
    ------------------------------------------------------------------
    An earlier version of this pipeline asked the GNN to predict C_true
    directly from the raw per-phase measurements. Diagnostic experiments
    showed this fails to beat a trivial zero-prediction baseline even
    with a tiny fixed-noise dataset the model could otherwise memorize
    easily (16-example overfit test: NMSE 1.03 -> 0.03; full dataset,
    single fixed realistic noise level -85dBm: NMSE stuck at ~1.0-1.04
    for 120 epochs). Root cause: at low noise the LS estimator is
    already near-exact (NMSE ~0.008 at -85dBm), so almost the entire
    target signal is "linear-algebra-recoverable" and a mean-pooled GNN
    has to rediscover that linear algebra from scratch with no
    inductive bias for it -- an unnecessarily hard learning problem.
    Exposing C_hat_ls as an input and framing the task as LEARNING A
    NOISE-DEPENDENT CORRECTION to it is both an easier regression
    target (the residual is typically much smaller than C_true itself)
    and the scientifically correct framing for a "noise-aware"
    contribution: the model's job becomes "when is LS untrustworthy,
    and how should its estimate be adjusted" rather than "reinvent
    channel estimation".
    """
    M, N, K = cfg.M, cfg.N, cfg.K

    G = generate_bs_irs_channel(cfg, rng)
    H_true, _ = generate_irs_user_channels(cfg, rng)

    thetas, h_eff_hat_stack, snr_db, C_true = _build_h_eff_hat_stack(
        G, H_true, cfg, rng, noise_power_dbm
    )
    Q = thetas.shape[0]  # = M

    theta_re = np.real(thetas)  # (Q, M)
    theta_im = np.imag(thetas)  # (Q, M)

    node_feats = np.zeros((K, Q, 2 * M + 2 * N + 1), dtype=np.float32)
    targets = np.zeros((K, 2 * M * N), dtype=np.float32)
    C_hat_ls = np.zeros((K, 2 * M * N), dtype=np.float32)

    for k in range(K):
        # Normalize by the FIXED analytic channel scale (see channel_magnitude_scale) --
        # brings both h_eff_hat and C_true onto an O(1) range without using any
        # per-sample statistic, so no target information leaks into the input.
        h_re = np.real(h_eff_hat_stack[:, k, :]) / CHANNEL_SCALE  # (Q, N)
        h_im = np.imag(h_eff_hat_stack[:, k, :]) / CHANNEL_SCALE  # (Q, N)
        # snr_db is the actual RECEIVED snr (tens of dB), already a reasonable
        # scale; z-score with a fixed, hand-computed range for conditioning.
        snr_norm = (snr_db[k] - 20.0) / 20.0  # received SNR typically ~0-50 dB in our sweeps
        snr_col = np.full((Q, 1), snr_norm, dtype=np.float32)

        node_feats[k] = np.concatenate(
            [theta_re, theta_im, h_re, h_im, snr_col], axis=1
        ).astype(np.float32)

        C_k = C_true[k]  # (M, N) complex
        targets[k] = np.concatenate(
            [np.real(C_k).flatten() / CHANNEL_SCALE, np.imag(C_k).flatten() / CHANNEL_SCALE]
        ).astype(np.float32)

        # LS point estimate -- IDENTICAL computation to multi_phase_pilot_estimate_C's
        # C_hat (same lstsq solve of Theta_stack @ C_k = h_eff_hat_stack[:,k,:]).
        rhs = h_eff_hat_stack[:, k, :]  # (Q, N)
        C_k_hat, *_ = np.linalg.lstsq(thetas, rhs, rcond=None)  # (M, N)
        C_hat_ls[k] = np.concatenate(
            [np.real(C_k_hat).flatten() / CHANNEL_SCALE, np.imag(C_k_hat).flatten() / CHANNEL_SCALE]
        ).astype(np.float32)

    return node_feats, targets, snr_db.astype(np.float32), C_hat_ls


class IRSGraphDataset(Dataset):
    """
    Pre-generates `n_samples` channel realizations ONCE at construction
    time (each with its own randomly drawn noise floor, uniform over
    `noise_dbm_range` -- the noise-aware mixed-SNR training signal), then
    reuses this FIXED pool across all training epochs.

    IMPORTANT lesson learned during Stage 3 debugging: an earlier version
    of this class generated a brand-new random realization on every
    __getitem__ call, so every epoch saw entirely fresh data and the
    model never got repeated exposure to any example -- with only ~20
    epochs before early stopping, this starved the optimizer and the
    loss barely moved. A tiny fixed-data overfitting test confirmed the
    model/loss/architecture themselves are fine (NMSE 1.03 -> 0.03 over
    300 epochs on 16 fixed examples) -- the bug was in HOW data was
    supplied, not the model. Pre-generating a pool and reusing it (like
    a normal finite dataset) fixes this while still being noise-aware,
    since each of the `n_samples` realizations already has its own
    random noise floor baked in.

    A fixed `seed_offset` makes the pool reproducible.
    """

    def __init__(self, cfg: SystemConfig, n_samples: int,
                 noise_dbm_range=(-100.0, -70.0), base_seed: int = 0):
        self.cfg = cfg
        self.n_samples = n_samples
        self.noise_dbm_range = noise_dbm_range
        self.base_seed = base_seed

        self._feats = []
        self._targets = []
        self._snr = []
        self._c_hat_ls = []
        for idx in range(n_samples):
            rng = np.random.default_rng(base_seed + idx)
            noise_dbm = rng.uniform(*noise_dbm_range)
            node_feats, targets, snr_db, c_hat_ls = make_graph_sample(cfg, rng, float(noise_dbm))
            self._feats.append(torch.from_numpy(node_feats))
            self._targets.append(torch.from_numpy(targets))
            self._snr.append(torch.from_numpy(snr_db))
            self._c_hat_ls.append(torch.from_numpy(c_hat_ls))

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        # Returns the SAME pre-generated sample every time idx is requested
        # (standard finite-dataset behavior -- enables multiple epochs over
        # the same pool, which fresh-every-call generation did not allow).
        return self._feats[idx], self._targets[idx], self._snr[idx], self._c_hat_ls[idx]


def collate_user_graphs(batch):
    """
    Each dataset item already contains K graphs (K, Q, F). Concatenate
    across the batch dimension so a DataLoader batch of size B yields
    (B*K, Q, F) -- i.e. every user-graph is treated as an independent
    training example. Simple and avoids needing a padded/batched-graph
    format since every graph has identical Q (=M) nodes.
    """
    feats = torch.cat([b[0] for b in batch], dim=0)      # (B*K, Q, F)
    targets = torch.cat([b[1] for b in batch], dim=0)    # (B*K, 2MN)
    snr_db = torch.cat([b[2] for b in batch], dim=0)     # (B*K,)
    c_hat_ls = torch.cat([b[3] for b in batch], dim=0)   # (B*K, 2MN)
    return feats, targets, snr_db, c_hat_ls


if __name__ == "__main__":
    cfg = SystemConfig()
    rng = np.random.default_rng(cfg.seed)
    node_feats, targets, snr_db, c_hat_ls = make_graph_sample(cfg, rng, noise_power_dbm=-85.0)
    print("node_feats shape:", node_feats.shape, "expected (K, Q, F) =",
          (cfg.K, cfg.M, 2 * cfg.M + 2 * cfg.N + 1))
    print("targets shape:", targets.shape, "expected (K, 2*M*N) =", (cfg.K, 2 * cfg.M * cfg.N))
    print("c_hat_ls shape:", c_hat_ls.shape)
    print("snr_db:", np.round(snr_db, 2))
    print("node_feats has NaN/Inf:", not np.all(np.isfinite(node_feats)))
    print("targets has NaN/Inf:", not np.all(np.isfinite(targets)))
    print("c_hat_ls has NaN/Inf:", not np.all(np.isfinite(c_hat_ls)))

    # Sanity: LS-vs-true relative error computed from c_hat_ls/targets should
    # match the recovery_err reported by the original Phase 2 estimator.
    err = np.linalg.norm(c_hat_ls - targets) / np.linalg.norm(targets)
    print(f"LS-vs-true relative error (all K stacked): {err:.4f}  (sanity cross-check)")

    ds = IRSGraphDataset(cfg, n_samples=4, base_seed=123)
    f, t, s, c = ds[0]
    print("\nDataset item 0 -> feats:", f.shape, "targets:", t.shape,
          "snr:", s.shape, "c_hat_ls:", c.shape)
