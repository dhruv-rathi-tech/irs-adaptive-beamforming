"""
Task 3 -- Generalization to Alternative Dimensions (N, M, K).

Evaluates scalability and flexibility:
  1. IRS Surface Scaling (M in {32, 64, 128}):
     - Zero-shot evaluation of frozen model (element-permutation-equivariant architecture)
     - Retrained model evaluation
     - Baseline Random IRS + RZF
  2. BS Antenna Scaling (N in {4, 8, 16}):
     - Retrained model vs Baseline Random IRS + RZF
  3. User Count Scaling (K in {2, 4, 8}):
     - Retrained model vs Baseline Random IRS + RZF

Computes both raw sum-rate (bps/Hz), per-user spectral efficiency (bps/Hz/user),
and per-antenna spectral efficiency (bps/Hz/antenna) for fair normalization.
"""
from __future__ import annotations

import json
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from config.system_config import SystemConfig
from gnn.beamforming_dataset import BeamformingDataset
from gnn.beamforming_model import DeepBeamformingNet, compute_true_sum_rate_torch
from simulation.channel import generate_bs_irs_channel, generate_irs_user_channels
from simulation.multi_phase_estimation import multi_phase_pilot_estimate_C

OUT_DIR = "results/task3_generalization"


def evaluate_configuration(cfg: SystemConfig, model: DeepBeamformingNet, n_samples: int = 100,
                           base_seed: int = 400000, noise_dbm: float = -85.0) -> tuple[float, float, float]:
    """Evaluates model and random RZF baseline on n_samples held-out realizations."""
    device = torch.device("cpu")
    sigma_watts = 3.16e-12

    model_rates = []
    rand_rates = []

    model.eval()
    with torch.no_grad():
        for i in range(n_samples):
            rng = np.random.default_rng(base_seed + i)
            G = generate_bs_irs_channel(cfg, rng)
            H, _ = generate_irs_user_channels(cfg, rng)
            est = multi_phase_pilot_estimate_C(G, H, cfg, rng, noise_power_dbm=noise_dbm)
            c_hat_np = np.stack(est["C_hat"], axis=0)

            ch_i = torch.from_numpy(c_hat_np).unsqueeze(0).to(torch.complex64).to(device)
            n_i = torch.tensor([noise_dbm], dtype=torch.float32).to(device)
            G_i = torch.from_numpy(G).unsqueeze(0).to(torch.complex64).to(device)
            H_i = torch.from_numpy(H).unsqueeze(0).to(torch.complex64).to(device)

            # Model forward
            th_m, W_m = model(ch_i, n_i, P_max=cfg.pilot_power)
            r_m = compute_true_sum_rate_torch(H_i, G_i, th_m, W_m, sigma_watts).item()
            model_rates.append(r_m)

            # Random baseline
            rng_r = np.random.default_rng(base_seed + i)
            th_r = torch.from_numpy(np.exp(1j * rng_r.uniform(0, 2*np.pi, (1, cfg.M)))).to(torch.complex64).to(device)
            H_eff = torch.einsum("bm,bkmn->bkn", th_r.conj(), ch_i)
            H_eff_H = H_eff.conj().transpose(-2, -1)
            G_eff = torch.bmm(H_eff, H_eff_H)
            I_K = torch.eye(cfg.K, dtype=torch.complex64, device=device).unsqueeze(0)
            inv_reg = torch.linalg.inv(G_eff + (cfg.K * sigma_watts / cfg.pilot_power) * I_K)
            W_unnorm = torch.bmm(H_eff_H, inv_reg)
            col_norms = torch.sqrt(torch.sum(W_unnorm.real**2 + W_unnorm.imag**2, dim=1, keepdim=True) + 1e-12)
            W_rzf = (W_unnorm / col_norms) * np.sqrt(cfg.pilot_power / cfg.K)
            r_r = compute_true_sum_rate_torch(H_i, G_i, th_r, W_rzf, sigma_watts).item()
            rand_rates.append(r_r)

    return float(np.mean(model_rates)), float(np.std(model_rates)), float(np.mean(rand_rates))


def run_dimension_generalization():
    os.makedirs(OUT_DIR, exist_ok=True)
    device = torch.device("cpu")
    sigma_watts = 3.16e-12

    # Frozen default model (M=64, N=8, K=4)
    frozen_model = DeepBeamformingNet(M=64, N=8, K=4, hidden_dim=64, n_blocks=2).to(device)
    frozen_model.load_state_dict(torch.load("results/gnn/best_beamforming_model.pt", map_location=device))
    frozen_model.eval()

    results_M = []
    results_N = []
    results_K = []

    # -------------------------------------------------------------
    # 1. IRS Elements M in {32, 64, 128} (Zero-shot vs Retrained)
    # -------------------------------------------------------------
    print("=" * 80)
    print("EVALUATING IRS ELEMENTS SCALING: M in {32, 64, 128}")
    print("=" * 80)
    for M_val in [32, 64, 128]:
        cfg_M = SystemConfig(M=M_val, N=8, K=4)
        print(f"\n--- Dimension M = {M_val} (N=8, K=4) ---")

        # Zero-shot evaluation with frozen model
        zs_mean, zs_std, rand_mean = evaluate_configuration(cfg_M, frozen_model, n_samples=80, base_seed=410000)

        # Retrained model for this M
        if M_val == 64:
            rt_mean, rt_std = zs_mean, zs_std
        else:
            cache_file = os.path.join(OUT_DIR, f"train_data_M_{M_val}.pt")
            train_ds = BeamformingDataset(cfg_M, n_samples=200, base_seed=420000, cache_file=cache_file)
            loader = DataLoader(train_ds, batch_size=16, shuffle=True)
            retrained_m = DeepBeamformingNet(M=M_val, N=8, K=4, hidden_dim=64, n_blocks=2).to(device)
            opt = torch.optim.Adam(retrained_m.parameters(), lr=1e-3)
            retrained_m.train()
            for _ in range(25):
                for c_hat, n_dbm, G_t, H_t in loader:
                    c_hat, n_dbm, G_t, H_t = c_hat.to(device), n_dbm.to(device), G_t.to(device), H_t.to(device)
                    opt.zero_grad()
                    th, W = retrained_m(c_hat, n_dbm, P_max=cfg_M.pilot_power)
                    loss = -compute_true_sum_rate_torch(H_t, G_t, th, W, sigma_watts).mean()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(retrained_m.parameters(), 1.0)
                    opt.step()
            rt_mean, rt_std, _ = evaluate_configuration(cfg_M, retrained_m, n_samples=80, base_seed=410000)

        results_M.append({
            "M": M_val,
            "zero_shot_sum_rate": zs_mean,
            "retrained_sum_rate": rt_mean,
            "random_rzf_sum_rate": rand_mean,
            "per_user_rate": rt_mean / 4.0,
            "per_antenna_rate": rt_mean / 8.0,
        })
        print(f"M={M_val:3d} | Random RZF: {rand_mean:5.2f} | Zero-Shot: {zs_mean:5.2f} | Retrained: {rt_mean:5.2f} bps/Hz")

    # -------------------------------------------------------------
    # 2. BS Antennas N in {4, 8, 16} (Retrained Models)
    # -------------------------------------------------------------
    print("\n" + "=" * 80)
    print("EVALUATING BS ANTENNA SCALING: N in {4, 8, 16}")
    print("=" * 80)
    for N_val in [4, 8, 16]:
        cfg_N = SystemConfig(M=64, N=N_val, K=4)
        print(f"\n--- Dimension N = {N_val} (M=64, K=4) ---")
        if N_val == 8:
            rt_mean, rt_std, rand_mean = evaluate_configuration(cfg_N, frozen_model, n_samples=80, base_seed=430000)
        else:
            cache_file = os.path.join(OUT_DIR, f"train_data_N_{N_val}.pt")
            train_ds = BeamformingDataset(cfg_N, n_samples=200, base_seed=440000, cache_file=cache_file)
            loader = DataLoader(train_ds, batch_size=16, shuffle=True)
            model_N = DeepBeamformingNet(M=64, N=N_val, K=4, hidden_dim=64, n_blocks=2).to(device)
            opt = torch.optim.Adam(model_N.parameters(), lr=1e-3)
            model_N.train()
            for _ in range(25):
                for c_hat, n_dbm, G_t, H_t in loader:
                    c_hat, n_dbm, G_t, H_t = c_hat.to(device), n_dbm.to(device), G_t.to(device), H_t.to(device)
                    opt.zero_grad()
                    th, W = model_N(c_hat, n_dbm, P_max=cfg_N.pilot_power)
                    loss = -compute_true_sum_rate_torch(H_t, G_t, th, W, sigma_watts).mean()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model_N.parameters(), 1.0)
                    opt.step()
            rt_mean, rt_std, rand_mean = evaluate_configuration(cfg_N, model_N, n_samples=80, base_seed=430000)

        results_N.append({
            "N": N_val,
            "retrained_sum_rate": rt_mean,
            "random_rzf_sum_rate": rand_mean,
            "per_user_rate": rt_mean / 4.0,
            "per_antenna_rate": rt_mean / N_val,
        })
        print(f"N={N_val:2d} | Random RZF: {rand_mean:5.2f} | Deep ML: {rt_mean:5.2f} bps/Hz | Per-Antenna SE: {rt_mean/N_val:4.2f} bps/Hz/ant")

    # -------------------------------------------------------------
    # 3. User Count K in {2, 4, 8} (Retrained Models)
    # -------------------------------------------------------------
    print("\n" + "=" * 80)
    print("EVALUATING USER COUNT SCALING: K in {2, 4, 8}")
    print("=" * 80)
    for K_val in [2, 4, 8]:
        cfg_K = SystemConfig(M=64, N=8, K=K_val)
        print(f"\n--- Dimension K = {K_val} (M=64, N=8) ---")
        if K_val == 4:
            rt_mean, rt_std, rand_mean = evaluate_configuration(cfg_K, frozen_model, n_samples=80, base_seed=450000)
        else:
            cache_file = os.path.join(OUT_DIR, f"train_data_K_{K_val}.pt")
            train_ds = BeamformingDataset(cfg_K, n_samples=200, base_seed=460000, cache_file=cache_file)
            loader = DataLoader(train_ds, batch_size=16, shuffle=True)
            model_K = DeepBeamformingNet(M=64, N=8, K=K_val, hidden_dim=64, n_blocks=2).to(device)
            opt = torch.optim.Adam(model_K.parameters(), lr=1e-3)
            model_K.train()
            for _ in range(25):
                for c_hat, n_dbm, G_t, H_t in loader:
                    c_hat, n_dbm, G_t, H_t = c_hat.to(device), n_dbm.to(device), G_t.to(device), H_t.to(device)
                    opt.zero_grad()
                    th, W = model_K(c_hat, n_dbm, P_max=cfg_K.pilot_power)
                    loss = -compute_true_sum_rate_torch(H_t, G_t, th, W, sigma_watts).mean()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model_K.parameters(), 1.0)
                    opt.step()
            rt_mean, rt_std, rand_mean = evaluate_configuration(cfg_K, model_K, n_samples=80, base_seed=450000)

        results_K.append({
            "K": K_val,
            "retrained_sum_rate": rt_mean,
            "random_rzf_sum_rate": rand_mean,
            "per_user_rate": rt_mean / K_val,
            "per_antenna_rate": rt_mean / 8.0,
        })
        print(f"K={K_val:2d} | Random RZF: {rand_mean:5.2f} | Deep ML: {rt_mean:5.2f} bps/Hz | Per-User SE: {rt_mean/K_val:4.2f} bps/Hz/user")

    # Export combined JSON and CSV
    all_results = {
        "M_scaling": results_M,
        "N_scaling": results_N,
        "K_scaling": results_K,
    }
    with open(os.path.join(OUT_DIR, "dimension_generalization_summary.json"), "w") as f:
        json.dump(all_results, f, indent=2)

    # Plot Figures
    generate_generalization_figures(results_M, results_N, results_K)


def generate_generalization_figures(results_M: list[dict], results_N: list[dict], results_K: list[dict]):
    # Figure 1: M Scaling (Zero-shot vs Retrained)
    plt.figure(figsize=(8, 5), dpi=300)
    m_vals = [r["M"] for r in results_M]
    zs = [r["zero_shot_sum_rate"] for r in results_M]
    rt = [r["retrained_sum_rate"] for r in results_M]
    rb = [r["random_rzf_sum_rate"] for r in results_M]

    plt.plot(m_vals, rt, "#2ca02c", marker="o", label="Retrained Deep Beamforming", linewidth=2.2, markersize=8)
    plt.plot(m_vals, zs, "#1f77b4", marker="s", linestyle="--", label="Zero-Shot Deep Beamforming (Frozen M=64 Model)", linewidth=2.0, markersize=7)
    plt.plot(m_vals, rb, "#7f7f7f", marker="x", linestyle=":", label="Random IRS + RZF Baseline", linewidth=1.8, markersize=7)

    plt.xlabel("Number of IRS Reflecting Elements (M)", fontsize=11, fontweight="bold")
    plt.ylabel("Achievable Sum-Rate (bps/Hz)", fontsize=11, fontweight="bold")
    plt.title("Task 3: Surface Dimension Scaling — Zero-Shot vs Retrained Performance", fontsize=12, fontweight="bold")
    plt.xticks(m_vals, [f"M={m}" for m in m_vals], fontweight="bold")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(fontsize=9, loc="center left")
    plt.tight_layout()
    fig1 = os.path.join(OUT_DIR, "fig_generalization_M_elements.png")
    plt.savefig(fig1)
    plt.close()
    print(f"Saved {fig1}")

    # Figure 2: N (BS antennas) and K (users) scaling
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=300)

    # Left: N Antennas
    n_vals = [r["N"] for r in results_N]
    n_rates = [r["retrained_sum_rate"] for r in results_N]
    n_rand = [r["random_rzf_sum_rate"] for r in results_N]
    axes[0].bar(np.arange(len(n_vals)) - 0.15, n_rand, 0.3, label="Random RZF", color="#7f7f7f", edgecolor="black")
    axes[0].bar(np.arange(len(n_vals)) + 0.15, n_rates, 0.3, label="Deep Beamforming", color="#1f77b4", edgecolor="black")
    axes[0].set_xticks(np.arange(len(n_vals)))
    axes[0].set_xticklabels([f"N={n}" for n in n_vals], fontweight="bold")
    axes[0].set_ylabel("Achievable Sum-Rate (bps/Hz)", fontweight="bold")
    axes[0].set_title("Base Station Antenna Scaling (M=64, K=4)", fontweight="bold")
    axes[0].grid(True, axis="y", linestyle="--", alpha=0.6)
    axes[0].legend()

    # Right: K Users
    k_vals = [r["K"] for r in results_K]
    k_rates = [r["retrained_sum_rate"] for r in results_K]
    k_rand = [r["random_rzf_sum_rate"] for r in results_K]
    axes[1].bar(np.arange(len(k_vals)) - 0.15, k_rand, 0.3, label="Random RZF", color="#7f7f7f", edgecolor="black")
    axes[1].bar(np.arange(len(k_vals)) + 0.15, k_rates, 0.3, label="Deep Beamforming", color="#2ca02c", edgecolor="black")
    axes[1].set_xticks(np.arange(len(k_vals)))
    axes[1].set_xticklabels([f"K={k}" for k in k_vals], fontweight="bold")
    axes[1].set_ylabel("Achievable Sum-Rate (bps/Hz)", fontweight="bold")
    axes[1].set_title("User Load Scaling (M=64, N=8)", fontweight="bold")
    axes[1].grid(True, axis="y", linestyle="--", alpha=0.6)
    axes[1].legend()

    plt.tight_layout()
    fig2 = os.path.join(OUT_DIR, "fig_generalization_NK_dimensions.png")
    plt.savefig(fig2)
    plt.close()
    print(f"Saved {fig2}")


if __name__ == "__main__":
    run_dimension_generalization()
