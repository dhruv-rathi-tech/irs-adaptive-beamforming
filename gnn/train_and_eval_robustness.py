"""
Task 2 -- Channel Distribution Robustness Evaluation.

Evaluates DeepBeamformingNet under changed physical propagation conditions:
  1. Varied Rician K-factors for BS-IRS link:
     - K=0.0 (Pure Rayleigh / Complete Blockage of dominant path)
     - K=2.0 (Weak LOS / Rich Scattering)
     - K=10.0 (Default Benchmark)
     - K=20.0 (Strong LOS / Near-Field Alignment)
  2. Varied Path-Loss Exponents:
     - alpha_BI=2.0 (Free Space / Pure LOS)
     - alpha_BI=2.2 (Default Benchmark)
     - alpha_BI=2.8 (Obstructed Urban / Cluttered Propagation)

Evaluates:
  - Zero-shot performance (frozen model trained on default distribution)
  - Retrained model performance (trained on the new distribution)
  - Baseline Random IRS + RZF
"""
from __future__ import annotations

import copy
import json
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import torch
from torch.utils.data import DataLoader

from config.system_config import SystemConfig
from gnn.beamforming_dataset import BeamformingDataset
from gnn.beamforming_model import DeepBeamformingNet, compute_true_sum_rate_torch
from simulation.channel import generate_bs_irs_channel, generate_irs_user_channels
from simulation.multi_phase_estimation import multi_phase_pilot_estimate_C

OUT_DIR = "results/task2_robustness"


def run_robustness_study():
    os.makedirs(OUT_DIR, exist_ok=True)
    device = torch.device("cpu")
    sigma_watts = 3.16e-12

    # Load frozen reference model (trained on K=10, alpha=2.2)
    frozen_model = DeepBeamformingNet(M=64, N=8, K=4, hidden_dim=64, n_blocks=2).to(device)
    frozen_model.load_state_dict(torch.load("results/gnn/best_beamforming_model.pt", map_location=device))
    frozen_model.eval()

    channel_settings = [
        {"name": "Default (K=10, alpha=2.2)", "K_rice": 10.0, "alpha_BI": 2.2},
        {"name": "Weak LOS (K=2.0, alpha=2.2)", "K_rice": 2.0, "alpha_BI": 2.2},
        {"name": "Pure Rayleigh (K=0.0, alpha=2.2)", "K_rice": 0.0, "alpha_BI": 2.2},
        {"name": "Strong LOS (K=20.0, alpha=2.0)", "K_rice": 20.0, "alpha_BI": 2.0},
        {"name": "High Path Loss (K=10.0, alpha=2.8)", "K_rice": 10.0, "alpha_BI": 2.8},
    ]

    results_table = []
    base_test_seed = 300000
    n_test_samples = 100
    noise_dbm_test = -85.0  # standard operating point

    for s_idx, setting in enumerate(channel_settings):
        set_name = setting["name"]
        K_rice = setting["K_rice"]
        alpha_BI = setting["alpha_BI"]
        print(f"\n[{s_idx+1}/{len(channel_settings)}] Processing Setting: {set_name} ...")

        # Custom config for this setting
        cfg = SystemConfig(rician_K_BI=K_rice, alpha_BI=alpha_BI)

        # 1. Generate 100 test realizations for this channel setting
        test_c_hat = []
        test_noise = []
        test_G = []
        test_H = []

        for i in range(n_test_samples):
            rng = np.random.default_rng(base_test_seed + i)
            G = generate_bs_irs_channel(cfg, rng)
            H, _ = generate_irs_user_channels(cfg, rng)
            est = multi_phase_pilot_estimate_C(G, H, cfg, rng, noise_power_dbm=noise_dbm_test)
            c_hat_np = np.stack(est["C_hat"], axis=0)

            test_c_hat.append(torch.from_numpy(c_hat_np).to(torch.complex64))
            test_noise.append(torch.tensor([noise_dbm_test], dtype=torch.float32))
            test_G.append(torch.from_numpy(G).to(torch.complex64))
            test_H.append(torch.from_numpy(H).to(torch.complex64))

        test_c_hat = torch.stack(test_c_hat, dim=0).to(device)
        test_noise = torch.stack(test_noise, dim=0).to(device)
        test_G = torch.stack(test_G, dim=0).to(device)
        test_H = torch.stack(test_H, dim=0).to(device)

        # 2. Evaluate Zero-Shot Performance of Frozen Model
        with torch.no_grad():
            th_zs, W_zs = frozen_model(test_c_hat, test_noise, P_max=cfg.pilot_power)
            rates_zs = compute_true_sum_rate_torch(test_H, test_G, th_zs, W_zs, sigma_watts).cpu().numpy()

        # 3. Retrain on this specific channel setting (fast 25-epoch training on 200 samples)
        retrained_model_path = os.path.join(OUT_DIR, f"model_retrained_setting_{s_idx}.pt")
        retrain_cache = os.path.join(OUT_DIR, f"train_data_setting_{s_idx}.pt")

        train_ds = BeamformingDataset(cfg, n_samples=200, base_seed=310000, cache_file=retrain_cache)
        train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)

        retrained_model = DeepBeamformingNet(M=64, N=8, K=4, hidden_dim=64, n_blocks=2).to(device)
        optimizer = torch.optim.Adam(retrained_model.parameters(), lr=1e-3)

        retrained_model.train()
        for epoch in range(25):
            for c_hat_ls, noise_dbm, G_t, H_t in train_loader:
                c_hat_ls = c_hat_ls.to(device)
                noise_dbm = noise_dbm.to(device)
                G_t = G_t.to(device)
                H_t = H_t.to(device)

                optimizer.zero_grad()
                th, W = retrained_model(c_hat_ls, noise_dbm, P_max=cfg.pilot_power)
                loss = -compute_true_sum_rate_torch(H_t, G_t, th, W, sigma_watts).mean()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(retrained_model.parameters(), 1.0)
                optimizer.step()

        torch.save(retrained_model.state_dict(), retrained_model_path)
        retrained_model.eval()

        with torch.no_grad():
            th_rt, W_rt = retrained_model(test_c_hat, test_noise, P_max=cfg.pilot_power)
            rates_rt = compute_true_sum_rate_torch(test_H, test_G, th_rt, W_rt, sigma_watts).cpu().numpy()

        # 4. Baseline Random IRS + RZF
        rates_rand = []
        for i in range(n_test_samples):
            rng_r = np.random.default_rng(base_test_seed + i)
            th_r = torch.from_numpy(np.exp(1j * rng_r.uniform(0, 2*np.pi, (1, 64)))).to(torch.complex64).to(device)
            ch_i = test_c_hat[i:i+1]
            H_eff = torch.einsum("bm,bkmn->bkn", th_r.conj(), ch_i)
            H_eff_H = H_eff.conj().transpose(-2, -1)
            G_eff = torch.bmm(H_eff, H_eff_H)
            I_K = torch.eye(4, dtype=torch.complex64, device=device).unsqueeze(0)
            inv_reg = torch.linalg.inv(G_eff + (4 * sigma_watts / cfg.pilot_power) * I_K)
            W_unnorm = torch.bmm(H_eff_H, inv_reg)
            col_norms = torch.sqrt(torch.sum(W_unnorm.real**2 + W_unnorm.imag**2, dim=1, keepdim=True) + 1e-12)
            W_rzf = (W_unnorm / col_norms) * np.sqrt(cfg.pilot_power / 4)
            r = compute_true_sum_rate_torch(test_H[i:i+1], test_G[i:i+1], th_r, W_rzf, sigma_watts).item()
            rates_rand.append(r)
        rates_rand = np.array(rates_rand)

        zs_mean = float(np.mean(rates_zs))
        rt_mean = float(np.mean(rates_rt))
        rand_mean = float(np.mean(rates_rand))

        diff_rt_zs = rt_mean - zs_mean
        t_stat, p_val = stats.ttest_rel(rates_rt, rates_zs)

        entry = {
            "setting_name": set_name,
            "K_rice": K_rice,
            "alpha_BI": alpha_BI,
            "zero_shot_mean": zs_mean,
            "zero_shot_std": float(np.std(rates_zs)),
            "retrained_mean": rt_mean,
            "retrained_std": float(np.std(rates_rt)),
            "random_rzf_mean": rand_mean,
            "gain_retrained_over_zero_shot": diff_rt_zs,
            "zero_shot_win_rate_vs_random": float(np.sum(rates_zs > rates_rand)),
            "p_val_retrained_vs_zs": float(p_val),
        }
        results_table.append(entry)
        print(f"Setting: {set_name:<35} | Random RZF: {rand_mean:5.2f} | Zero-Shot: {zs_mean:5.2f} | Retrained: {rt_mean:5.2f} | Retrain Gain: {diff_rt_zs:+5.2f}")

    # Export CSV
    csv_path = os.path.join(OUT_DIR, "robustness_summary.csv")
    import csv
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        keys = list(results_table[0].keys())
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(results_table)
    print(f"\nSaved {csv_path}")

    # Generate Publication Figure
    generate_robustness_plot(results_table)


def generate_robustness_plot(results_table: list[dict]):
    names = [r["setting_name"].replace(" (", "\n(").replace("alpha", "α") for r in results_table]
    zs_means = [r["zero_shot_mean"] for r in results_table]
    rt_means = [r["retrained_mean"] for r in results_table]
    rand_means = [r["random_rzf_mean"] for r in results_table]

    x = np.arange(len(names))
    width = 0.26

    plt.figure(figsize=(10, 5.5), dpi=300)
    plt.bar(x - width, rand_means, width, label="Random IRS + RZF Baseline", color="#7f7f7f", edgecolor="black", linewidth=1)
    plt.bar(x, zs_means, width, label="Zero-Shot Deep Beamforming (Frozen Model)", color="#1f77b4", edgecolor="black", linewidth=1)
    plt.bar(x + width, rt_means, width, label="Retrained Deep Beamforming", color="#2ca02c", edgecolor="black", linewidth=1)

    plt.ylabel("True Achievable Sum-Rate (bps/Hz)", fontsize=11, fontweight="bold")
    plt.title("Task 2: Channel Robustness — Zero-Shot vs Retrained Transfer Across Fading Environments", fontsize=12, fontweight="bold")
    plt.xticks(x, names, fontsize=9, fontweight="bold")
    plt.legend(fontsize=9, loc="upper right")
    plt.grid(True, axis="y", linestyle="--", alpha=0.6)
    plt.tight_layout()

    fig_path = os.path.join(OUT_DIR, "fig_channel_robustness_comparison.png")
    plt.savefig(fig_path)
    plt.close()
    print(f"Saved {fig_path}")


if __name__ == "__main__":
    run_robustness_study()
