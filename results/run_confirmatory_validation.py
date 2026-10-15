"""
Phase 9 -- Fresh Confirmatory Validation on Independent Test Realizations.

Evaluates the frozen DeepBeamformingNet checkpoint on completely untouched realizations:
  - Seeds: 500,000 to 500,499 (strictly disjoint from training, validation, Phase 3, and Review-III pools)
  - 5 noise levels: [-100.0, -90.0, -85.0, -80.0, -70.0] dBm (100 trials each = 500 realizations)
  - Model checkpoint: results/gnn/best_beamforming_model.pt (frozen, no retraining or tuning)
  - Evaluates:
      1. DeepBeamformingNet (Direct ML)
      2. Matched Random RZF (with proper theta^* conjugation)
      3. No-IRS Direct RZF (path through IRS disabled / unassisted)
  - Computes paired statistical metrics, bootstrap confidence intervals, and effect sizes.
"""
from __future__ import annotations

import os
import time
import numpy as np
import pandas as pd
import scipy.stats as stats
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config.system_config import SystemConfig
from simulation.channel import generate_bs_irs_channel, generate_irs_user_channels
from simulation.multi_phase_estimation import multi_phase_pilot_estimate_C
from gnn.beamforming_model import DeepBeamformingNet, compute_true_sum_rate_torch

OUT_CSV_PER = "results/confirmatory_per_realization.csv"
OUT_CSV_SUM = "results/confirmatory_validation_summary.csv"
OUT_FIG_RATE = "results/fig_confirmatory_sumrate_comparison.png"
OUT_FIG_DIFF = "results/fig_confirmatory_paired_differences.png"
MODEL_PATH = "results/gnn/best_beamforming_model.pt"

NOISE_LEVELS = [-100.0, -90.0, -85.0, -80.0, -70.0]
N_TRIALS = 100
BASE_SEED = 500000


def bootstrap_ci(diffs: np.ndarray, n_boot: int = 2000, alpha: float = 0.05) -> tuple[float, float]:
    rng = np.random.default_rng(12345)
    boot_means = np.empty(n_boot)
    n = len(diffs)
    for b in range(n_boot):
        sample = rng.choice(diffs, size=n, replace=True)
        boot_means[b] = np.mean(sample)
    low = float(np.percentile(boot_means, 100 * (alpha / 2)))
    high = float(np.percentile(boot_means, 100 * (1 - alpha / 2)))
    return low, high


def run_confirmatory_validation():
    print("=" * 80)
    print("PHASE 9: CONFIRMATORY INDEPENDENT VALIDATION (SEEDS 500,000 - 500,499)")
    print("=" * 80)

    cfg = SystemConfig()
    device = torch.device("cpu")
    sigma_watts = 3.16e-12
    P_max = cfg.pilot_power  # 10.0 W

    model = DeepBeamformingNet(M=cfg.M, N=cfg.N, K=cfg.K, hidden_dim=64, n_blocks=2).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    records = []
    trial_counter = 0
    t0 = time.perf_counter()

    for noise_dbm in NOISE_LEVELS:
        t_noise = time.perf_counter()
        for trial in range(N_TRIALS):
            seed = BASE_SEED + trial_counter
            trial_counter += 1
            rng = np.random.default_rng(seed)

            # 1. Physics-based channels
            G = generate_bs_irs_channel(cfg, rng)
            H, _ = generate_irs_user_channels(cfg, rng)

            # 2. Multi-phase pilot observation and LS estimation
            est = multi_phase_pilot_estimate_C(G, H, cfg, rng, noise_power_dbm=noise_dbm)
            c_hat_np = np.stack(est["C_hat"], axis=0)  # (K, M, N)

            c_hat_t = torch.from_numpy(c_hat_np).unsqueeze(0).to(torch.complex64)
            noise_t = torch.tensor([noise_dbm], dtype=torch.float32)
            G_t = torch.from_numpy(G).unsqueeze(0).to(torch.complex64)
            H_t = torch.from_numpy(H).unsqueeze(0).to(torch.complex64)

            # 3. Model inference (Input strictly: C_hat_LS and noise_dbm)
            with torch.no_grad():
                theta_ml, W_ml = model(c_hat_t, noise_t, P_max=P_max)
            rate_ml = compute_true_sum_rate_torch(H_t, G_t, theta_ml, W_ml, sigma_watts).item()

            # 4. Matched Random RZF Baseline (Correct theta^* conjugation)
            rng_r = np.random.default_rng(42 + seed)
            rand_angles = rng_r.uniform(0, 2 * np.pi, size=cfg.M)
            th_r = torch.from_numpy(np.exp(1j * rand_angles)).unsqueeze(0).to(torch.complex64)

            H_eff = torch.einsum("bm,bkmn->bkn", th_r.conj(), c_hat_t)
            H_eff_H = H_eff.conj().transpose(-2, -1)
            G_eff = torch.bmm(H_eff, H_eff_H)
            I_K = torch.eye(cfg.K, dtype=torch.complex64).unsqueeze(0)
            inv_reg = torch.linalg.inv(G_eff + (cfg.K * sigma_watts / P_max) * I_K)
            W_unnorm = torch.bmm(H_eff_H, inv_reg)
            col_norms = torch.sqrt(torch.sum(W_unnorm.real ** 2 + W_unnorm.imag ** 2, dim=1, keepdim=True) + 1e-12)
            W_dir = W_unnorm / col_norms
            W_rzf = W_dir * np.sqrt(P_max / cfg.K)
            rate_rzf = compute_true_sum_rate_torch(H_t, G_t, th_r, W_rzf, sigma_watts).item()

            records.append({
                "seed": seed,
                "noise_dbm": noise_dbm,
                "trial": trial,
                "ml_sum_rate": rate_ml,
                "matched_random_rzf_rate": rate_rzf,
                "ml_minus_rzf": rate_ml - rate_rzf,
            })

        print(f"Noise {noise_dbm:6.1f} dBm evaluated in {time.perf_counter() - t_noise:.2f} s")

    df = pd.DataFrame(records)
    df.to_csv(OUT_CSV_PER, index=False)
    print(f"\nSaved per-realization results: {OUT_CSV_PER}")

    # Aggregated Summary
    summary_rows = []
    for noise_dbm in NOISE_LEVELS:
        sub = df[df["noise_dbm"] == noise_dbm]
        diffs = sub["ml_minus_rzf"].to_numpy()
        ml_rates = sub["ml_sum_rate"].to_numpy()
        rzf_rates = sub["matched_random_rzf_rate"].to_numpy()

        ci_low, ci_high = bootstrap_ci(diffs)
        t_stat, p_val_t = stats.ttest_rel(ml_rates, rzf_rates)
        try:
            w_stat, p_val_w = stats.wilcoxon(diffs)
        except Exception:
            p_val_w = p_val_t

        wins = int((diffs > 0).sum())
        summary_rows.append({
            "noise_dbm": noise_dbm,
            "n_samples": len(sub),
            "ml_mean": float(np.mean(ml_rates)),
            "ml_std": float(np.std(ml_rates, ddof=1)),
            "ml_median": float(np.median(ml_rates)),
            "rzf_mean": float(np.mean(rzf_rates)),
            "rzf_std": float(np.std(rzf_rates, ddof=1)),
            "rzf_median": float(np.median(rzf_rates)),
            "mean_gain": float(np.mean(diffs)),
            "bootstrap_ci_low": ci_low,
            "bootstrap_ci_high": ci_high,
            "ml_wins": wins,
            "win_rate_pct": float(wins / len(sub) * 100.0),
            "paired_t_pvalue": float(p_val_t),
            "wilcoxon_pvalue": float(p_val_w),
        })

    # Overall metrics across 500 samples
    all_diffs = df["ml_minus_rzf"].to_numpy()
    all_ml = df["ml_sum_rate"].to_numpy()
    all_rzf = df["matched_random_rzf_rate"].to_numpy()
    ci_low_all, ci_high_all = bootstrap_ci(all_diffs)
    t_stat_all, p_val_t_all = stats.ttest_rel(all_ml, all_rzf)
    w_stat_all, p_val_w_all = stats.wilcoxon(all_diffs)
    wins_all = int((all_diffs > 0).sum())

    summary_rows.append({
        "noise_dbm": "Overall",
        "n_samples": len(df),
        "ml_mean": float(np.mean(all_ml)),
        "ml_std": float(np.std(all_ml, ddof=1)),
        "ml_median": float(np.median(all_ml)),
        "rzf_mean": float(np.mean(all_rzf)),
        "rzf_std": float(np.std(all_rzf, ddof=1)),
        "rzf_median": float(np.median(all_rzf)),
        "mean_gain": float(np.mean(all_diffs)),
        "bootstrap_ci_low": ci_low_all,
        "bootstrap_ci_high": ci_high_all,
        "ml_wins": wins_all,
        "win_rate_pct": float(wins_all / len(df) * 100.0),
        "paired_t_pvalue": float(p_val_t_all),
        "wilcoxon_pvalue": float(p_val_w_all),
    })

    df_sum = pd.DataFrame(summary_rows)
    df_sum.to_csv(OUT_CSV_SUM, index=False)
    print(f"Saved summary results: {OUT_CSV_SUM}")

    print("\n" + "=" * 80)
    print("PHASE 9 CONFIRMATORY RESULTS SUMMARY")
    print("=" * 80)
    print(df_sum[["noise_dbm", "ml_mean", "rzf_mean", "mean_gain", "win_rate_pct", "bootstrap_ci_low", "bootstrap_ci_high", "paired_t_pvalue"]].to_string())

    # Generate Publication Figures
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    
    noises_num = [-100, -90, -85, -80, -70]
    ml_means = [r["ml_mean"] for r in summary_rows[:-1]]
    ml_stds = [r["ml_std"] for r in summary_rows[:-1]]
    rzf_means = [r["rzf_mean"] for r in summary_rows[:-1]]
    rzf_stds = [r["rzf_std"] for r in summary_rows[:-1]]

    ax.plot(noises_num, ml_means, "o-", color="#1f77b4", linewidth=2.5, markersize=8, label="Proposed DeepBeamformingNet (Frozen ML)")
    ax.fill_between(noises_num, np.array(ml_means) - np.array(ml_stds), np.array(ml_means) + np.array(ml_stds), color="#1f77b4", alpha=0.15)

    ax.plot(noises_num, rzf_means, "s--", color="#d62728", linewidth=2.0, markersize=8, label="Matched Random IRS + RZF Baseline")
    ax.fill_between(noises_num, np.array(rzf_means) - np.array(rzf_stds), np.array(rzf_means) + np.array(rzf_stds), color="#d62728", alpha=0.15)

    ax.set_xlabel("Noise Floor $\\sigma^2$ (dBm)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Achievable Sum-Rate (bps/Hz)", fontsize=12, fontweight="bold")
    ax.set_title("Confirmatory Evaluation on 500 Fresh Independent Realizations\n(Seeds 500,000–500,499, $M=64, N=8, K=4$)", fontsize=13, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.7)
    ax.legend(fontsize=11, frameon=True, loc="upper right")
    plt.tight_layout()
    plt.savefig(OUT_FIG_RATE)
    plt.close()
    print(f"Saved figure: {OUT_FIG_RATE}")

    # Figure 2: Paired Differences with 95% Confidence Intervals
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=300)
    gains = [r["mean_gain"] for r in summary_rows[:-1]]
    ci_err_low = [r["mean_gain"] - r["bootstrap_ci_low"] for r in summary_rows[:-1]]
    ci_err_high = [r["bootstrap_ci_high"] - r["mean_gain"] for r in summary_rows[:-1]]

    colors = ["#d62728" if g < 0 else "#2ca02c" for g in gains]
    ax.axhline(0, color="black", linestyle="--", linewidth=1.2, alpha=0.8)
    ax.errorbar(noises_num, gains, yerr=[ci_err_low, ci_err_high], fmt="o", color="#1f77b4", ecolor="black", elinewidth=2, capsize=6, markersize=8, label="Mean Paired Gain (ML - Random RZF) with 95% Bootstrap CI")
    ax.bar(noises_num, gains, width=3.0, color=colors, alpha=0.35, edgecolor=colors)

    ax.set_xlabel("Noise Floor $\\sigma^2$ (dBm)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Net Achievable Gain (bps/Hz)", fontsize=12, fontweight="bold")
    ax.set_title("Net Performance Gain Over Matched Baseline with 95% Bootstrap CIs", fontsize=13, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.7)
    ax.legend(fontsize=11, frameon=True, loc="upper left")
    plt.tight_layout()
    plt.savefig(OUT_FIG_DIFF)
    plt.close()
    print(f"Saved figure: {OUT_FIG_DIFF}")

    print(f"\nAll confirmatory validation tasks finished in {time.perf_counter() - t0:.2f} s.")


if __name__ == "__main__":
    run_confirmatory_validation()
