"""
Task 4 -- Controlled Noise-Floor Input Feature Ablation Study.

Compares:
  Variant A (Noise-Aware): Receives LS cascaded channel estimate C_hat_LS + scalar noise floor sigma_dBm.
  Variant B (Ablated): Receives identical C_hat_LS without noise floor feature (use_noise_floor=False).

Both models share identical architecture depth, training samples, validation splits, optimizer, and test sets.
Evaluated across all 5 noise floors: [-100, -90, -85, -80, -70] dBm.
"""
from __future__ import annotations

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

OUT_DIR = "results/task4_noise_ablation"
NOISE_LEVELS = [-100.0, -90.0, -85.0, -80.0, -70.0]


def train_ablated_model(epochs: int = 35, lr: float = 1e-3, batch_size: int = 16, seed: int = 42) -> str:
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cpu")

    cfg = SystemConfig()
    sigma_watts = 3.16e-12
    P_max = cfg.pilot_power

    os.makedirs(OUT_DIR, exist_ok=True)

    train_cache = "results/gnn/train_data_300.pt"
    val_cache = "results/gnn/val_data_60.pt"

    train_ds = BeamformingDataset(cfg, n_samples=300, base_seed=1000, cache_file=train_cache)
    val_ds = BeamformingDataset(cfg, n_samples=60, base_seed=50000, cache_file=val_cache)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = DeepBeamformingNet(M=cfg.M, N=cfg.N, K=cfg.K, hidden_dim=64, n_blocks=2,
                               use_noise_floor=False).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=5)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nTraining Ablated DeepBeamformingNet without noise-floor input ({n_params:,} parameters) ...")

    best_val_rate = -float("inf")
    best_model_path = os.path.join(OUT_DIR, "model_ablated_no_noise.pt")
    history = {
        "variant": "ablated_no_noise_floor",
        "n_params": n_params,
        "train_mean_rate": [],
        "val_mean_rate": [],
        "val_median_rate": [],
    }

    t0 = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        train_rates = []
        for c_hat_ls, noise_dbm, G_true, H_true in train_loader:
            c_hat_ls = c_hat_ls.to(device)
            G_true = G_true.to(device)
            H_true = H_true.to(device)

            optimizer.zero_grad()
            # Forward pass without passing noise_dbm
            theta_ml, W_ml = model(c_hat_ls, noise_dbm=None, P_max=P_max)
            rates = compute_true_sum_rate_torch(H_true, G_true, theta_ml, W_ml, sigma_watts)
            loss = -rates.mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_rates.append(rates.detach().cpu())

        train_mean = float(torch.cat(train_rates).mean().item())

        # Evaluate on val
        model.eval()
        val_rates = []
        with torch.no_grad():
            for c_hat_ls, noise_dbm, G_true, H_true in val_loader:
                c_hat_ls = c_hat_ls.to(device)
                G_true = G_true.to(device)
                H_true = H_true.to(device)
                theta_ml, W_ml = model(c_hat_ls, noise_dbm=None, P_max=P_max)
                rates = compute_true_sum_rate_torch(H_true, G_true, theta_ml, W_ml, sigma_watts)
                val_rates.append(rates.cpu())
        val_rates = torch.cat(val_rates)
        val_mean = float(val_rates.mean().item())
        val_median = float(val_rates.median().item())
        scheduler.step(val_mean)

        history["train_mean_rate"].append(train_mean)
        history["val_mean_rate"].append(val_mean)
        history["val_median_rate"].append(val_median)

        if val_mean > best_val_rate:
            best_val_rate = val_mean
            torch.save(model.state_dict(), best_model_path)
            mark = "(* BEST *)"
        else:
            mark = ""

        if epoch % 5 == 0 or epoch == epochs or mark:
            print(f"Epoch {epoch:2d}/{epochs:2d} | Train Rate: {train_mean:6.3f} | Val Rate: {val_mean:6.3f} (med: {val_median:6.3f}) {mark}")

    elapsed = time.time() - t0
    history["total_time_sec"] = elapsed
    history["best_val_mean_rate"] = best_val_rate

    history_path = os.path.join(OUT_DIR, "history_ablated_no_noise.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    print(f"Ablated training completed in {elapsed:.1f}s. Best Val Rate: {best_val_rate:.4f} bps/Hz -> Saved to {best_model_path}\n")
    return best_model_path


def evaluate_noise_ablation():
    cfg = SystemConfig()
    sigma_watts = 3.16e-12
    P_max = cfg.pilot_power
    device = torch.device("cpu")

    # Load Noise-Aware (Frozen Baseline)
    aware_model = DeepBeamformingNet(M=cfg.M, N=cfg.N, K=cfg.K, hidden_dim=64, n_blocks=2, use_noise_floor=True).to(device)
    aware_model.load_state_dict(torch.load("results/gnn/best_beamforming_model.pt", map_location=device))
    aware_model.eval()

    # Load Ablated Model
    ablated_model = DeepBeamformingNet(M=cfg.M, N=cfg.N, K=cfg.K, hidden_dim=64, n_blocks=2, use_noise_floor=False).to(device)
    ablated_path = os.path.join(OUT_DIR, "model_ablated_no_noise.pt")
    ablated_model.load_state_dict(torch.load(ablated_path, map_location=device))
    ablated_model.eval()

    # Load Phase 5 held-out test data (500 realizations, seeds 200,000 to 200,499)
    test_data = torch.load("results/phase5_quantization/phase5_test_data.pt", weights_only=False)
    c_hat = test_data["c_hat"]
    noise_dbm = test_data["noise_dbm"]
    G_true = test_data["G_true"]
    H_true = test_data["H_true"]
    seeds = test_data["seeds"]
    n_samples = len(seeds)

    rates_aware = np.zeros(n_samples)
    rates_ablated = np.zeros(n_samples)
    times_aware = []
    times_ablated = []

    print(f"\nEvaluating Noise-Aware vs Ablated models on {n_samples} held-out realizations ...")
    with torch.no_grad():
        for i in range(n_samples):
            ch_i = c_hat[i:i+1].to(device)
            n_i = noise_dbm[i:i+1].to(device)
            G_i = G_true[i:i+1].to(device)
            H_i = H_true[i:i+1].to(device)

            # Noise-Aware
            t0 = time.perf_counter()
            th_aware, W_aware = aware_model(ch_i, n_i, P_max=P_max)
            times_aware.append(time.perf_counter() - t0)
            rates_aware[i] = compute_true_sum_rate_torch(H_i, G_i, th_aware, W_aware, sigma_watts).item()

            # Ablated (no noise input)
            t0 = time.perf_counter()
            th_ablated, W_ablated = ablated_model(ch_i, noise_dbm=None, P_max=P_max)
            times_ablated.append(time.perf_counter() - t0)
            rates_ablated[i] = compute_true_sum_rate_torch(H_i, G_i, th_ablated, W_ablated, sigma_watts).item()

    # Export Per-Realization CSV
    noise_arr = noise_dbm.numpy()
    per_real_csv = os.path.join(OUT_DIR, "noise_ablation_per_realization.csv")
    with open(per_real_csv, "w", encoding="utf-8") as f:
        f.write("seed,noise_dbm,rate_noise_aware,rate_ablated,diff_aware_minus_ablated\n")
        for i in range(n_samples):
            diff = rates_aware[i] - rates_ablated[i]
            f.write(f"{seeds[i]},{noise_arr[i]:.1f},{rates_aware[i]:.4f},{rates_ablated[i]:.4f},{diff:+.4f}\n")
    print(f"Saved {per_real_csv}")

    # Summary table per noise floor
    summary_rows = []
    print("\n" + "=" * 105)
    print("TASK 4 NOISE-FLOOR INPUT ABLATION SUMMARY (500 REALIZATIONS)")
    print("=" * 105)
    print(f"{'Noise':>7} | {'N':>4} | {'Noise-Aware (bps/Hz)':>22} | {'Ablated (bps/Hz)':>20} | {'Gain':>8} | {'Win Rate':>9} | {'p (t-test)':>11}")
    print("-" * 105)

    for nl in NOISE_LEVELS:
        mask = (noise_arr == nl)
        sub_aware = rates_aware[mask]
        sub_ablated = rates_ablated[mask]
        diffs = sub_aware - sub_ablated
        n_pts = len(diffs)
        wins = int(np.sum(diffs > 0))
        win_rate = (wins / n_pts) * 100.0
        t_res = stats.ttest_rel(sub_aware, sub_ablated)
        w_res = stats.wilcoxon(sub_aware, sub_ablated)
        ci = stats.t.interval(0.95, df=n_pts-1, loc=np.mean(diffs), scale=stats.sem(diffs))

        row = {
            "noise_dbm": nl,
            "n": n_pts,
            "aware_mean": float(np.mean(sub_aware)),
            "aware_median": float(np.median(sub_aware)),
            "aware_std": float(np.std(sub_aware)),
            "ablated_mean": float(np.mean(sub_ablated)),
            "ablated_median": float(np.median(sub_ablated)),
            "ablated_std": float(np.std(sub_ablated)),
            "mean_gain": float(np.mean(diffs)),
            "win_rate": win_rate,
            "wins": wins,
            "ci_low": float(ci[0]),
            "ci_high": float(ci[1]),
            "p_ttest": float(t_res.pvalue),
            "p_wilcoxon": float(w_res.pvalue),
        }
        summary_rows.append(row)
        print(f"{nl:6.1f}  | {n_pts:4d} | {row['aware_mean']:10.2f} ± {row['aware_std']:4.2f}     | {row['ablated_mean']:10.2f} ± {row['ablated_std']:4.2f}   | {row['mean_gain']:+8.2f} | {win_rate:7.1f}% | {row['p_ttest']:11.2e}")

    print("=" * 105)

    # Overall Metrics
    all_diffs = rates_aware - rates_ablated
    overall_aware = float(np.mean(rates_aware))
    overall_ablated = float(np.mean(rates_ablated))
    overall_gain = float(np.mean(all_diffs))
    overall_wins = int(np.sum(all_diffs > 0))
    overall_win_rate = (overall_wins / len(all_diffs)) * 100.0
    overall_ci = stats.t.interval(0.95, df=len(all_diffs)-1, loc=overall_gain, scale=stats.sem(all_diffs))
    t_all = stats.ttest_rel(rates_aware, rates_ablated)
    w_all = stats.wilcoxon(rates_aware, rates_ablated)

    print(f"\nOVERALL METRICS ACROSS ALL 500 REALIZATIONS:")
    print(f"  Noise-Aware Model Mean Sum-Rate: {overall_aware:.2f} bps/Hz")
    print(f"  Ablated Model Mean Sum-Rate:     {overall_ablated:.2f} bps/Hz")
    print(f"  Overall Mean Difference:         {overall_gain:+.2f} bps/Hz ({(overall_gain)/overall_ablated*100:+.2f}%)")
    print(f"  Overall Win Rate:                {overall_win_rate:.1f}% ({overall_wins}/500)")
    print(f"  Overall 95% Confidence Interval: [{overall_ci[0]:+.2f}, {overall_ci[1]:+.2f}] bps/Hz")
    print(f"  Paired t-test p-value:           {t_all.pvalue:.4e}")
    print(f"  Wilcoxon test p-value:           {w_all.pvalue:.4e}")

    # Latency comparison
    print(f"\nLATENCY BENCHMARK:")
    print(f"  Noise-Aware Forward Pass: {np.mean(times_aware)*1000:.2f} ms")
    print(f"  Ablated Forward Pass:     {np.mean(times_ablated)*1000:.2f} ms")

    # Export Summary CSV
    summary_csv = os.path.join(OUT_DIR, "noise_ablation_summary.csv")
    with open(summary_csv, "w", encoding="utf-8") as f:
        keys = list(summary_rows[0].keys())
        f.write(",".join(keys) + "\n")
        for r in summary_rows:
            f.write(",".join(str(r[k]) for k in keys) + "\n")
    print(f"Saved {summary_csv}")

    # Generate Figures
    generate_ablation_figures(summary_rows, rates_aware, rates_ablated, noise_arr)


def generate_ablation_figures(summary_rows: list[dict], rates_aware: np.ndarray, rates_ablated: np.ndarray, noise_arr: np.ndarray):
    noise_levels = [r["noise_dbm"] for r in summary_rows]
    aware_m = [r["aware_mean"] for r in summary_rows]
    aware_s = [r["aware_std"] for r in summary_rows]
    ablated_m = [r["ablated_mean"] for r in summary_rows]
    ablated_s = [r["ablated_std"] for r in summary_rows]
    gains = [r["mean_gain"] for r in summary_rows]
    ci_l = [r["ci_low"] for r in summary_rows]
    ci_h = [r["ci_high"] for r in summary_rows]

    # Figure 1: Noise-Aware vs Ablated Sum-Rate
    plt.figure(figsize=(8, 5.5), dpi=300)
    plt.plot(noise_levels, aware_m, "#1f77b4", marker="o", label="Noise-Aware (Inputs: C_hat_LS + sigma_dBm)", linewidth=2.4, markersize=7)
    plt.plot(noise_levels, ablated_m, "#ff7f0e", marker="s", linestyle="--", label="Ablated Model (Input: C_hat_LS only)", linewidth=2.2, markersize=7)
    plt.fill_between(noise_levels, np.array(aware_m) - np.array(aware_s), np.array(aware_m) + np.array(aware_s), color="#1f77b4", alpha=0.15)
    plt.fill_between(noise_levels, np.array(ablated_m) - np.array(ablated_s), np.array(ablated_m) + np.array(ablated_s), color="#ff7f0e", alpha=0.15)

    plt.xlabel("Pilot Noise Floor (dBm)", fontsize=12, fontweight="bold")
    plt.ylabel("Achievable Sum-Rate (bps/Hz)", fontsize=12, fontweight="bold")
    plt.title("Task 4: Noise-Floor Input Ablation Study (N=500)", fontsize=13, fontweight="bold")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(fontsize=10, loc="center left")
    plt.tight_layout()
    fig1 = os.path.join(OUT_DIR, "fig_noise_ablation_comparison.png")
    plt.savefig(fig1)
    plt.close()
    print(f"Saved {fig1}")

    # Figure 2: Paired Gain of Noise Awareness with 95% CI error bars
    plt.figure(figsize=(8, 5), dpi=300)
    yerr_low = np.array(gains) - np.array(ci_l)
    yerr_high = np.array(ci_h) - np.array(gains)
    plt.errorbar(noise_levels, gains, yerr=[yerr_low, yerr_high], fmt="o-", color="#2ca02c",
                 ecolor="#2ca02c", elinewidth=2, capsize=5, capthick=1.5, linewidth=2.2, markersize=8,
                 label="Paired Gain of Noise-Floor Input (Aware - Ablated) ± 95% CI")
    plt.axhline(0, color="gray", linestyle="--", linewidth=1.2)
    plt.xlabel("Pilot Noise Floor (dBm)", fontsize=12, fontweight="bold")
    plt.ylabel("Paired Gain (bps/Hz)", fontsize=12, fontweight="bold")
    plt.title("Net Benefit of Explicit Noise-Floor Awareness Across Noise Regimes", fontsize=12, fontweight="bold")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(fontsize=10, loc="lower right")
    plt.tight_layout()
    fig2 = os.path.join(OUT_DIR, "fig_noise_ablation_gain.png")
    plt.savefig(fig2)
    plt.close()
    print(f"Saved {fig2}")


if __name__ == "__main__":
    train_ablated_model(epochs=35)
    evaluate_noise_ablation()
