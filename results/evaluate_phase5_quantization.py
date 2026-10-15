"""
Phase 5 -- Evaluation of 1-bit and 2-bit Quantization-Aware Beamforming.

Evaluates 5 primary methods on identical 500 held-out channel realizations:
  1. Continuous Deep Beamforming (frozen reference checkpoint)
  2. 1-bit Post-Training Quantization (PTQ)
  3. 2-bit Post-Training Quantization (PTQ)
  4. 1-bit Quantization-Aware Trained (QAT)
  5. 2-bit Quantization-Aware Trained (QAT)
  Along with Random IRS + RZF baselines (Continuous, 1-bit, 2-bit).

Seeds: 200,000 to 200,499 (strictly disjoint from all training, val, and Phase 4 test sets).
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

from config.system_config import SystemConfig
from gnn.beamforming_model import DeepBeamformingNet, compute_true_sum_rate_torch
from gnn.quantization import quantize_phase_ste, quantize_phase_numpy
from simulation.channel import generate_bs_irs_channel, generate_irs_user_channels
from simulation.multi_phase_estimation import multi_phase_pilot_estimate_C

NOISE_LEVELS = [-100.0, -90.0, -85.0, -80.0, -70.0]
N_TRIALS_PER_NOISE = 100
BASE_SEED = 200000
OUT_DIR = "results/phase5_quantization"


def generate_or_load_test_data(cfg: SystemConfig) -> dict:
    os.makedirs(OUT_DIR, exist_ok=True)
    cache_file = os.path.join(OUT_DIR, "phase5_test_data.pt")
    if os.path.exists(cache_file):
        print(f"Loading cached Phase 5 test dataset from {cache_file} ...")
        return torch.load(cache_file, weights_only=False)

    print(f"Generating Phase 5 test dataset: 5 noise levels x {N_TRIALS_PER_NOISE} = 500 realizations (seeds {BASE_SEED} to {BASE_SEED + 499}) ...")
    c_hat_list = []
    noise_list = []
    G_list = []
    H_list = []
    seeds_list = []

    trial_counter = 0
    t0 = time.time()
    for noise_dbm in NOISE_LEVELS:
        for _ in range(N_TRIALS_PER_NOISE):
            seed = BASE_SEED + trial_counter
            trial_counter += 1
            rng = np.random.default_rng(seed)

            G = generate_bs_irs_channel(cfg, rng)
            H, _ = generate_irs_user_channels(cfg, rng)
            est = multi_phase_pilot_estimate_C(G, H, cfg, rng, noise_power_dbm=noise_dbm)
            c_hat_np = np.stack(est["C_hat"], axis=0)

            c_hat_list.append(torch.from_numpy(c_hat_np).to(torch.complex64))
            noise_list.append(torch.tensor(noise_dbm, dtype=torch.float32))
            G_list.append(torch.from_numpy(G).to(torch.complex64))
            H_list.append(torch.from_numpy(H).to(torch.complex64))
            seeds_list.append(seed)

    data = {
        "c_hat": torch.stack(c_hat_list, dim=0),
        "noise_dbm": torch.stack(noise_list, dim=0),
        "G_true": torch.stack(G_list, dim=0),
        "H_true": torch.stack(H_list, dim=0),
        "seeds": np.array(seeds_list),
    }
    torch.save(data, cache_file)
    print(f"Generated and cached {len(seeds_list)} test realizations in {time.time() - t0:.1f}s.")
    return data


def run_evaluation():
    cfg = SystemConfig()
    sigma_watts = 3.16e-12
    P_max = cfg.pilot_power
    device = torch.device("cpu")

    # 1. Load models
    cont_model = DeepBeamformingNet(M=cfg.M, N=cfg.N, K=cfg.K, hidden_dim=64, n_blocks=2).to(device)
    cont_model.load_state_dict(torch.load("results/gnn/best_beamforming_model.pt", map_location=device))
    cont_model.eval()

    qat1_model = DeepBeamformingNet(M=cfg.M, N=cfg.N, K=cfg.K, hidden_dim=64, n_blocks=2, quant_bits=1).to(device)
    qat1_model.load_state_dict(torch.load(os.path.join(OUT_DIR, "model_1bit_qat.pt"), map_location=device))
    qat1_model.eval()

    qat2_model = DeepBeamformingNet(M=cfg.M, N=cfg.N, K=cfg.K, hidden_dim=64, n_blocks=2, quant_bits=2).to(device)
    qat2_model.load_state_dict(torch.load(os.path.join(OUT_DIR, "model_2bit_qat.pt"), map_location=device))
    qat2_model.eval()

    # 2. Load dataset
    test_data = generate_or_load_test_data(cfg)
    c_hat = test_data["c_hat"]
    noise_dbm = test_data["noise_dbm"]
    G_true = test_data["G_true"]
    H_true = test_data["H_true"]
    seeds = test_data["seeds"]
    n_samples = len(seeds)

    rates = {
        "cont": np.zeros(n_samples),
        "ptq1": np.zeros(n_samples),
        "ptq2": np.zeros(n_samples),
        "qat1": np.zeros(n_samples),
        "qat2": np.zeros(n_samples),
        "rand_cont": np.zeros(n_samples),
        "rand_1bit": np.zeros(n_samples),
        "rand_2bit": np.zeros(n_samples),
    }

    times = {
        "cont": [],
        "ptq1": [],
        "ptq2": [],
        "qat1": [],
        "qat2": [],
    }

    print("\nRunning inference across 500 realizations for all quantization models ...")
    batch_size = 1
    with torch.no_grad():
        for i in range(n_samples):
            ch_i = c_hat[i:i+1].to(device)
            n_i = noise_dbm[i:i+1].to(device)
            G_i = G_true[i:i+1].to(device)
            H_i = H_true[i:i+1].to(device)

            # Continuous model
            t0 = time.perf_counter()
            th_cont, p_cont, a_cont = cont_model.forward_phase(ch_i, n_i)
            W_cont = cont_model.construct_beamformer(ch_i, th_cont, p_cont, a_cont, P_max=P_max)
            times["cont"].append(time.perf_counter() - t0)
            rates["cont"][i] = compute_true_sum_rate_torch(H_i, G_i, th_cont, W_cont, sigma_watts).item()

            # 1-bit PTQ (quantize continuous phase, re-derive optimal W_ML via RZF)
            t0 = time.perf_counter()
            th_ptq1 = quantize_phase_ste(th_cont, bits=1)
            W_ptq1 = cont_model.construct_beamformer(ch_i, th_ptq1, p_cont, a_cont, P_max=P_max)
            times["ptq1"].append(time.perf_counter() - t0)
            rates["ptq1"][i] = compute_true_sum_rate_torch(H_i, G_i, th_ptq1, W_ptq1, sigma_watts).item()

            # 2-bit PTQ
            t0 = time.perf_counter()
            th_ptq2 = quantize_phase_ste(th_cont, bits=2)
            W_ptq2 = cont_model.construct_beamformer(ch_i, th_ptq2, p_cont, a_cont, P_max=P_max)
            times["ptq2"].append(time.perf_counter() - t0)
            rates["ptq2"][i] = compute_true_sum_rate_torch(H_i, G_i, th_ptq2, W_ptq2, sigma_watts).item()

            # 1-bit QAT
            t0 = time.perf_counter()
            th_qat1, W_qat1 = qat1_model(ch_i, n_i, P_max=P_max)
            times["qat1"].append(time.perf_counter() - t0)
            rates["qat1"][i] = compute_true_sum_rate_torch(H_i, G_i, th_qat1, W_qat1, sigma_watts).item()

            # 2-bit QAT
            t0 = time.perf_counter()
            th_qat2, W_qat2 = qat2_model(ch_i, n_i, P_max=P_max)
            times["qat2"].append(time.perf_counter() - t0)
            rates["qat2"][i] = compute_true_sum_rate_torch(H_i, G_i, th_qat2, W_qat2, sigma_watts).item()

            # Random baselines (seeded per realization)
            rng_b = np.random.default_rng(seeds[i])
            rand_angles = rng_b.uniform(0, 2 * np.pi, size=(1, cfg.M))
            th_rand_cont = torch.from_numpy(np.exp(1j * rand_angles)).to(torch.complex64)
            th_rand_1bit = quantize_phase_ste(th_rand_cont, bits=1)
            th_rand_2bit = quantize_phase_ste(th_rand_cont, bits=2)

            for key, th_r in [("rand_cont", th_rand_cont), ("rand_1bit", th_rand_1bit), ("rand_2bit", th_rand_2bit)]:
                H_eff_r = torch.einsum("bm,bkmn->bkn", th_r.conj(), ch_i)
                H_eff_H = H_eff_r.conj().transpose(-2, -1)
                G_eff = torch.bmm(H_eff_r, H_eff_H)
                I_K = torch.eye(cfg.K, dtype=torch.complex64).unsqueeze(0)
                inv_reg = torch.linalg.inv(G_eff + (cfg.K * sigma_watts / P_max) * I_K)
                W_unnorm = torch.bmm(H_eff_H, inv_reg)
                col_norms = torch.sqrt(torch.sum(W_unnorm.real**2 + W_unnorm.imag**2, dim=1, keepdim=True) + 1e-12)
                W_rzf = (W_unnorm / col_norms) * np.sqrt(P_max / cfg.K)
                rates[key][i] = compute_true_sum_rate_torch(H_i, G_i, th_r, W_rzf, sigma_watts).item()

    # 3. Export Per-Realization CSV
    per_real_csv = os.path.join(OUT_DIR, "phase5_quantization_per_realization.csv")
    noise_arr = noise_dbm.numpy()
    with open(per_real_csv, "w", encoding="utf-8") as f:
        f.write("seed,noise_dbm,rate_cont,rate_ptq1,rate_qat1,rate_ptq2,rate_qat2,rate_rand_cont,rate_rand_1bit,rate_rand_2bit\n")
        for i in range(n_samples):
            f.write(f"{seeds[i]},{noise_arr[i]:.1f},{rates['cont'][i]:.4f},{rates['ptq1'][i]:.4f},{rates['qat1'][i]:.4f},"
                    f"{rates['ptq2'][i]:.4f},{rates['qat2'][i]:.4f},{rates['rand_cont'][i]:.4f},"
                    f"{rates['rand_1bit'][i]:.4f},{rates['rand_2bit'][i]:.4f}\n")
    print(f"Saved {per_real_csv}")

    # 4. Statistical Summary per Noise Floor
    summary_rows = []
    print("\n" + "=" * 115)
    print("PHASE 5 QUANTIZATION RESULTS SUMMARY (500 REALIZATIONS)")
    print("=" * 115)
    print(f"{'Noise':>7} | {'Continuous':>10} | {'1-bit PTQ':>10} | {'1-bit QAT':>10} | {'QAT1 Gain':>10} | {'2-bit PTQ':>10} | {'2-bit QAT':>10} | {'QAT2 Gain':>10} | {'Rand 1-bit':>10}")
    print("-" * 115)

    for nl in NOISE_LEVELS:
        mask = (noise_arr == nl)
        r_c = rates["cont"][mask]
        r_p1 = rates["ptq1"][mask]
        r_q1 = rates["qat1"][mask]
        r_p2 = rates["ptq2"][mask]
        r_q2 = rates["qat2"][mask]
        r_rb1 = rates["rand_1bit"][mask]

        diff_q1_p1 = r_q1 - r_p1
        diff_q2_p2 = r_q2 - r_p2

        t1 = stats.ttest_rel(r_q1, r_p1)
        t2 = stats.ttest_rel(r_q2, r_p2)

        row = {
            "noise_dbm": nl,
            "n": int(np.sum(mask)),
            "cont_mean": float(np.mean(r_c)),
            "cont_std": float(np.std(r_c)),
            "ptq1_mean": float(np.mean(r_p1)),
            "ptq1_std": float(np.std(r_p1)),
            "qat1_mean": float(np.mean(r_q1)),
            "qat1_std": float(np.std(r_q1)),
            "ptq2_mean": float(np.mean(r_p2)),
            "ptq2_std": float(np.std(r_p2)),
            "qat2_mean": float(np.mean(r_q2)),
            "qat2_std": float(np.std(r_q2)),
            "rand1_mean": float(np.mean(r_rb1)),
            "gain_qat1_over_ptq1": float(np.mean(diff_q1_p1)),
            "gain_qat2_over_ptq2": float(np.mean(diff_q2_p2)),
            "p_val_qat1_vs_ptq1": float(t1.pvalue),
            "p_val_qat2_vs_ptq2": float(t2.pvalue),
        }
        summary_rows.append(row)
        print(f"{nl:6.1f}  | {row['cont_mean']:10.2f} | {row['ptq1_mean']:10.2f} | {row['qat1_mean']:10.2f} | {row['gain_qat1_over_ptq1']:+10.2f} | {row['ptq2_mean']:10.2f} | {row['qat2_mean']:10.2f} | {row['gain_qat2_over_ptq2']:+10.2f} | {row['rand1_mean']:10.2f}")
    print("=" * 115)

    # Overall Metrics
    overall_c = float(np.mean(rates["cont"]))
    overall_p1 = float(np.mean(rates["ptq1"]))
    overall_q1 = float(np.mean(rates["qat1"]))
    overall_p2 = float(np.mean(rates["ptq2"]))
    overall_q2 = float(np.mean(rates["qat2"]))

    print(f"\nOVERALL ACROSS ALL 500 REALIZATIONS:")
    print(f"  Continuous Frozen Deep Beamforming: {overall_c:.2f} bps/Hz (100.0% reference)")
    print(f"  1-bit PTQ (Post-Training):          {overall_p1:.2f} bps/Hz (Loss: {overall_c - overall_p1:.2f} bps/Hz, -{(overall_c - overall_p1)/overall_c*100:.1f}%)")
    print(f"  1-bit QAT (Quantization-Aware):     {overall_q1:.2f} bps/Hz (Loss: {overall_c - overall_q1:.2f} bps/Hz, -{(overall_c - overall_q1)/overall_c*100:.1f}%)")
    print(f"  QAT 1-bit Gain over PTQ 1-bit:      +{overall_q1 - overall_p1:+.2f} bps/Hz (Win rate: {np.sum(rates['qat1'] > rates['ptq1'])}/500 = {np.sum(rates['qat1'] > rates['ptq1'])/5.0:.1f}%)")
    print(f"  2-bit PTQ (Post-Training):          {overall_p2:.2f} bps/Hz (Loss: {overall_c - overall_p2:.2f} bps/Hz, -{(overall_c - overall_p2)/overall_c*100:.1f}%)")
    print(f"  2-bit QAT (Quantization-Aware):     {overall_q2:.2f} bps/Hz (Loss: {overall_c - overall_q2:.2f} bps/Hz, -{(overall_c - overall_q2)/overall_c*100:.1f}%)")
    print(f"  QAT 2-bit Gain over PTQ 2-bit:      +{overall_q2 - overall_p2:+.2f} bps/Hz (Win rate: {np.sum(rates['qat2'] > rates['ptq2'])}/500 = {np.sum(rates['qat2'] > rates['ptq2'])/5.0:.1f}%)")

    # Latency
    print(f"\nLATENCY BENCHMARK (Inference Forward Pass per realization):")
    print(f"  Continuous ML: {np.mean(times['cont'])*1000:.2f} ms")
    print(f"  1-bit QAT:     {np.mean(times['qat1'])*1000:.2f} ms")
    print(f"  2-bit QAT:     {np.mean(times['qat2'])*1000:.2f} ms")

    # Save summary CSV
    summary_csv = os.path.join(OUT_DIR, "phase5_quantization_summary.csv")
    with open(summary_csv, "w", encoding="utf-8") as f:
        keys = list(summary_rows[0].keys())
        f.write(",".join(keys) + "\n")
        for r in summary_rows:
            f.write(",".join(str(r[k]) for k in keys) + "\n")
    print(f"Saved {summary_csv}")

    # Generate Publication Figures
    generate_figures(summary_rows, rates, noise_arr)


def generate_figures(summary_rows: list[dict], rates: dict, noise_arr: np.ndarray):
    noise_levels = [r["noise_dbm"] for r in summary_rows]
    c_m = [r["cont_mean"] for r in summary_rows]
    p1_m = [r["ptq1_mean"] for r in summary_rows]
    q1_m = [r["qat1_mean"] for r in summary_rows]
    p2_m = [r["ptq2_mean"] for r in summary_rows]
    q2_m = [r["qat2_mean"] for r in summary_rows]
    r1_m = [r["rand1_mean"] for r in summary_rows]

    # Figure 1: Sum-Rate vs Noise Floor for Continuous, PTQ, QAT
    plt.figure(figsize=(8.5, 5.5), dpi=300)
    plt.plot(noise_levels, c_m, "k-o", label="Continuous ML (Infinite Resolution)", linewidth=2.5, markersize=7)
    plt.plot(noise_levels, q2_m, "#2ca02c", marker="s", label="2-bit QAT (Proposed STE)", linewidth=2.2, markersize=7)
    plt.plot(noise_levels, p2_m, "#2ca02c", marker="^", linestyle="--", label="2-bit PTQ (Post-Training)", linewidth=1.8, markersize=6)
    plt.plot(noise_levels, q1_m, "#1f77b4", marker="D", label="1-bit QAT (Proposed STE)", linewidth=2.2, markersize=7)
    plt.plot(noise_levels, p1_m, "#1f77b4", marker="v", linestyle="--", label="1-bit PTQ (Post-Training)", linewidth=1.8, markersize=6)
    plt.plot(noise_levels, r1_m, "#7f7f7f", marker="x", linestyle=":", label="1-bit Random IRS + RZF", linewidth=1.5, markersize=6)

    plt.xlabel("Pilot Noise Floor (dBm)", fontsize=12, fontweight="bold")
    plt.ylabel("Achievable Sum-Rate (bps/Hz)", fontsize=12, fontweight="bold")
    plt.title("Phase 5: Quantization-Aware IRS Beamforming (N=500)", fontsize=13, fontweight="bold")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(fontsize=9, loc="center left")
    plt.tight_layout()
    fig1_path = os.path.join(OUT_DIR, "fig_phase5_quantization_sumrate.png")
    plt.savefig(fig1_path)
    plt.close()
    print(f"Saved {fig1_path}")

    # Figure 2: Quantization Rate Loss vs Bit Resolution (Bar Chart)
    plt.figure(figsize=(8, 5), dpi=300)
    methods = ["1-bit PTQ", "1-bit QAT", "2-bit PTQ", "2-bit QAT"]
    losses = [
        np.mean(rates["cont"]) - np.mean(rates["ptq1"]),
        np.mean(rates["cont"]) - np.mean(rates["qat1"]),
        np.mean(rates["cont"]) - np.mean(rates["ptq2"]),
        np.mean(rates["cont"]) - np.mean(rates["qat2"]),
    ]
    colors = ["#aec7e8", "#1f77b4", "#98df8a", "#2ca02c"]
    bars = plt.bar(methods, losses, color=colors, edgecolor="black", linewidth=1.2, width=0.55)
    for b, loss in zip(bars, losses):
        plt.text(b.get_x() + b.get_width()/2.0, loss + 0.05, f"{loss:.2f} bps/Hz", ha="center", va="bottom", fontweight="bold", fontsize=10)

    plt.ylabel("Sum-Rate Degradation vs Continuous (bps/Hz)", fontsize=11, fontweight="bold")
    plt.title("Quantization Degradation: PTQ vs Quantization-Aware Training (STE)", fontsize=12, fontweight="bold")
    plt.grid(True, axis="y", linestyle="--", alpha=0.6)
    plt.ylim(0, max(losses) * 1.25)
    plt.tight_layout()
    fig2_path = os.path.join(OUT_DIR, "fig_phase5_quantization_degradation.png")
    plt.savefig(fig2_path)
    plt.close()
    print(f"Saved {fig2_path}")


if __name__ == "__main__":
    run_evaluation()
