"""
Regenerate Corrected Matched Random-RZF Baselines for Original Phase 3 (500 Realizations).

This script resolves the evidence gap by directly recomputing the matched Random-IRS + RZF baseline
across all 500 original independent channel realizations stored in results/phase3_independent_eval_data.mat.

Mathematical Consistency:
  Physical composite channel: H_all_true[k, :] = (theta^*)^T * C_k
  Matched RZF effective channel: H_eff[k, :] = (theta^*)^T * C_hat_k
  Mismatched RZF (original bug): H_eff[k, :] = theta^T * C_hat_k

Outputs:
  - results/phase3_independent_per_realization_corrected.csv
  - results/phase3_independent_summary_corrected.csv
"""
from __future__ import annotations

import os
import scipy.io as sio
import scipy.stats as stats
import numpy as np
import pandas as pd
import torch

from gnn.beamforming_model import DeepBeamformingNet, compute_true_sum_rate_torch

EVAL_DATA_PATH = "results/phase3_independent_eval_data.mat"
OLD_CSV_PATH = "results/phase3_independent_per_realization.csv"
OUT_PER_CSV = "results/phase3_independent_per_realization_corrected.csv"
OUT_SUM_CSV = "results/phase3_independent_summary_corrected.csv"
MODEL_PATH = "results/gnn/best_beamforming_model.pt"


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


def main():
    print("=" * 80)
    print("REGENERATING CORRECTED MATCHED RANDOM-RZF BASELINE ON PHASE 3 (500 SAMPLES)")
    print("=" * 80)

    assert os.path.exists(EVAL_DATA_PATH), f"Missing {EVAL_DATA_PATH}"
    assert os.path.exists(OLD_CSV_PATH), f"Missing {OLD_CSV_PATH}"
    assert os.path.exists(MODEL_PATH), f"Missing {MODEL_PATH}"

    d_eval = sio.loadmat(EVAL_DATA_PATH)
    records = d_eval["records"]
    n_total = records.shape[1]
    assert n_total == 500, f"Expected 500 records, got {n_total}"

    df_orig = pd.read_csv(OLD_CSV_PATH)
    assert len(df_orig) == 500, f"Expected 500 rows in original CSV, got {len(df_orig)}"

    sigma_watts = 3.16e-12
    P_max = 10.0
    M, N, K = 64, 8, 4

    device = torch.device("cpu")
    model = DeepBeamformingNet(M=M, N=N, K=K, hidden_dim=64, n_blocks=2).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    res_matched = np.zeros(n_total)
    res_mismatched = np.zeros(n_total)
    res_ml = np.zeros(n_total)

    for i in range(n_total):
        rec = records[0, i]
        H_true = rec["H_true"]
        if H_true.shape == (1, 1):
            H_true = H_true[0, 0]
        G_true = rec["G_true"]
        if G_true.shape == (1, 1):
            G_true = G_true[0, 0]
        C_hat_cell = rec["C_hat"]
        if C_hat_cell.shape == (1, 1):
            C_hat_cell = C_hat_cell[0, 0]

        # Use MATLAB convention: rng(42 + (i + 1))
        rng = np.random.default_rng(42 + (i + 1))
        theta_rand = np.exp(1j * rng.uniform(0, 2 * np.pi, size=M))

        # 1. Matched RZF (Corrected): H_eff = conj(theta_rand)^T * C_hat
        H_eff_match = np.zeros((K, N), dtype=complex)
        for k in range(K):
            C_k = C_hat_cell[0, k]
            H_eff_match[k, :] = np.conj(theta_rand) @ C_k
        H_eff_H = H_eff_match.conj().T
        reg = H_eff_match @ H_eff_H + (K * sigma_watts / P_max) * np.eye(K)
        W_unnorm = H_eff_H @ np.linalg.inv(reg)
        for k in range(K):
            W_unnorm[:, k] = W_unnorm[:, k] / np.linalg.norm(W_unnorm[:, k])
        W_rzf_match = W_unnorm * np.sqrt(P_max / K)

        # 2. Mismatched RZF (Historical Bug): H_eff = theta_rand^T * C_hat
        H_eff_mis = np.zeros((K, N), dtype=complex)
        for k in range(K):
            C_k = C_hat_cell[0, k]
            H_eff_mis[k, :] = theta_rand @ C_k
        H_eff_H_mis = H_eff_mis.conj().T
        reg_mis = H_eff_mis @ H_eff_H_mis + (K * sigma_watts / P_max) * np.eye(K)
        W_unnorm_mis = H_eff_H_mis @ np.linalg.inv(reg_mis)
        for k in range(K):
            W_unnorm_mis[:, k] = W_unnorm_mis[:, k] / np.linalg.norm(W_unnorm_mis[:, k])
        W_rzf_mis = W_unnorm_mis * np.sqrt(P_max / K)

        # True physical channel: H_all_true = H_true^H * diag(conj(theta_rand)) * G_true
        Phi = np.diag(np.conj(theta_rand))
        H_all_true = H_true.conj().T @ Phi @ G_true

        r_match = 0.0
        r_mis = 0.0
        for k in range(K):
            S_m = np.abs(H_all_true[k, :] @ W_rzf_match[:, k]) ** 2
            IN_m = sum(np.abs(H_all_true[k, :] @ W_rzf_match[:, j]) ** 2 for j in range(K) if j != k)
            r_match += np.log2(1.0 + S_m / (IN_m + sigma_watts))

            S_mis = np.abs(H_all_true[k, :] @ W_rzf_mis[:, k]) ** 2
            IN_mis = sum(np.abs(H_all_true[k, :] @ W_rzf_mis[:, j]) ** 2 for j in range(K) if j != k)
            r_mis += np.log2(1.0 + S_mis / (IN_mis + sigma_watts))

        res_matched[i] = r_match
        res_mismatched[i] = r_mis

        # 3. Retrained Model Inference (Direct ML)
        c_k_list = [C_hat_cell[0, k] for k in range(K)]
        c_hat_np = np.stack(c_k_list, axis=0)
        c_hat_t = torch.from_numpy(c_hat_np).unsqueeze(0).to(torch.complex64)
        noise_val = float(rec["noise_dbm"][0, 0][0, 0])
        noise_t = torch.tensor([noise_val], dtype=torch.float32)
        G_t = torch.from_numpy(G_true).unsqueeze(0).to(torch.complex64)
        H_t = torch.from_numpy(H_true).unsqueeze(0).to(torch.complex64)

        with torch.no_grad():
            theta_ml, W_ml = model(c_hat_t, noise_t, P_max=P_max)
            r_ml = compute_true_sum_rate_torch(H_t, G_t, theta_ml, W_ml, sigma_watts).item()
        res_ml[i] = r_ml

    # Merge into corrected per-realization dataframe
    df_corrected = df_orig.copy()
    df_corrected["original_unconjugated_ml_rate"] = df_orig["ml_direct_rate"]
    df_corrected["ml_direct_rate"] = res_ml
    df_corrected["matched_random_rzf_rate"] = res_matched
    df_corrected["reproduced_mismatched_rzf_rate"] = res_mismatched
    df_corrected["ml_minus_matched_rzf"] = df_corrected["ml_direct_rate"] - df_corrected["matched_random_rzf_rate"]
    df_corrected.to_csv(OUT_PER_CSV, index=False)
    print(f"Saved corrected per-realization CSV: {OUT_PER_CSV}")

    # Generate summary CSV
    summary_rows = []
    noise_levels = [-100.0, -90.0, -85.0, -80.0, -70.0]

    for noise_dbm in noise_levels:
        sub = df_corrected[df_corrected["noise_floor"] == noise_dbm]
        diffs = sub["ml_minus_matched_rzf"].to_numpy()
        ml_rates = sub["ml_direct_rate"].to_numpy()
        rzf_rates = sub["matched_random_rzf_rate"].to_numpy()
        ao_rates = sub["ls_full_ao_rate"].to_numpy()
        old_rzf = sub["random_rzf_rate"].to_numpy()

        ci_low, ci_high = bootstrap_ci(diffs)
        t_stat, p_val_t = stats.ttest_rel(ml_rates, rzf_rates)
        try:
            w_stat, p_val_w = stats.wilcoxon(diffs)
        except Exception:
            p_val_w = p_val_t

        wins_vs_rzf = int((diffs > 0).sum())
        wins_vs_ao = int((sub["ml_direct_rate"] > sub["ls_full_ao_rate"]).sum())

        summary_rows.append({
            "noise_dbm": noise_dbm,
            "n_samples": len(sub),
            "ls_full_ao_mean": float(np.mean(ao_rates)),
            "ls_full_ao_std": float(np.std(ao_rates, ddof=1)),
            "ml_direct_mean": float(np.mean(ml_rates)),
            "ml_direct_std": float(np.std(ml_rates, ddof=1)),
            "matched_rzf_mean": float(np.mean(rzf_rates)),
            "matched_rzf_std": float(np.std(rzf_rates, ddof=1)),
            "buggy_old_rzf_mean": float(np.mean(old_rzf)),
            "mean_gain_over_rzf": float(np.mean(diffs)),
            "bootstrap_ci_low": ci_low,
            "bootstrap_ci_high": ci_high,
            "ml_wins_vs_rzf": wins_vs_rzf,
            "ml_win_rate_vs_rzf_pct": float(wins_vs_rzf / len(sub) * 100.0),
            "ml_wins_vs_ao": wins_vs_ao,
            "ml_win_rate_vs_ao_pct": float(wins_vs_ao / len(sub) * 100.0),
            "paired_t_pvalue": float(p_val_t),
            "wilcoxon_pvalue": float(p_val_w),
        })

    # Overall across all 500 samples
    all_diffs = df_corrected["ml_minus_matched_rzf"].to_numpy()
    all_ml = df_corrected["ml_direct_rate"].to_numpy()
    all_rzf = df_corrected["matched_random_rzf_rate"].to_numpy()
    all_ao = df_corrected["ls_full_ao_rate"].to_numpy()
    all_old_rzf = df_corrected["random_rzf_rate"].to_numpy()

    ci_low_all, ci_high_all = bootstrap_ci(all_diffs)
    t_stat_all, p_val_t_all = stats.ttest_rel(all_ml, all_rzf)
    w_stat_all, p_val_w_all = stats.wilcoxon(all_diffs)
    wins_all_rzf = int((all_diffs > 0).sum())
    wins_all_ao = int((df_corrected["ml_direct_rate"] > df_corrected["ls_full_ao_rate"]).sum())

    summary_rows.append({
        "noise_dbm": "Overall",
        "n_samples": len(df_corrected),
        "ls_full_ao_mean": float(np.mean(all_ao)),
        "ls_full_ao_std": float(np.std(all_ao, ddof=1)),
        "ml_direct_mean": float(np.mean(all_ml)),
        "ml_direct_std": float(np.std(all_ml, ddof=1)),
        "matched_rzf_mean": float(np.mean(all_rzf)),
        "matched_rzf_std": float(np.std(all_rzf, ddof=1)),
        "buggy_old_rzf_mean": float(np.mean(all_old_rzf)),
        "mean_gain_over_rzf": float(np.mean(all_diffs)),
        "bootstrap_ci_low": ci_low_all,
        "bootstrap_ci_high": ci_high_all,
        "ml_wins_vs_rzf": wins_all_rzf,
        "ml_win_rate_vs_rzf_pct": float(wins_all_rzf / len(df_corrected) * 100.0),
        "ml_wins_vs_ao": wins_all_ao,
        "ml_win_rate_vs_ao_pct": float(wins_all_ao / len(df_corrected) * 100.0),
        "paired_t_pvalue": float(p_val_t_all),
        "wilcoxon_pvalue": float(p_val_w_all),
    })

    df_sum = pd.DataFrame(summary_rows)
    df_sum.to_csv(OUT_SUM_CSV, index=False)
    print(f"Saved corrected summary CSV: {OUT_SUM_CSV}")

    print("\n" + "=" * 80)
    print("PHASE 3 CORRECTED RESULTS SUMMARY")
    print("=" * 80)
    print(df_sum[["noise_dbm", "ls_full_ao_mean", "ml_direct_mean", "matched_rzf_mean", "buggy_old_rzf_mean", "mean_gain_over_rzf", "ml_win_rate_vs_rzf_pct", "paired_t_pvalue"]].to_string())


if __name__ == "__main__":
    main()
