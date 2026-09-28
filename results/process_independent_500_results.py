"""
Phase 3 -- Comprehensive Results Processing & Statistical Analysis (500 Independent Realizations).

Merges multi-worker results, computes paired statistics, exports CSVs,
generates publication-quality figures, and performs runtime benchmarking.
"""
from __future__ import annotations

import glob
import os
import re
import numpy as np
import scipy.io as sio
import scipy.stats as stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS_DIR = "results"
EVAL_DATA_PATH = os.path.join(RESULTS_DIR, "phase3_independent_eval_data.mat")
MERGED_RESULTS_PATH = os.path.join(RESULTS_DIR, "phase3_independent_evaluation_results.mat")
PER_REALIZATION_CSV = os.path.join(RESULTS_DIR, "phase3_independent_per_realization.csv")
SUMMARY_CSV = os.path.join(RESULTS_DIR, "phase3_independent_summary.csv")

FIG_SUMRATE = os.path.join(RESULTS_DIR, "fig_phase3_independent_sumrate.png")
FIG_GAIN = os.path.join(RESULTS_DIR, "fig_phase3_independent_gain.png")
FIG_PAIRED = os.path.join(RESULTS_DIR, "fig_phase3_independent_paired.png")
FIG_DISTRIBUTION = os.path.join(RESULTS_DIR, "fig_phase3_independent_distribution.png")
FIG_CDF = os.path.join(RESULTS_DIR, "fig_phase3_independent_cdf.png")


def merge_worker_results(n_total: int = 500) -> dict:
    """Merge results from partial worker files or completed part files."""
    if os.path.exists(MERGED_RESULTS_PATH):
        print(f"Loading existing merged results from {MERGED_RESULTS_PATH}...")
        return sio.loadmat(MERGED_RESULTS_PATH)

    # Search for part files
    part_files = sorted(glob.glob(os.path.join(RESULTS_DIR, "phase3_independent_results_part*.mat")))
    if not part_files:
        # Fall back to checkpoint files
        part_files = sorted(glob.glob(os.path.join(RESULTS_DIR, "phase3_independent_checkpoint_part*.mat")))

    if not part_files:
        raise FileNotFoundError("No worker result or checkpoint files found in results/!")

    print(f"Found {len(part_files)} worker files to merge: {part_files}")

    # Load master records from eval_data to index seeds
    d_eval = sio.loadmat(EVAL_DATA_PATH)
    eval_records = d_eval["records"]
    assert eval_records.shape[1] == n_total, f"Expected {n_total} records, got {eval_records.shape[1]}"

    # Preallocate merged arrays
    merged = {
        "seeds": np.zeros(n_total, dtype=int),
        "noise_vec": np.zeros(n_total, dtype=float),
        "trials": np.zeros(n_total, dtype=int),
        "res_perfect": np.zeros(n_total, dtype=float),
        "res_ls_full": np.zeros(n_total, dtype=float),
        "res_ls_trunc1": np.zeros(n_total, dtype=float),
        "res_rand_rzf": np.zeros(n_total, dtype=float),
        "res_ml_direct": np.zeros(n_total, dtype=float),
        "res_ml_trunc1": np.zeros(n_total, dtype=float),
        "time_ls_full": np.zeros(n_total, dtype=float),
        "iters_ls_full": np.zeros(n_total, dtype=float),
        "time_ls_trunc1": np.zeros(n_total, dtype=float),
        "time_ml_trunc1": np.zeros(n_total, dtype=float),
        "time_ml_direct": np.zeros(n_total, dtype=float),
        "time_ls_est": np.zeros(n_total, dtype=float),
    }

    covered_indices = set()

    for pf in part_files:
        d = sio.loadmat(pf)
        s_idx = int(d["start_idx"].item())
        e_idx = int(d["end_idx"].item())
        n_part = e_idx - s_idx + 1

        # Check mask if checkpoint
        if "completed_mask" in d:
            mask = d["completed_mask"].flatten().astype(bool)
        else:
            mask = np.ones(n_part, dtype=bool)

        for local_i in range(n_part):
            if mask[local_i]:
                global_i = s_idx - 1 + local_i
                covered_indices.add(global_i)
                merged["seeds"][global_i] = int(d["seeds"].flatten()[local_i])
                merged["noise_vec"][global_i] = float(d["noise_vec"].flatten()[local_i])
                merged["trials"][global_i] = int(d["trials"].flatten()[local_i])
                merged["res_perfect"][global_i] = float(d["res_perfect"].flatten()[local_i])
                merged["res_ls_full"][global_i] = float(d["res_ls_full"].flatten()[local_i])
                merged["res_ls_trunc1"][global_i] = float(d["res_ls_trunc1"].flatten()[local_i])
                merged["res_rand_rzf"][global_i] = float(d["res_rand_rzf"].flatten()[local_i])
                merged["res_ml_direct"][global_i] = float(d["res_ml_direct"].flatten()[local_i])
                merged["res_ml_trunc1"][global_i] = float(d["res_ml_trunc1"].flatten()[local_i])
                merged["time_ls_full"][global_i] = float(d["time_ls_full"].flatten()[local_i])
                merged["iters_ls_full"][global_i] = float(d["iters_ls_full"].flatten()[local_i])
                merged["time_ls_trunc1"][global_i] = float(d["time_ls_trunc1"].flatten()[local_i])
                merged["time_ml_trunc1"][global_i] = float(d["time_ml_trunc1"].flatten()[local_i])
                merged["time_ml_direct"][global_i] = float(d["time_ml_direct"].flatten()[local_i])
                merged["time_ls_est"][global_i] = float(d["time_ls_est"].flatten()[local_i])

    print(f"Total realizations merged: {len(covered_indices)}/{n_total}")
    if len(covered_indices) == n_total:
        sio.savemat(MERGED_RESULTS_PATH, merged)
        print(f"Saved complete merged results to {MERGED_RESULTS_PATH}")

    return merged


def export_per_realization_csv(data: dict):
    """Save detailed per-realization results."""
    n_total = len(data["noise_vec"])
    with open(PER_REALIZATION_CSV, "w", encoding="utf-8") as f:
        headers = [
            "test_seed", "noise_floor", "trial",
            "ls_full_ao_rate", "ml_direct_rate", "perfect_csi_rate", "random_rzf_rate",
            "ls_ao1_rate", "ml_ao1_rate",
            "ml_minus_ls_diff", "ml_over_ls_ratio",
            "time_ml_direct", "time_ls_full", "time_ls_est", "time_ml_ao1", "time_ls_ao1"
        ]
        f.write(",".join(headers) + "\n")
        for i in range(n_total):
            seed = data["seeds"][i]
            noise = data["noise_vec"][i]
            trial = data["trials"][i]
            r_ls = data["res_ls_full"][i]
            r_ml = data["res_ml_direct"][i]
            r_perf = data["res_perfect"][i]
            r_rand = data["res_rand_rzf"][i]
            r_ls1 = data["res_ls_trunc1"][i]
            r_ml1 = data["res_ml_trunc1"][i]
            diff = r_ml - r_ls
            ratio = r_ml / max(1e-6, r_ls)
            t_ml = data["time_ml_direct"][i]
            t_ls = data["time_ls_full"][i]
            t_est = data["time_ls_est"][i]
            t_ml1 = data["time_ml_trunc1"][i]
            t_ls1 = data["time_ls_trunc1"][i]

            row = [
                f"{seed}", f"{noise:.1f}", f"{trial}",
                f"{r_ls:.4f}", f"{r_ml:.4f}", f"{r_perf:.4f}", f"{r_rand:.4f}",
                f"{r_ls1:.4f}", f"{r_ml1:.4f}",
                f"{diff:+.4f}", f"{ratio:.4f}",
                f"{t_ml:.6f}", f"{t_ls:.4f}", f"{t_est:.6f}", f"{t_ml1:.4f}", f"{t_ls1:.4f}"
            ]
            f.write(",".join(row) + "\n")
    print(f"Saved {PER_REALIZATION_CSV} successfully.")


def compute_statistical_summary(data: dict) -> list[dict]:
    """Compute per-noise-level statistical summaries and paired tests."""
    noise_levels = [-100.0, -90.0, -85.0, -80.0, -70.0]
    nv = data["noise_vec"]
    r_ls = data["res_ls_full"]
    r_ml = data["res_ml_direct"]
    r_perf = data["res_perfect"]
    r_rand = data["res_rand_rzf"]
    r_ls1 = data["res_ls_trunc1"]
    r_ml1 = data["res_ml_trunc1"]

    summary_rows = []

    for nl in noise_levels:
        mask = (nv == nl)
        n_pts = np.sum(mask)
        if n_pts == 0:
            continue

        sub_ls = r_ls[mask]
        sub_ml = r_ml[mask]
        sub_perf = r_perf[mask]
        sub_rand = r_rand[mask]
        sub_ls1 = r_ls1[mask]
        sub_ml1 = r_ml1[mask]

        diffs = sub_ml - sub_ls
        wins = int(np.sum(diffs > 0))
        losses = int(np.sum(diffs < 0))
        ties = int(np.sum(diffs == 0))
        win_rate = (wins / n_pts) * 100.0

        mean_diff = float(np.mean(diffs))
        med_diff = float(np.median(diffs))
        std_diff = float(np.std(diffs, ddof=1)) if n_pts > 1 else 0.0

        # 95% Confidence Interval for paired difference
        se_diff = std_diff / np.sqrt(n_pts)
        ci_t = stats.t.ppf(0.975, df=n_pts - 1) if n_pts > 1 else 1.96
        ci_low = mean_diff - ci_t * se_diff
        ci_high = mean_diff + ci_t * se_diff

        # Paired tests
        if n_pts > 1 and not np.all(diffs == 0):
            t_res = stats.ttest_rel(sub_ml, sub_ls)
            p_ttest = float(t_res.pvalue)
            try:
                w_res = stats.wilcoxon(sub_ml, sub_ls)
                p_wilcoxon = float(w_res.pvalue)
            except Exception:
                p_wilcoxon = np.nan
        else:
            p_ttest = np.nan
            p_wilcoxon = np.nan

        row = {
            "noise_dbm": nl,
            "n": int(n_pts),
            "ls_mean": float(np.mean(sub_ls)),
            "ls_median": float(np.median(sub_ls)),
            "ls_std": float(np.std(sub_ls, ddof=1)),
            "ls_min": float(np.min(sub_ls)),
            "ls_max": float(np.max(sub_ls)),
            "ml_mean": float(np.mean(sub_ml)),
            "ml_median": float(np.median(sub_ml)),
            "ml_std": float(np.std(sub_ml, ddof=1)),
            "ml_min": float(np.min(sub_ml)),
            "ml_max": float(np.max(sub_ml)),
            "perfect_mean": float(np.mean(sub_perf)),
            "perfect_median": float(np.median(sub_perf)),
            "random_rzf_mean": float(np.mean(sub_rand)),
            "random_rzf_median": float(np.median(sub_rand)),
            "ls_ao1_mean": float(np.mean(sub_ls1)),
            "ml_ao1_mean": float(np.mean(sub_ml1)),
            "absolute_gain": mean_diff,
            "median_gain": med_diff,
            "percentage_gain": float((np.mean(sub_ml) - np.mean(sub_ls)) / max(1e-6, np.mean(sub_ls)) * 100.0),
            "win_rate": win_rate,
            "wins": wins,
            "losses": losses,
            "ties": ties,
            "mean_difference_ci_low": ci_low,
            "mean_difference_ci_high": ci_high,
            "paired_t_pvalue": p_ttest,
            "wilcoxon_pvalue": p_wilcoxon,
        }
        summary_rows.append(row)

    return summary_rows


def export_summary_csv(summary_rows: list[dict]):
    """Save final summary table to CSV."""
    fieldnames = [
        "noise_dbm", "n",
        "ls_mean", "ls_median", "ls_std",
        "ml_mean", "ml_median", "ml_std",
        "perfect_mean", "random_rzf_mean", "ls_ao1_mean", "ml_ao1_mean",
        "absolute_gain", "percentage_gain",
        "win_rate", "wins", "losses",
        "mean_difference_ci_low", "mean_difference_ci_high",
        "paired_t_pvalue", "wilcoxon_pvalue"
    ]
    with open(SUMMARY_CSV, "w", encoding="utf-8") as f:
        f.write(",".join(fieldnames) + "\n")
        for r in summary_rows:
            line = [
                f"{r['noise_dbm']:.1f}", f"{r['n']}",
                f"{r['ls_mean']:.4f}", f"{r['ls_median']:.4f}", f"{r['ls_std']:.4f}",
                f"{r['ml_mean']:.4f}", f"{r['ml_median']:.4f}", f"{r['ml_std']:.4f}",
                f"{r['perfect_mean']:.4f}", f"{r['random_rzf_mean']:.4f}",
                f"{r['ls_ao1_mean']:.4f}", f"{r['ml_ao1_mean']:.4f}",
                f"{r['absolute_gain']:+.4f}", f"{r['percentage_gain']:+.2f}",
                f"{r['win_rate']:.2f}", f"{r['wins']}", f"{r['losses']}",
                f"{r['mean_difference_ci_low']:+.4f}", f"{r['mean_difference_ci_high']:+.4f}",
                f"{r['paired_t_pvalue']:.4e}" if not np.isnan(r['paired_t_pvalue']) else "N/A",
                f"{r['wilcoxon_pvalue']:.4e}" if not np.isnan(r['wilcoxon_pvalue']) else "N/A"
            ]
            f.write(",".join(line) + "\n")
    print(f"Saved {SUMMARY_CSV} successfully.")


def plot_figures(data: dict, summary_rows: list[dict]):
    """Generate the 5 required publication-quality figures."""
    noise_levels = np.array([r["noise_dbm"] for r in summary_rows])

    # 1. Sum-rate vs Noise Floor
    plt.figure(figsize=(8, 5.5), dpi=300)
    perf_means = [r["perfect_mean"] for r in summary_rows]
    ml_means = [r["ml_mean"] for r in summary_rows]
    ls_means = [r["ls_mean"] for r in summary_rows]
    rand_means = [r["random_rzf_mean"] for r in summary_rows]
    ml_stds = [r["ml_std"] for r in summary_rows]
    ls_stds = [r["ls_std"] for r in summary_rows]

    plt.plot(noise_levels, perf_means, "k--o", label="Perfect-CSI AO (Upper Bound)", linewidth=2, markersize=7)
    plt.plot(noise_levels, ml_means, "#1f77b4", marker="s", label="Proposed Deep Beamforming (Direct ML)", linewidth=2.2, markersize=7)
    plt.plot(noise_levels, ls_means, "#d62728", marker="^", label="Conventional LS -> Full AO", linewidth=2.2, markersize=7)
    plt.plot(noise_levels, rand_means, "#7f7f7f", marker="x", linestyle=":", label="Random IRS + RZF", linewidth=1.8, markersize=7)

    # Shaded error bands
    plt.fill_between(noise_levels, np.array(ml_means) - np.array(ml_stds), np.array(ml_means) + np.array(ml_stds), color="#1f77b4", alpha=0.15)
    plt.fill_between(noise_levels, np.array(ls_means) - np.array(ls_stds), np.array(ls_means) + np.array(ls_stds), color="#d62728", alpha=0.15)

    plt.xlabel("Pilot Noise Floor (dBm)", fontsize=12, fontweight="bold")
    plt.ylabel("Achievable Sum-Rate (bps/Hz)", fontsize=12, fontweight="bold")
    plt.title(f"Independent Monte Carlo Verification (N={len(data['seeds'])})", fontsize=13, fontweight="bold")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(fontsize=10, loc="center left")
    plt.tight_layout()
    plt.savefig(FIG_SUMRATE)
    plt.close()
    print(f"Saved {FIG_SUMRATE}")

    # 2. Gain vs Noise Floor (with 95% CI error bars)
    plt.figure(figsize=(8, 5.5), dpi=300)
    gains = [r["absolute_gain"] for r in summary_rows]
    ci_lows = [r["mean_difference_ci_low"] for r in summary_rows]
    ci_highs = [r["mean_difference_ci_high"] for r in summary_rows]
    yerr_lower = np.array(gains) - np.array(ci_lows)
    yerr_upper = np.array(ci_highs) - np.array(gains)

    plt.errorbar(noise_levels, gains, yerr=[yerr_lower, yerr_upper],
                 fmt="o-", color="#2ca02c", ecolor="#2ca02c", elinewidth=2, capsize=5, capthick=1.5,
                 linewidth=2.2, markersize=8, label="Mean Paired Gain (ML - LS) ± 95% CI")
    plt.axhline(0, color="gray", linestyle="--", linewidth=1.2)
    plt.xlabel("Pilot Noise Floor (dBm)", fontsize=12, fontweight="bold")
    plt.ylabel("Sum-Rate Difference: $R_{\\mathrm{ML}} - R_{\\mathrm{LS}}$ (bps/Hz)", fontsize=12, fontweight="bold")
    plt.title("Net Beamforming Gain vs Pilot Noise Floor (95% CI)", fontsize=13, fontweight="bold")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(FIG_GAIN)
    plt.close()
    print(f"Saved {FIG_GAIN}")

    # 3. Per-realization Paired Scatter Plot
    plt.figure(figsize=(7.5, 7.5), dpi=300)
    r_ls_all = data["res_ls_full"]
    r_ml_all = data["res_ml_direct"]
    nv_all = data["noise_vec"]

    colors = {-100.0: "#440154", -90.0: "#3b528b", -85.0: "#21918c", -80.0: "#5ec962", -70.0: "#fde725"}
    for nl in [-100.0, -90.0, -85.0, -80.0, -70.0]:
        mask = (nv_all == nl)
        plt.scatter(r_ls_all[mask], r_ml_all[mask], label=f"{nl:.0f} dBm", color=colors[nl], alpha=0.75, edgecolors="none", s=35)

    lim_max = max(np.max(r_ls_all), np.max(r_ml_all)) + 3
    plt.plot([0, lim_max], [0, lim_max], "r--", linewidth=1.8, label="Unity Line (ML = LS)")
    plt.fill_between([0, lim_max], [0, lim_max], [lim_max, lim_max], color="green", alpha=0.06, label="ML Advantage Zone")
    plt.xlim(0, lim_max)
    plt.ylim(0, lim_max)
    plt.xlabel("Conventional LS -> Full AO Sum-Rate (bps/Hz)", fontsize=12, fontweight="bold")
    plt.ylabel("Proposed Deep Beamforming Sum-Rate (bps/Hz)", fontsize=12, fontweight="bold")
    plt.title(f"Paired Evaluation Scatter: 500 Realizations", fontsize=13, fontweight="bold")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(fontsize=10, loc="lower right")
    plt.tight_layout()
    plt.savefig(FIG_PAIRED)
    plt.close()
    print(f"Saved {FIG_PAIRED}")

    # 4. Distribution / Boxplot of Gain
    plt.figure(figsize=(8.5, 5.5), dpi=300)
    diffs_per_noise = [data["res_ml_direct"][nv_all == nl] - data["res_ls_full"][nv_all == nl] for nl in noise_levels]
    box = plt.boxplot(diffs_per_noise, positions=range(len(noise_levels)), patch_artist=True, widths=0.55,
                      medianprops=dict(color="red", linewidth=2.0))
    for patch in box["boxes"]:
        patch.set_facecolor("#9ecae1")
        patch.set_alpha(0.8)

    plt.axhline(0, color="black", linestyle="--", linewidth=1.2)
    plt.xticks(range(len(noise_levels)), [f"{nl:.0f} dBm" for nl in noise_levels], fontsize=11, fontweight="bold")
    plt.xlabel("Pilot Noise Floor", fontsize=12, fontweight="bold")
    plt.ylabel("Paired Gain: $R_{\\mathrm{ML}} - R_{\\mathrm{LS}}$ (bps/Hz)", fontsize=12, fontweight="bold")
    plt.title("Distribution of Paired Beamforming Advantage (N=100 per Noise Level)", fontsize=13, fontweight="bold")
    plt.grid(True, linestyle="--", alpha=0.5, axis="y")
    plt.tight_layout()
    plt.savefig(FIG_DISTRIBUTION)
    plt.close()
    print(f"Saved {FIG_DISTRIBUTION}")

    # 5. Empirical CDF of Paired Improvement
    plt.figure(figsize=(8, 5.5), dpi=300)
    cdf_colors = {-100.0: "#807dba", -90.0: "#4292c6", -85.0: "#41ab5d", -80.0: "#fe9929", -70.0: "#d94701"}
    for nl in noise_levels:
        diffs = np.sort(data["res_ml_direct"][nv_all == nl] - data["res_ls_full"][nv_all == nl])
        cdf = np.arange(1, len(diffs) + 1) / len(diffs)
        plt.step(diffs, cdf, label=f"{nl:.0f} dBm", color=cdf_colors[nl], linewidth=2.2, where="post")

    plt.axvline(0, color="gray", linestyle="--", linewidth=1.2)
    plt.xlabel("Paired Difference: $R_{\\mathrm{ML}} - R_{\\mathrm{LS}}$ (bps/Hz)", fontsize=12, fontweight="bold")
    plt.ylabel("Empirical Cumulative Probability", fontsize=12, fontweight="bold")
    plt.title("CDF of Paired ML Improvement Across Noise Levels", fontsize=13, fontweight="bold")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(fontsize=10, loc="lower right")
    plt.tight_layout()
    plt.savefig(FIG_CDF)
    plt.close()
    print(f"Saved {FIG_CDF}")


def benchmark_runtime(data: dict):
    """Accurately compute execution latencies."""
    t_ml_inf = data["time_ml_direct"]
    t_ls_est = data["time_ls_est"]
    t_ls_ao = data["time_ls_full"]
    t_iters = data["iters_ls_full"]

    t_ml_total = t_ls_est + t_ml_inf
    t_ls_total = t_ls_est + t_ls_ao

    print("\n" + "=" * 80)
    print("DETAILED RUNTIME BENCHMARK (APPLES-TO-APPLES LATENCY COMPARISON)")
    print("=" * 80)
    print(f"Number of evaluated realizations: {len(t_ml_inf)}")
    print(f"1. Multi-phase LS Channel Estimation: {np.mean(t_ls_est)*1000:.3f} ms ± {np.std(t_ls_est)*1000:.3f} ms")
    print(f"2. Deep Beamforming Pure Inference:   {np.mean(t_ml_inf)*1000:.3f} ms ± {np.std(t_ml_inf)*1000:.3f} ms")
    print(f"   -> ML Total Pipeline (Est + Infer):{np.mean(t_ml_total)*1000:.3f} ms ± {np.std(t_ml_total)*1000:.3f} ms")
    print(f"3. Conventional LS -> Full AO Solver: {np.mean(t_ls_ao):.3f} s ± {np.std(t_ls_ao):.3f} s (avg {np.mean(t_iters):.1f} iters)")
    print(f"   -> LS Total Pipeline (Est + AO):   {np.mean(t_ls_total):.3f} s ± {np.std(t_ls_total):.3f} s")
    print("-" * 80)
    speedup_optimization_only = np.mean(t_ls_ao) / max(1e-6, np.mean(t_ml_inf))
    speedup_end_to_end = np.mean(t_ls_total) / max(1e-6, np.mean(t_ml_total))
    print(f"Speedup (Optimization only: Full AO vs ML Forward Pass): {speedup_optimization_only:.1f}x")
    print(f"Speedup (End-to-End Pipeline: LS+AO vs LS+ML):            {speedup_end_to_end:.1f}x")
    print("=" * 80 + "\n")


def print_formatted_summary(summary_rows: list[dict], data: dict):
    """Print complete summary table and overall statistics to stdout."""
    print("\n" + "=" * 105)
    print("PHASE 3 INDEPENDENT MONTE CARLO VERIFICATION RESULTS (500 REALIZATIONS)")
    print("=" * 105)
    print(f"{'Noise':>7} | {'N':>4} | {'Rand RZF':>8} | {'LS Full AO':>10} | {'ML Direct':>10} | {'ML+AO(1)':>9} | {'Perf CSI':>8} | {'Gain':>8} | {'Win Rate':>8} | {'p (t-test)':>11}")
    print("-" * 105)
    for r in summary_rows:
        p_str = f"{r['paired_t_pvalue']:.2e}" if not np.isnan(r['paired_t_pvalue']) else "N/A"
        print(f"{r['noise_dbm']:6.1f}  | {r['n']:4d} | {r['random_rzf_mean']:8.2f} | {r['ls_mean']:10.2f} | {r['ml_mean']:10.2f} | {r['ml_ao1_mean']:9.2f} | {r['perfect_mean']:8.2f} | {r['absolute_gain']:+8.2f} | {r['win_rate']:7.1f}% | {p_str:>11}")
    print("=" * 105)

    # Overall across all 500 realizations
    nv_all = data["noise_vec"]
    r_ls_all = data["res_ls_full"]
    r_ml_all = data["res_ml_direct"]
    diff_all = r_ml_all - r_ls_all
    overall_ls_mean = float(np.mean(r_ls_all))
    overall_ml_mean = float(np.mean(r_ml_all))
    overall_gain = float(np.mean(diff_all))
    overall_pct_gain = (overall_gain / overall_ls_mean) * 100.0
    overall_wins = int(np.sum(diff_all > 0))
    overall_win_rate = (overall_wins / len(diff_all)) * 100.0

    std_all = float(np.std(diff_all, ddof=1))
    se_all = std_all / np.sqrt(len(diff_all))
    ci_low_all = overall_gain - 1.96 * se_all
    ci_high_all = overall_gain + 1.96 * se_all
    t_all = stats.ttest_rel(r_ml_all, r_ls_all)
    w_all = stats.wilcoxon(r_ml_all, r_ls_all)

    print("\nOVERALL METRICS ACROSS ALL 500 INDEPENDENT REALIZATIONS:")
    print(f"  Overall Mean LS Full AO Sum-Rate: {overall_ls_mean:.2f} bps/Hz")
    print(f"  Overall Mean ML Direct Sum-Rate:  {overall_ml_mean:.2f} bps/Hz")
    print(f"  Overall Mean Gain (ML - LS):      {overall_gain:+.2f} bps/Hz ({overall_pct_gain:+.1f}%)")
    print(f"  Overall Paired Win Rate:          {overall_win_rate:.1f}% ({overall_wins}/500)")
    print(f"  Overall 95% Confidence Interval:  [{ci_low_all:+.2f}, {ci_high_all:+.2f}] bps/Hz")
    print(f"  Overall Paired t-test p-value:    {t_all.pvalue:.4e}")
    print(f"  Overall Wilcoxon test p-value:    {w_all.pvalue:.4e}")
    print("=" * 105 + "\n")


def main():
    data = merge_worker_results(n_total=500)
    export_per_realization_csv(data)
    summary_rows = compute_statistical_summary(data)
    export_summary_csv(summary_rows)
    plot_figures(data, summary_rows)
    benchmark_runtime(data)
    print_formatted_summary(summary_rows, data)


if __name__ == "__main__":
    main()
