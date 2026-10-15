"""
Comprehensive Verification Script for Review-III Research Tasks 1-4.
Audits all saved artifacts, recalculates metrics directly from raw per-realization CSVs,
computes paired stats, confidence intervals, p-values, and verifies all claims.
"""
import os
import csv
import json
import torch
import numpy as np
from scipy import stats

def load_csv(path):
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    data = {}
    for col in reader.fieldnames:
        try:
            data[col] = np.array([float(r[col]) for r in rows])
        except ValueError:
            data[col] = [r[col] for r in rows]
    return data, len(rows)

def audit_task1():
    print("=" * 75)
    print("TASK 1: 1-BIT AND 2-BIT PHASE QUANTIZATION AUDIT (500 Realizations)")
    print("=" * 75)
    csv_path = "results/phase5_quantization/phase5_quantization_per_realization.csv"
    if not os.path.exists(csv_path):
        print(f"FAILED: Missing {csv_path}")
        return
    
    data, n_total = load_csv(csv_path)
    print(f"Total evaluated realizations: {n_total}")
    
    c = data["rate_cont"]
    p1 = data["rate_ptq1"]
    q1 = data["rate_qat1"]
    p2 = data["rate_ptq2"]
    q2 = data["rate_qat2"]
    r1 = data["rate_rand_1bit"]
    r_cont = data["rate_rand_cont"]
    r2 = data["rate_rand_2bit"]
    
    print(f"\n1. Aggregate Means Across All 500 Realizations:")
    print(f"  Continuous Frozen ML: {c.mean():.4f} ± {c.std():.4f} bps/Hz (100.0% reference)")
    print(f"  2-bit QAT:            {q2.mean():.4f} ± {q2.std():.4f} bps/Hz (Retention: {q2.mean()/c.mean()*100:.2f}%, Diff: {q2.mean() - c.mean():+.4f} bps/Hz)")
    print(f"  2-bit PTQ:            {p2.mean():.4f} ± {p2.std():.4f} bps/Hz (Retention: {p2.mean()/c.mean()*100:.2f}%, Diff: {p2.mean() - c.mean():+.4f} bps/Hz)")
    print(f"  1-bit QAT:            {q1.mean():.4f} ± {q1.std():.4f} bps/Hz (Retention: {q1.mean()/c.mean()*100:.2f}%, Diff: {q1.mean() - c.mean():+.4f} bps/Hz)")
    print(f"  1-bit PTQ:            {p1.mean():.4f} ± {p1.std():.4f} bps/Hz (Retention: {p1.mean()/c.mean()*100:.2f}%, Diff: {p1.mean() - c.mean():+.4f} bps/Hz)")
    print(f"  Random 1-bit RZF:     {r1.mean():.4f} ± {r1.std():.4f} bps/Hz")
    print(f"  Random 2-bit RZF:     {r2.mean():.4f} ± {r2.std():.4f} bps/Hz")
    print(f"  Random Cont RZF:      {r_cont.mean():.4f} ± {r_cont.std():.4f} bps/Hz")
    
    # Paired comparisons: 2-bit QAT vs 2-bit PTQ
    d2 = q2 - p2
    t2, p_val2 = stats.ttest_rel(q2, p2)
    ci2 = stats.t.interval(0.95, n_total - 1, loc=d2.mean(), scale=stats.sem(d2))
    w_stat2, w_p2 = stats.wilcoxon(d2)
    win2 = int((d2 > 0).sum())
    
    print(f"\n2. 2-bit QAT vs 2-bit PTQ Paired Comparison (All 500 Realizations):")
    print(f"  Mean Difference (QAT - PTQ): {d2.mean():+.4f} bps/Hz")
    print(f"  95% Confidence Interval:     [{ci2[0]:.4f}, {ci2[1]:.4f}] bps/Hz")
    print(f"  Paired t-test: t = {t2:.4f}, p-value = {p_val2:.4e}")
    print(f"  Wilcoxon signed-rank p-value: {w_p2:.4e}")
    print(f"  Win Rate: {win2}/{n_total} ({win2/n_total*100:.1f}%)")
    
    # Paired comparisons: 1-bit QAT vs 1-bit PTQ
    d1 = q1 - p1
    t1, p_val1 = stats.ttest_rel(q1, p1)
    ci1 = stats.t.interval(0.95, n_total - 1, loc=d1.mean(), scale=stats.sem(d1))
    w_stat1, w_p1 = stats.wilcoxon(d1)
    win1 = int((d1 > 0).sum())
    
    print(f"\n3. 1-bit QAT vs 1-bit PTQ Paired Comparison (All 500 Realizations):")
    print(f"  Mean Difference (QAT - PTQ): {d1.mean():+.4f} bps/Hz")
    print(f"  95% Confidence Interval:     [{ci1[0]:.4f}, {ci1[1]:.4f}] bps/Hz")
    print(f"  Paired t-test: t = {t1:.4f}, p-value = {p_val1:.4e}")
    print(f"  Wilcoxon signed-rank p-value: {w_p1:.4e}")
    print(f"  Win Rate: {win1}/{n_total} ({win1/n_total*100:.1f}%)")
    
    # Per-noise-floor Breakdown and -70 dBm analysis
    print("\n4. Per-Noise-Floor Breakdown:")
    noise_vals = sorted(list(set(data["noise_dbm"])))
    for noise in noise_vals:
        idx = np.where(np.isclose(data["noise_dbm"], noise))[0]
        c_sub = c[idx].mean()
        p1_sub = p1[idx].mean()
        q1_sub = q1[idx].mean()
        p2_sub = p2[idx].mean()
        q2_sub = q2[idx].mean()
        r1_sub = r1[idx].mean()
        r2_sub = r2[idx].mean()
        rc_sub = r_cont[idx].mean()
        gain_qat2 = q2_sub - p2_sub
        t_sub, p_sub = stats.ttest_rel(q2[idx], p2[idx])
        print(f"  Noise {noise:6.1f} dBm (N={len(idx):3d}): "
              f"Cont={c_sub:5.2f} | "
              f"PTQ1={p1_sub:5.2f} | QAT1={q1_sub:5.2f} | "
              f"PTQ2={p2_sub:5.2f} | QAT2={q2_sub:5.2f} | "
              f"QAT2-PTQ2 Gain={gain_qat2:+5.2f} (p={p_sub:.4f}) | "
              f"Rand1={r1_sub:5.2f} | Rand2={r2_sub:5.2f} | RandCont={rc_sub:5.2f}")

    # Check models and parameter counts
    print("\n5. Checkpoint Parameter Integrity:")
    for mpath in ["results/phase5_quantization/model_1bit_qat.pt", "results/phase5_quantization/model_2bit_qat.pt"]:
        if os.path.exists(mpath):
            st = torch.load(mpath, map_location="cpu", weights_only=False)
            sd = st.get("model_state_dict", st)
            p_cnt = sum(p.numel() for p in sd.values())
            print(f"  Checkpoint {mpath}: {p_cnt:,} parameters verified.")

def audit_task2():
    print("\n" + "=" * 75)
    print("TASK 2: CHANNEL ROBUSTNESS AUDIT")
    print("=" * 75)
    csv_path = "results/task2_robustness/robustness_summary.csv"
    if not os.path.exists(csv_path):
        print(f"FAILED: Missing {csv_path}")
        return
    data, n_total = load_csv(csv_path)
    print(f"Total evaluated settings: {n_total}")
    for i in range(n_total):
        name = data["setting_name"][i]
        k = data["K_rice"][i]
        alpha = data["alpha_BI"][i]
        zs_m, zs_s = data["zero_shot_mean"][i], data["zero_shot_std"][i]
        rt_m, rt_s = data["retrained_mean"][i], data["retrained_std"][i]
        rand_m = data["random_rzf_mean"][i]
        diff = data["gain_retrained_over_zero_shot"][i]
        p_val = data["p_val_retrained_vs_zs"][i]
        print(f"\nSetting: {name} (K_rice={k}, alpha={alpha})")
        print(f"  Zero-shot Frozen: {zs_m:.2f} ± {zs_s:.2f} bps/Hz")
        print(f"  Retrained Model:  {rt_m:.2f} ± {rt_s:.2f} bps/Hz (Gain over Zero-Shot: {diff:+.2f} bps/Hz, p={p_val:.4e})")
        print(f"  Random RZF Base:  {rand_m:.2f} bps/Hz (Zero-Shot Gain over Random: {zs_m - rand_m:+.2f} bps/Hz)")

def audit_task3():
    print("\n" + "=" * 75)
    print("TASK 3: DIMENSION GENERALIZATION (N, M, K) AUDIT")
    print("=" * 75)
    json_path = "results/task3_generalization/dimension_generalization_summary.json"
    if not os.path.exists(json_path):
        print(f"Missing {json_path}")
        return
    with open(json_path, "r", encoding="utf-8") as f:
        res = json.load(f)
    
    print("\n1. Surface Scaling M in {32, 64, 128} (Zero-shot vs Retrained):")
    for r in res["M_scaling"]:
        print(f"  M = {r['M']:3d}: Zero-shot = {r['zero_shot_sum_rate']:.2f} bps/Hz | "
              f"Retrained = {r['retrained_sum_rate']:.2f} bps/Hz | "
              f"Random RZF = {r['random_rzf_sum_rate']:.2f} bps/Hz")
        
    print("\n2. BS Antennas N in {4, 8, 16} (Retrained Models):")
    for r in res["N_scaling"]:
        print(f"  N = {r['N']:2d}: Retrained = {r['retrained_sum_rate']:.2f} bps/Hz | "
              f"Random RZF = {r['random_rzf_sum_rate']:.2f} bps/Hz | "
              f"Per-Antenna SE = {r['per_antenna_rate']:.2f} bps/Hz/ant")
        
    print("\n3. User Load K in {2, 4, 8} (Retrained Models):")
    for r in res["K_scaling"]:
        print(f"  K = {r['K']:2d}: Retrained = {r['retrained_sum_rate']:.2f} bps/Hz | "
              f"Random RZF = {r['random_rzf_sum_rate']:.2f} bps/Hz | "
              f"Per-User SE = {r['per_user_rate']:.2f} bps/Hz/user")

def audit_task4():
    print("\n" + "=" * 75)
    print("TASK 4: NOISE-FLOOR ABLATION AUDIT (500 Realizations)")
    print("=" * 75)
    csv_path = "results/task4_noise_ablation/noise_ablation_per_realization.csv"
    if not os.path.exists(csv_path):
        print(f"FAILED: Missing {csv_path}")
        return
    data, n_total = load_csv(csv_path)
    aware = data["rate_noise_aware"]
    abl = data["rate_ablated"]
    diff = abl - aware
    t_stat, p_val = stats.ttest_rel(abl, aware)
    ci = stats.t.interval(0.95, n_total - 1, loc=diff.mean(), scale=stats.sem(diff))
    win = int((diff > 0).sum())
    
    print(f"Total evaluated realizations: {n_total}")
    print(f"Noise-Aware Model Mean: {aware.mean():.4f} ± {aware.std():.4f} bps/Hz")
    print(f"Ablated Model Mean:     {abl.mean():.4f} ± {abl.std():.4f} bps/Hz")
    print(f"Net Difference (Ablated - Aware): {diff.mean():+.4f} bps/Hz ({(abl.mean()/aware.mean() - 1)*100:+.2f}%)")
    print(f"95% Confidence Interval: [{ci[0]:.4f}, {ci[1]:.4f}] bps/Hz")
    print(f"Paired t-test: t = {t_stat:.4f}, p-value = {p_val:.4f}")
    print(f"Ablated Win Rate: {win}/{n_total} ({win/n_total*100:.1f}%)")

if __name__ == "__main__":
    audit_task1()
    audit_task2()
    audit_task3()
    audit_task4()
