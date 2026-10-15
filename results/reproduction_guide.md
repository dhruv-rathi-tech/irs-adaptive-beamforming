# Master Reproduction Guide: Noise-Resilient IRS Beamforming

This guide provides explicit, end-to-end commands to reproduce every unit test, benchmark, ablation study, and confirmatory validation experiment reported in the paper.

---

## 1. Environment & Dependencies

### Prerequisites
- Python 3.10+ (tested on Python 3.12.8 on Windows 64-bit)
- PyTorch 2.6.0+ (CPU or CUDA)
- NumPy, SciPy, Pandas, Matplotlib

### Setup
```bash
# 1. Activate the virtual environment
.\venv\Scripts\activate

# 2. Set PYTHONPATH to repository root (PowerShell)
$env:PYTHONPATH="."

# 3. Verify dependency versions
python -c "import torch, numpy, scipy, pandas, matplotlib; print('PyTorch:', torch.__version__, '| NumPy:', numpy.__version__, '| Pandas:', pandas.__version__)"
```

---

## 2. Unit Test Suite (Physical Sanity, Conjugation, and Quantization)

Run all unit test suites to verify physical consistency, IRS phase conjugation, unit-modulus constraints, and power normalization:

```bash
# Run physical channel and pilot orthogonality tests
python -X utf8 -m tests.test_phase1

# Run GNN component unit tests
python -X utf8 -m tests.test_phase3

# Run 1-bit and 2-bit quantization and STE gradient flow tests
python -X utf8 -m tests.test_quantization

# Run deep beamforming pipeline, strict power bounds, and conjugation consistency tests
python -X utf8 -m tests.test_deep_beamforming
```
*Expected Outcome:* All 4 test suites pass with code 0 (`ALL TESTS PASSED`).

---

## 3. Confirmatory Validation (500 Fresh Realizations, Seeds 500,000–500,499)

Runs the frozen `results/gnn/best_beamforming_model.pt` checkpoint on 500 untouched channel realizations across 5 noise tiers (−100, −90, −85, −80, −70 dBm), computing matched Random RZF, bootstrap confidence intervals, and publication figures:

```bash
python -m results.run_confirmatory_validation
```
*Outputs Generated:*
- `results/confirmatory_per_realization.csv`: Per-trial rates for ML and matched RZF.
- `results/confirmatory_validation_summary.csv`: Aggregated means, standard deviations, win rates, and bootstrap 95% CIs.
- `results/fig_confirmatory_sumrate_comparison.png`: Sum-rate vs. noise floor plot with uncertainty bands.
- `results/fig_confirmatory_paired_differences.png`: Paired difference bar chart with bootstrap error bars.

---

## 4. Phase 5 Quantization Studies (1-bit & 2-bit QAT vs PTQ)

To reproduce the discrete phase quantization experiments:

```bash
# Evaluate 1-bit and 2-bit discrete phase configurations across 500 test channels
python -m results.evaluate_phase5_quantization
```
*Outputs Generated:*
- `results/phase5_quantization/phase5_quantization_per_realization.csv`
- `results/phase5_quantization/phase5_quantization_summary.csv`
- `results/phase5_quantization/fig_quantization_comparison.png`

---

## 5. Channel Robustness & Fading Distribution Study (Task 2)

Evaluates zero-shot transfer against retrained models across 5 fading distributions (Rician $K \in \{0, 2, 10, 20\}$ and path-loss $\alpha \in \{2.0, 2.2, 2.8\}$) using the corrected matched Random RZF baseline:

```bash
python -m gnn.train_and_eval_robustness
```
*Outputs Generated:*
- `results/task2_robustness/robustness_summary.csv`
- `results/task2_robustness/fig_channel_robustness_comparison.png`

---

## 6. Dimension Generalization Study (Task 3)

Evaluates surface scaling ($M \in \{32, 64, 128\}$), BS antenna scaling ($N \in \{4, 8, 16\}$), and user load ($K \in \{2, 4, 8\}$) using the corrected baseline:

```bash
python -m gnn.train_and_eval_generalization
```
*Outputs Generated:*
- `results/task3_generalization/dimension_generalization_summary.json`
- `results/task3_generalization/fig_generalization_M_elements.png`
- `results/task3_generalization/fig_generalization_NK_dimensions.png`

---

## 7. Noise-Floor Feature Ablation Study (Task 4)

Verifies the statistical equivalence ($p = 0.5818$) between the noise-aware model and the noise-ablated model:

```bash
python -m gnn.train_and_eval_noise_ablation
```
*Outputs Generated:*
- `results/task4_noise_ablation/noise_ablation_summary.csv`
- `results/task4_noise_ablation/noise_ablation_per_realization.csv`
- `results/task4_noise_ablation/fig_noise_ablation_comparison.png`

---

## 8. Latency Benchmark

Runs the full, reproducible latency benchmark measuring pure PyTorch model forward pass, multi-phase LS estimation, RZF precoder inversion, and recorded MATLAB CVX AO optimization:

```bash
python -m results.benchmark_runtime
```
*Outputs Generated:*
- `results/benchmark_runtime_summary.json`
- `results/benchmark_runtime_report.md`

---

## 9. Regenerating Corrected Phase 3 Baseline (500 Realizations)

Directly recomputes the corrected matched Random-IRS + RZF baseline across all 500 original Phase 3 channel realizations:

```bash
python -m results.generate_phase3_corrected_baselines
```
*Outputs Generated:*
- `results/phase3_independent_per_realization_corrected.csv`
- `results/phase3_independent_summary_corrected.csv`


---

## 10. Expanded 15-Point Pilot Noise-Floor Sweep under Fixed Data-Channel Noise (1,500 Realizations)

Executes the fine-grained pilot estimation noise sweep across 15 values (`-100` to `-60` dBm) under fixed data-channel noise ($\sigma_{\text{data}}^2 \approx -85\text{ dBm}$):

```bash
python -m results.run_noise_floor_expansion
```
*Outputs Generated:*
- `results/noise_floor_expanded/noise_floor_expanded_per_realization.csv`: 1,500 row per-trial results.
- `results/noise_floor_expanded/noise_floor_expanded_summary.csv`: Aggregated means, sample stds, 95% CIs.
- `results/noise_floor_expanded/noise_floor_expanded_comparisons.csv`: Paired tests, win rates, Bonferroni-adjusted p-values.
- `results/noise_floor_expanded/fig_noise_floor_expanded_curves.png`: 15-point sum-rate curve with uncertainty bands.
- `results/noise_floor_expanded/fig_noise_floor_expanded_gain.png`: Net gain bar chart showing crossover.
- `results/noise_floor_expanded/EXPANDED_EXPERIMENT_REPORT.md`: Comprehensive experimental report.
