# Publication Change Log: Scientific Integrity Audit & Mathematical Repairs

This document records all scientific corrections, methodological repairs, provenance reconciliations, and claim adjustments made to the Intelligent Reflecting Surface (IRS) Adaptive Beamforming project.

---

## 1. Mathematical Repair of IRS Phase Conjugation in Baseline RZF

### Defect Identified:
In the original baseline evaluation scripts (`matlab/run_eval_range.m`, `gnn/train_and_eval_robustness.py`, `gnn/train_and_eval_generalization.py`, `gnn/train_beamforming.py`, and `results/process_phase3_results.py`), the effective channel used to construct the active Regularized Zero-Forcing (RZF) precoder was formulated as:
$$\hat{H}_{\text{eff}}[k, :] = \boldsymbol{\theta}^T \hat{C}_k$$
However, the true physical channel evaluation used:
$$H_{\text{all\_true}}[k, :] = h_k^H \operatorname{diag}(\boldsymbol{\theta}^*) G = (\boldsymbol{\theta}^*)^T C_k$$
Because $\boldsymbol{\theta} \neq \boldsymbol{\theta}^*$ for non-real phase shifts, the active precoder $W_{\text{RZF}}$ was optimized for the complex conjugate of the physical channel. This inverted multi-user spatial nulling and artificially depressed the Random RZF baseline to $\approx 1.93\text{--}2.22$ bps/Hz across all noise regimes.

### Corrective Action:
1. Enforced strict mathematical conjugation consistency across all codebase files:
   - `gnn/train_and_eval_robustness.py` (line 142): `torch.einsum("bm,bkmn->bkn", th_r.conj(), ch_i)`
   - `gnn/train_and_eval_generalization.py` (line 70): `torch.einsum("bm,bkmn->bkn", th_r.conj(), ch_i)`
   - `gnn/train_beamforming.py` (line 61): `torch.einsum("bm,bkmn->bkn", theta_rand.conj(), c_hat_ls)`
   - `matlab/run_eval_range.m` (line 140): `H_eff_rand(k, :) = conj(theta_rand).' * C_hat{k};`
   - `results/process_phase3_results.py` (line 52): `H_eff[k, :] = np.conj(theta_rand) @ C_k`
2. Added unit tests in `tests/test_deep_beamforming.py` confirming that `conj(theta)^T * C_k` strictly matches $H_{\text{all\_true}}$ and that mismatched effective channels have $\approx 100\%$ relative error.

### Empirical Consequence & Physical Insight:
- When evaluated with matching conjugation, Random RZF achieves **30.48 bps/Hz at −100 dBm** because the Base Station has $N=8$ antennas serving $K=4$ users (excess active degrees of freedom $N/K = 2$). At high pilot SNR, low estimation error allows the BS to null multi-user interference cleanly even with random IRS reflections.
- As pilot noise increases (−90 to −70 dBm), classical matrix inversion $(H_{\text{eff}} H_{\text{eff}}^H + \alpha I)^{-1}$ inverts estimation noise, collapsing Random RZF to **4.94 bps/Hz** and classical AO to **3.76 bps/Hz**.
- DeepBeamformingNet maintains **19.36 to 22.51 bps/Hz** across all noise floors, averaging **21.45 bps/Hz** and winning **383/500 (76.6%)** of paired trials against the corrected baseline.

---

## 2. Removal of Unsupported 9-Point Sweep Table in `README.md`

### Defect Identified:
`README.md` lines 69–80 previously displayed a 9-noise sweep table with 50 trials per noise floor (summing to 450 trials, with 331 wins), while claiming an overall win rate of "81.6% (408/500)". No generating script, raw CSV, or MATLAB artifact existed for this 9-point configuration.

### Corrective Action:
- Deleted the unsupported 9-point table from `README.md`.
- Replaced it with the verified 500-sample independent evaluation dataset (5 noise tiers $\times$ 100 trials: $-100, -90, -85, -80, -70$ dBm) directly matching `results/phase3_independent_summary.csv` and `PROJECT_DOCUMENTATION.md` Table 14.2.
- Included the corrected Matched Random RZF baseline alongside Full AO.

---

## 3. Latency Benchmark Provenance & Rigorous Calibration

### Defect Identified:
`README.md` previously claimed a "53.6 s vs 1.15 ms (~46,600×)" speedup without hardware measurement provenance or error bounds.

### Corrective Action:
- Benchmarked both methods repeatedly on identical hardware (Intel Core CPU, 16 GB RAM):
  - Classical Alternating Optimization (MATLAB CVX / SeDuMi): $10.49 \pm 4.24\text{ s}$ per realization.
  - Deep Beamforming Inference (`DeepBeamformingNet` PyTorch CPU): $2.73 \pm 4.47\text{ ms}$ per realization.
- Calculated exact acceleration metrics:
  - Algorithmic Optimization Speedup: **$3,846.2\times$**.
  - End-to-End Speedup (including $44.89\text{ ms}$ multi-phase LS estimation): **$221.3\times$** ($47.62\text{ ms}$ vs. $10.54\text{ s}$).
- Updated all documentation and manuscripts to report these calibrated, reproducible timings.

---

## 4. Reconciliation of Table 15.3 Empirical Standard Deviations

### Defect Identified:
In `PROJECT_DOCUMENTATION.md` Table 15.3, standard deviations for $-100$, $-90$, and $-85$ dBm noise floors were manually entered as incrementing values (`2.85, 2.86, 2.87, 2.86, 2.88`) rather than the true sample statistics.

### Corrective Action:
- Recomputed all sample standard deviations ($s$, $\text{ddof}=1$) directly from `results/phase5_quantization/phase5_quantization_summary.csv`.
- Overwrote Table 15.3 with exact values (e.g., 1-bit QAT std at $-100$ dBm is $2.60$, not $2.87$; 2-bit QAT std is $3.01$, not $2.88$).

---

## 5. Noise-Floor Feature Ablation & Scientific Thesis Realignment

### Defect Identified:
The original project title and claims asserted that the model was "Noise-Aware" and that feeding the scalar noise floor $\sigma_{\text{dBm}}$ provided a distinct performance advantage. However, the controlled Task 4 ablation demonstrated:
- Noise-Aware Model: $21.39 \pm 3.73$ bps/Hz
- Ablated Model (No Noise Feature): $21.42 \pm 3.69$ bps/Hz
- Net Difference: $-0.03$ bps/Hz (95% CI: $[-0.13, +0.07]$, paired $t$-test $p = 0.5818$)

### Corrective Action:
- Retained the null hypothesis: feeding the scalar noise floor provides no statistically detectable benefit under tested conditions.
- Realigned the paper's title and central scientific claim from **"Noise-Aware"** to **"Noise-Resilient"**.
- Provided the verified physical rationale: the multi-phase LS channel estimate $\hat{C}_{\text{LS}} = C + E$ already embeds estimation noise variance in its residuals, rendering the scalar feature redundant. The true contribution is unsupervised direct sum-rate optimization that avoids noise-sensitive matrix inversion.

---

## 6. Hardware Circuit Power Dissipation Claims

### Defect Identified:
Prior text made general claims implying measured PIN diode energy and power savings from IRS quantization.

### Corrective Action:
- Clarified that circuit power dissipation benefits are discussed on theoretical and literature grounds (e.g., Wu & Zhang 2019, Huang et al. 2020); no physical circuit power consumption was measured on hardware testbeds.
- Confined all empirical power discussions to RF transmit power $P_{\max} = 10\text{ W}$ and algorithmic computational latency.
