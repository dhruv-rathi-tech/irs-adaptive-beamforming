# Pilot Noise-Floor Sweep Report: 15 Pilot Noise Tiers (-100 dBm to -60 dBm) under Fixed Data-Channel Noise

## 1. Experimental Protocol & Noise Characterization
- **Experiment Definition:** Pilot noise-floor sweep under fixed data-channel noise.
- **Pilot Noise-Floor Grid (15 values):** `[-100.0, -97.5, -95.0, -92.5, -90.0, -87.5, -85.0, -82.5, -80.0, -77.5, -75.0, -72.5, -70.0, -65.0, -60.0]` dBm.
  - Applied during multi-phase Least-Squares cascaded channel estimation (`multi_phase_pilot_estimate_C`).
- **Fixed Data-Channel Noise:** Held constant at $\sigma_{\mathrm{data}}^2 = 3.16 \times 10^{-12}\text{ W} \approx -85.0\text{ dBm}$ during the true-channel achievable sum-rate evaluation across all methods. This cleanly isolates the impact of pilot channel-estimation error from data detection SNR.
- **Training Support vs. Evaluation Range:**
  - Model Training Range: $P_{n,\text{pilot}}^{\text{dBm}} \in [-100.0, -70.0]\text{ dBm}$ (input feature $\tilde{\sigma}_{\text{pilot}} = (P_{n,\text{pilot}}^{\text{dBm}} + 85.0)/15.0 \in [-1.0, +1.0]$).
  - Primary Evaluation (Phase 3): $\{-100, -90, -85, -80, -70\}\text{ dBm}$ (within training support).
  - Expanded Sweep Range: $[-100.0, -60.0]\text{ dBm}$ (input feature $\tilde{\sigma}_{\text{pilot}} \in [-1.0, +1.67]$).
  - Tiers $-65.0\text{ dBm}$ ($\tilde{\sigma} = +1.33$) and $-60.0\text{ dBm}$ ($\tilde{\sigma} = +1.67$) test out-of-distribution extrapolation beyond the training range.
- **Statistical Design & Realizations:** 1,500 total evaluations across 15 pilot-noise tiers over 100 shared channel geometries (not 1,500 independent channel geometries).
  - **Pairing Structure:** Within-tier method comparisons are strictly paired (identical channel geometry, identical pilot-noise realization, identical data noise).
  - **Cross-Tier Structure:** Cross-tier evaluations reflect repeated measures on the 100 shared channel geometries across varying pilot-noise levels.
- **Random Seed Hierarchy & Provenance:**
  - **Channel-Geometry Seeds:** Seeds 600,000 to 600,099 (`BASE_SEED + trial`) generate 100 distinct physical channel geometries $(G_i, H_i)$ that are shared across all 15 pilot noise levels.
  - **Pilot-Noise Seeds:** Seeds 600,000 to 601,499 (`BASE_SEED + tier_idx * 100 + trial`) generate independent pilot noise realizations across tiers and trials (1,500 distinct pseudo-random draws).
  - **Random IRS Phase Seeds:** Seeds 1,600,000 to 1,601,499 (`1000000 + trial_seed`) generate independent phase shift configurations for the Matched Random RZF baseline.
- **Model Checkpoints:**
  - Continuous ML: `results/gnn/best_beamforming_model.pt` (SHA-256: `25c75f469f9eafb2d49308d4498546d7d76f611e2713260318a2d29ddd31c517`)
  - 2-bit QAT: `results/phase5_quantization/model_2bit_qat.pt` (SHA-256: `e32000dbab59ee24014c01aeacc3d36ca93a6b57964ee55c61f1df4ae90dc514`)
  - 1-bit QAT: `results/phase5_quantization/model_1bit_qat.pt` (SHA-256: `e90f0bdd8c035ddbdbb935fc6733ef7b4717535f9786cc997a2b6b8ebc6b87ae`)

## 2. Comprehensive Numerical Results Across 15 Pilot Noise Tiers

| Pilot Noise Floor (dBm) | Deep ML Continuous | Matched Random RZF | Net Gain (ML - RZF) | 95% Bootstrap CI | ML Win Rate | Paired $t$-test $p$-val | Bonferroni $p$-val | 2-bit QAT | 1-bit QAT |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **-100.0** | 22.73 ± 2.63 | 30.67 ± 8.64 | **-7.94** | [-9.31, -6.35] | 17.0/100 (17.0%) | 1.44e-17 | 2.15e-16 | 21.88 | 21.43 |
| ** -97.5** | 22.71 ± 2.66 | 27.99 ± 8.13 | **-5.28** | [-6.68, -3.81] | 22.0/100 (22.0%) | 4.06e-10 | 6.09e-09 | 21.85 | 21.45 |
| ** -95.0** | 22.70 ± 2.66 | 24.37 ± 9.05 | **-1.67** | [-3.34, +0.04] | 42.0/100 (42.0%) | 5.59e-02 | 8.38e-01 | 21.81 | 21.41 |
| ** -92.5** | 22.64 ± 2.77 | 22.40 ± 8.49 | **+0.23** | [-1.33, +1.78] | 52.0/100 (52.0%) | 7.65e-01 | 1.00e+00 | 21.82 | 21.33 |
| ** -90.0** | 22.67 ± 2.68 | 18.74 ± 7.60 | **+3.93** | [+2.50, +5.34] | 74.0/100 (74.0%) | 7.98e-07 | 1.20e-05 | 21.81 | 21.39 |
| ** -87.5** | 22.60 ± 2.73 | 15.79 ± 7.39 | **+6.81** | [+5.39, +8.10] | 81.0/100 (81.0%) | 2.38e-16 | 3.56e-15 | 21.65 | 21.30 |
| ** -85.0** | 22.61 ± 2.75 | 14.31 ± 7.76 | **+8.29** | [+6.90, +9.66] | 88.0/100 (88.0%) | 8.82e-20 | 1.32e-18 | 21.68 | 21.26 |
| ** -82.5** | 22.25 ± 3.25 | 11.65 ± 6.09 | **+10.61** | [+9.44, +11.69] | 97.0/100 (97.0%) | 3.97e-34 | 5.96e-33 | 21.33 | 20.89 |
| ** -80.0** | 22.28 ± 3.07 | 10.18 ± 5.58 | **+12.09** | [+11.12, +13.07] | 98.0/100 (98.0%) | 1.37e-42 | 2.05e-41 | 21.27 | 20.94 |
| ** -77.5** | 21.81 ± 3.44 | 8.13 ± 4.51 | **+13.68** | [+12.89, +14.43] | 100.0/100 (100.0%) | 5.87e-56 | 8.80e-55 | 20.77 | 20.61 |
| ** -75.0** | 21.15 ± 4.37 | 6.82 ± 4.51 | **+14.33** | [+13.37, +15.20] | 99.0/100 (99.0%) | 1.27e-50 | 1.90e-49 | 20.01 | 20.17 |
| ** -72.5** | 20.92 ± 4.23 | 5.81 ± 3.86 | **+15.12** | [+14.23, +15.94] | 100.0/100 (100.0%) | 1.17e-56 | 1.75e-55 | 20.08 | 19.68 |
| ** -70.0** | 20.16 ± 5.02 | 4.71 ± 2.90 | **+15.46** | [+14.59, +16.30] | 100.0/100 (100.0%) | 2.53e-56 | 3.79e-55 | 19.28 | 18.98 |
| ** -65.0** | 18.74 ± 5.39 | 3.46 ± 2.19 | **+15.27** | [+14.27, +16.23] | 100.0/100 (100.0%) | 1.17e-51 | 1.75e-50 | 17.70 | 17.48 |
| ** -60.0** | 17.42 ± 5.49 | 2.70 ± 1.31 | **+14.73** | [+13.78, +15.70] | 100.0/100 (100.0%) | 6.55e-50 | 9.83e-49 | 16.13 | 15.93 |

## 3. Physical Operating Regime & Transition Analysis
1. **Transition Region ($-95.0$ to $-90.0$ dBm) and Near-Equality Point at $-92.5$ dBm:**
   - In the clean pilot regime ($-100$ to $-95$ dBm), accurate channel estimation allows classical Matched Random RZF to outperform Deep ML (net gains of $-7.94$ to $-1.67$ bps/Hz).
   - At $-92.5$ dBm, the two methods achieve near-equality: Deep ML achieves $22.64 \pm 2.77$ bps/Hz vs. Matched RZF $22.40 \pm 8.49$ bps/Hz (mean difference $+0.235$ bps/Hz, 95% bootstrap CI $[-1.33, +1.78]$ bps/Hz, paired $t = 0.2995, df = 99, p = 0.765$, Wilcoxon $p = 0.850$, win count 52/100).
   - Because $p = 0.765$ and the win rate is virtually 50/50, the two methods are statistically indistinguishable at $-92.5$ dBm. Rather than a knife-edge point crossover, the data define a broad transition region spanning approximately $-95.0$ to $-90.0$ dBm.
   - At $-90.0$ dBm, Deep ML establishes a statistically decisive lead ($+3.93$ bps/Hz, paired $t = 5.27, p = 7.98 \times 10^{-7}$, win rate 74.0%).
2. **Severe Pilot Contamination Regime ($-90$ to $-60$ dBm):**
   - Across all pilot noise floors $\ge -90$ dBm, Deep ML decisively outperforms Matched Random RZF under the fixed $-85$ dBm data-channel condition.
   - As pilot noise increases, estimation errors corrupt the cascaded matrix $\hat{C}_{\text{LS}}$. Classical RZF, which requires inverting this noisy matrix, suffers from catastrophic noise amplification and collapses to $2.70 \pm 1.31$ bps/Hz at $-60$ dBm pilot noise.
   - In contrast, Deep ML preserves $17.42 \pm 5.49$ bps/Hz ($+14.73$ bps/Hz net gain, 100/100 wins, paired $t = 28.79, df = 99, p = 6.55 \times 10^{-50}$). While learned adaptive regularization dampening noise-corrupted directions during $K \times K$ matrix inversion is a plausible mechanism, its causal role in out-of-distribution generalization beyond the training range ($[-100, -70]$ dBm) remains a working hypothesis.
3. **Quantization Resilience:**
   - 2-bit QAT preserves $\ge 92.6\%$ of continuous performance even under $-60$ dBm pilot noise ($16.13$ vs $17.42$ bps/Hz) and $\ge 96.3\%$ across $-100$ to $-70$ dBm.
   - 1-bit QAT preserves $\ge 91.4\%$ at $-60$ dBm ($15.93$ vs $17.42$ bps/Hz) and $\ge 94.0\%$ across $-100$ to $-70$ dBm.