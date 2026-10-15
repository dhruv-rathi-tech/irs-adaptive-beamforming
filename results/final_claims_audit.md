# Final Claims Audit: Scientific Truth & Verification Matrix

This audit evaluates every major scientific and empirical claim in the repository against direct evidence, reproducible scripts, raw per-realization outputs, and established communications literature.

Each claim is assigned one of the following formal statuses:
- **VERIFIED:** Supported by reproducible code, raw per-realization artifacts, and valid statistics.
- **SUPPORTED WITH LIMITATIONS:** Empirically verified, but holds only within specific stated regimes or assumptions.
- **INCONCLUSIVE:** Evidence is mixed or statistically insufficient.
- **UNSUPPORTED:** Claim has no underlying experimental provenance or execution trace.
- **CONTRADICTED:** Experimental evidence directly refutes the claim.
- **NOT TESTED:** Experiment was never executed.

---

## 1. Audit Summary Matrix

| ID | Exact Scientific Claim | Status | Supporting Artifact | P-value / Confidence Interval | Documented Limitations / Required Action |
|:---:|:---|:---:|:---|:---|:---|
| **C1** | **Overall ML vs. Classical AO Superiority:** `DeepBeamformingNet` achieves higher average multi-user sum-rate than CVX Alternating Optimization across the multi-noise operating sweep. | **SUPPORTED WITH LIMITATIONS** | `results/phase3_independent_summary.csv` (seeds 100k–100.5k) | Mean diff: $+8.44$ bps/Hz, 95% CI: $[+7.59, +9.30]$, $p = 1.06 \times 10^{-62}$, Win rate: 81.6% (408/500) | Holds decisively in moderate-to-severe noise (−90 to −70 dBm). At high SNR (−100 dBm), classical AO wins (27.40 vs 22.51 bps/Hz). Claim must state operating regime. |
| **C2** | **High-SNR Boundary at −100 dBm:** Classical methods outperform direct ML at high pilot SNR. | **VERIFIED** | `results/phase3_independent_summary.csv`, `results/confirmatory_validation_summary.csv` | AO: $27.40$ bps/Hz, RZF: $30.48$ bps/Hz vs. ML: $22.51$ bps/Hz ($p = 2.32 \times 10^{-7}$) | Physical reason: Base station possesses excess antennas ($N=8 > K=4$). Low pilot estimation error enables sharp classical multi-user zero-forcing. |
| **C3** | **Severe-Noise Dominance (−70 dBm):** Direct ML achieves dramatic gains over classical methods under severe pilot observation noise. | **VERIFIED** | `results/phase3_independent_summary.csv`, `results/confirmatory_validation_summary.csv` | Gain over AO: $+15.60$ bps/Hz (+415.3%), Win rate: 100/100, $p = 9.64 \times 10^{-56}$; Gain over RZF: $+14.42$ bps/Hz | Classical matrix inversion $(H_{\text{eff}} H_{\text{eff}}^H + \alpha I)^{-1}$ inverts estimation noise, destroying zero-forcing. Direct ML avoids matrix inversion. |
| **C4** | **Performance vs. Matched Random RZF:** `DeepBeamformingNet` outperforms an active RZF precoder operating with random IRS phase shifts. | **SUPPORTED WITH LIMITATIONS** | `results/phase3_independent_per_realization.csv`, `results/confirmatory_validation_summary.csv` | Phase 3: ML 21.45 vs RZF 15.63 bps/Hz (76.6% win rate); Confirmatory: ML 21.43 vs RZF 15.92 bps/Hz (72.4% win rate, $p = 1.51 \times 10^{-27}$) | Repaired conjugation defect. ML wins decisively in noise (−90 to −70 dBm, 71–100% win rate), but Random RZF wins at −100 dBm (30.48 vs 22.51 bps/Hz). |
| **C5** | **Computational Latency Speedup:** Deep beamforming accelerates optimization from seconds to milliseconds. | **VERIFIED** | `results/benchmark_runtime.py`, `results/phase3_independent_per_realization.csv` | CPU forward pass: $2.73 \pm 4.47$ ms vs. CVX AO: $10.49 \pm 4.24$ s. Algorithmic speedup: $3,846.2\times$ | End-to-end latency including multi-phase LS estimation ($44.89$ ms) is $47.62$ ms vs $10.54$ s ($221.3\times$ speedup). Unsupported claims of 46,600× purged. |
| **C6** | **Scalar Noise-Floor Input Benefit:** Explicitly inputting the noise floor $\sigma_{\text{dBm}}$ improves beamforming performance. | **CONTRADICTED** | `results/task4_noise_ablation/noise_ablation_summary.csv` (seeds 200k–200.5k) | Net difference: $-0.03$ bps/Hz, 95% CI: $[-0.13, +0.07]$, $p = 0.5818$ (statistically indistinguishable) | Scalar noise is redundant because $\hat{C}_{\text{LS}}$ residuals already embed estimation noise variance. Title and thesis realigned to **"Noise-Resilient"**. |
| **C7** | **1-bit and 2-bit Quantization Retention:** Discrete phase quantization retains near-continuous sum-rate performance. | **VERIFIED** | `results/phase5_quantization/phase5_quantization_summary.csv` (seeds 200k–200.5k) | 2-bit QAT: $21.28 \pm 3.33$ bps/Hz (99.5% of continuous $21.39$); 1-bit QAT: $20.46 \pm 3.26$ bps/Hz (95.6% of continuous) | QAT vs PTQ difference is statistically significant only at severe noise (−70 dBm: $+1.87$ bps/Hz, $p = 0.0035$), not globally. |
| **C8** | **Zero-Shot Transfer Across Fading Distributions:** Model trained on Rician $K=10$ generalizes to alternative channel distributions. | **SUPPORTED WITH LIMITATIONS** | `results/task2_robustness/robustness_summary.csv` (seeds 300k–300.1k) | Achieves $17.00$ to $23.11$ bps/Hz across all settings. Wins 83–96% in Rician LoS ($K=10, 20$) and high path loss ($\alpha=2.8$). | In pure Rayleigh fading ($K=0$), matched Random RZF outperforms Zero-Shot ML ($24.72$ vs $17.22$ bps/Hz) because isotropic scattering gives full spatial rank. |
| **C9** | **Zero-Shot Dimension Generalization to Arbitrary M:** Permutation-equivariant ElementBlock scales to different IRS element counts without retraining. | **VERIFIED** | `results/task3_generalization/dimension_generalization_summary.json` (seeds 410k–410.1k) | Zero-shot: $M=32 \to 19.36$, $M=64 \to 21.46$, $M=128 \to 23.87$ bps/Hz. Parameter count strictly independent of $M$. | True zero-shot applies strictly to $M$. Variations in BS antennas $N$ or users $K$ require model re-projection. Computational complexity is $\mathcal{O}(M)$, not $\mathcal{O}(\log M)$. |
| **C10** | **Hardware Power Savings & PIN Diode Dissipation:** Project proves measured hardware circuit power reduction. | **UNSUPPORTED / THEORETICAL ONLY** | Literature models (Wu & Zhang 2019, Huang et al. 2020) | No hardware testbed measurements conducted. Transmit power is constrained to $P_{\max}=10$ W. | Must be labeled strictly as theoretical microwave circuit motivation from literature. All empirical power claims restricted to RF transmit power. |
| **C11** | **Out-of-Sample Independent Confirmation:** Model findings hold on completely fresh, unexamined channel realizations. | **VERIFIED** | `results/confirmatory_validation_summary.csv` (seeds 500k–500.5k) | Confirmatory ML mean: $21.43$ bps/Hz vs. Phase 3: $21.45$ bps/Hz. Net gain over RZF: $+5.51$ bps/Hz, 95% CI: $[+4.56, +6.39]$, $p = 1.51 \times 10^{-27}$ | Evaluated on 500 virgin test channels with zero test leakage. Reproduces Phase 3 findings within 0.02 bps/Hz. |

---

## 2. Definitive Status of Core Research Manuscript

Based on the audit matrix:
- **No fabricated, cherry-picked, or manipulated results exist in the repository.**
- **The IRS phase-conjugation baseline bug has been mathematically repaired and verified across all code.**
- **All aggregate tables and figures are programmatically generated directly from saved per-realization CSV files.**
- **The scientific claim has been shifted from "Noise-Aware" to "Noise-Resilient" in strict accordance with the ablation findings.**
- **The high-SNR boundary (−100 dBm) and pure Rayleigh boundary ($K=0$) are transparently documented as physical limitations of deep optimization.**
