# Intelligent Reflecting Surface (IRS) Assisted Adaptive Beamforming

[![GitHub Repository](https://img.shields.io/badge/GitHub-Repository-blue?logo=github)](https://github.com/dhruv-rathi-tech/irs-adaptive-beamforming)
[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-blue.svg?logo=python)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg?logo=pytorch)](https://pytorch.org/)
[![MATLAB](https://img.shields.io/badge/MATLAB-R2020a%2B-orange.svg?logo=mathworks)](https://www.mathworks.com/products/matlab.html)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An end-to-end research, simulation, and benchmark framework for Intelligent Reflecting Surface (IRS) assisted multi-user MISO wireless communication systems under practical, noisy channel conditions.

This repository implements the complete pipeline from physical channel modeling and multi-phase pilot estimation to classical Alternating Optimization (AO) with Semidefinite Relaxation (SDR) and end-to-end unsupervised **Deep Beamforming (`DeepBeamformingNet`)**. The deep beamformer directly maps noisy cascaded channel estimates to joint active transmit precoders and passive IRS reflection phase shifts, achieving dramatic sum-rate gains in moderate-to-severe noise regimes and a **~46,000× latency speedup** suitable for real-time deployment.

---

## Key Highlights & Findings

- **Decoupled Cascaded Channel Architecture:** Eliminates high-dimensional active RF chain requirements at the IRS by estimating the cascaded user-IRS-BS link matrix $C_k \in \mathbb{C}^{M \times N}$ via multi-phase least-squares (LS) estimation ($Q = M = 64$ pilot phases, $\tau = 16$ symbols).
- **Physics-Grounded Propagation:** Rician fading for BS-IRS link ($K_{\text{Rician}} = 10$), Rayleigh fading for IRS-User scattering links, 3D geometric positioning, and distance-dependent path loss with directional antenna gains ($19.6\text{ dB}$).
- **Classical Optimization Baselines:** MATLAB CVX implementation of Alternating Optimization (AO) with Semidefinite Relaxation (SDR) and Gaussian randomization under both Perfect-CSI (theoretical upper bound) and Estimated-CSI evaluated on true channels.
- **End-to-End Deep Beamforming:** Complex MLP network trained with negative sum-rate loss directly on true physical channels, learning noise-robust beam patterns that overcome residual LS channel estimation errors.
- **Large-Scale Independent Verification (500 Held-Out Realizations):**
  - **Overall Paired Win Rate:** **81.6% (408 / 500 realizations)** won by Deep Beamforming over classical AO.
  - **Moderate-to-Severe Noise Regime ($-85$ to $-70\text{ dBm}$):** **99.3% (298 / 300 realizations)** win rate with paired gains up to **+15.53 bps/Hz (+500%)**.
  - **Inference Speedup:** Average execution time drops from **53.6 s** (MATLAB CVX AO) to **1.15 ms** (PyTorch CPU), representing a **$46,600\times$ speedup** well within standard channel coherence intervals ($T_c \approx 5\text{--}10\text{ ms}$).

---

## System Model

```
                    +-----------------------------+
                    |  Intelligent Reflecting     |
                    |  Surface (IRS, M = 64)      |
                    +-----------------------------+
                               ^         \
       BS-IRS Link (G)         |          \  IRS-User Link (H_k)
       Rician Fading (K=10)    |           \ Rayleigh Fading
                               |            v
    +-----------------------------+      +-------------------+
    | Base Station (BS, N = 8)   |      | User k (K = 4)    |
    | Transmit Power P_max = 10 W |      | Single antenna    |
    +-----------------------------+      +-------------------+
```

### Network Configuration
- **Base Station (BS):** $N = 8$ transmit antennas configured as a Uniform Linear Array (ULA).
- **IRS:** $M = 64$ passive reflecting elements with phase shifts $\boldsymbol{\theta} = [e^{j\theta_1}, \dots, e^{j\theta_M}]^T$, where $|\theta_m| = 1$.
- **Users:** $K = 4$ single-antenna receivers randomly distributed at distances $d_{\text{IU}} \in [5, 30]\text{ m}$.

### Cascaded Channel Formulation
The effective channel for user $k$ under IRS phase vector $\boldsymbol{\theta}$ is:
$$h_{\text{eff}, k} = h_k^H \text{diag}(\boldsymbol{\theta}) G = \boldsymbol{\theta}^T C_k$$
where $C_k = \text{diag}(h_k^H) G \in \mathbb{C}^{M \times N}$ is the cascaded channel matrix.

### Received Signal & Achievable Sum-Rate
The received signal at user $k$ is:
$$y_k = \boldsymbol{\theta}^T C_k w_k s_k + \sum_{j \neq k} \boldsymbol{\theta}^T C_k w_j s_j + n_k, \quad n_k \sim \mathcal{CN}(0, \sigma^2)$$

The true-channel achievable sum-rate is:
$$R_{\text{sum}}(W, \boldsymbol{\theta}) = \sum_{k=1}^K \log_2 \left(1 + \frac{|\boldsymbol{\theta}^T C_k w_k|^2}{\sum_{j \neq k} |\boldsymbol{\theta}^T C_k w_j|^2 + \sigma^2}\right)$$
subject to $\sum_{k=1}^K \|w_k\|^2 \le P_{\max}$ and $|\theta_m| = 1, \forall m$.

---

## Performance Summary (500 Independent Realizations)

Comprehensive evaluation conducted across 500 completely independent test channel realizations (seeds `100000`–`100499`) evaluated at identical physical channel instances:

| Noise Floor ($\sigma^2$) | Received SNR | Full AO Sum-Rate | Deep Beamforming | Absolute Gain | Relative Gain | ML Win Rate | $p$-value ($t$-test) |
|---|---|---|---|---|---|---|---|
| **-70.0 dBm** | -16.4 dB | 3.11 bps/Hz | **18.64 bps/Hz** | **+15.53 bps/Hz** | **+500.0%** | **100% (50/50)** | $< 10^{-15}$ |
| **-75.0 dBm** | -11.4 dB | 7.64 bps/Hz | **21.20 bps/Hz** | **+13.56 bps/Hz** | **+177.5%** | **100% (50/50)** | $< 10^{-15}$ |
| **-77.5 dBm** | -8.9 dB | 11.83 bps/Hz | **24.22 bps/Hz** | **+12.39 bps/Hz** | **+104.7%** | **100% (50/50)** | $< 10^{-15}$ |
| **-80.0 dBm** | -6.4 dB | 14.86 bps/Hz | **24.89 bps/Hz** | **+10.03 bps/Hz** | **+67.5%** | **98.0% (49/50)** | $< 10^{-15}$ |
| **-85.0 dBm** | -1.4 dB | 18.12 bps/Hz | **25.15 bps/Hz** | **+7.03 bps/Hz** | **+38.8%** | **100% (50/50)** | $< 10^{-15}$ |
| **-90.0 dBm** | +3.6 dB | 22.86 bps/Hz | **24.78 bps/Hz** | **+1.92 bps/Hz** | **+8.4%** | **78.0% (39/50)** | $2.3 \times 10^{-5}$ |
| **-92.5 dBm** | +6.1 dB | **24.36 bps/Hz** | 24.30 bps/Hz | -0.06 bps/Hz | -0.2% | 46.0% (23/50) | $0.85$ (stat. tied) |
| **-95.0 dBm** | +8.6 dB | **25.99 bps/Hz** | 24.46 bps/Hz | -1.53 bps/Hz | -5.9% | 28.0% (14/50) | $4.1 \times 10^{-5}$ |
| **-100.0 dBm** | +13.6 dB | **29.74 bps/Hz** | 24.84 bps/Hz | -4.90 bps/Hz | -16.5% | 12.0% (6/50) | $< 10^{-15}$ |
| **Overall** | — | 16.57 bps/Hz | **23.57 bps/Hz** | **+7.00 bps/Hz** | **+42.2%** | **81.6% (408/500)** | $< 10^{-15}$ |

### Computation Latency Benchmark
- **Classical AO (CVX / SeDuMi):** $53.6\text{ s}$ average per channel realization.
- **Deep Beamforming (`DeepBeamformingNet` CPU):** $1.15\text{ ms}$ average per channel realization.
- **Speedup Factor:** **$46,600\times$**, enabling sub-millisecond real-time beam adaptation.

---

## Repository Structure

```
irs_project/
|-- README.md                         # Top-level project documentation
|-- PROJECT_DOCUMENTATION.md          # Comprehensive technical and mathematical audit
|-- config/
|   `-- system_config.py              # Central wireless simulation parameters
|-- simulation/
|   |-- channel.py                    # Rician BS-IRS & Rayleigh IRS-User channel generator
|   |-- pilot.py                      # Orthogonal pilot generation & noise floor modeling
|   `-- multi_phase_estimation.py     # Multi-phase LS cascaded channel estimation
|-- gnn/
|   |-- beamforming_model.py          # DeepBeamformingNet architecture & complex MLP layers
|   |-- beamforming_dataset.py        # Dataset loader for complex channel tensors
|   |-- train_beamforming.py          # Unsupervised training pipeline (sum-rate maximization)
|   |-- model.py                      # Preliminary GNN estimation model (Phase 3 archive)
|   |-- dataset.py                    # Graph dataset builder (Phase 3 archive)
|   `-- evaluate.py                   # Model evaluation utilities
|-- matlab/
|   |-- Opt_func_perfectCSI.m         # AO baseline with perfect CSI (upper bound)
|   |-- Opt_func_Ck.m                 # AO baseline with estimated cascaded CSI
|   |-- Bisection.m                   # Optimal transmit power allocation bisection search
|   |-- compute_sum_rate.m            # True-channel achievable sum rate calculation
|   |-- export_channels.py            # Channel matrix export to MATLAB .mat format
|   |-- run_phase2.m                  # Phase 2 classical baseline single-point runner
|   |-- run_snr_sweep.m               # Multi-realization SNR sweep runner
|   |-- run_tau_sweep.m               # Pilot overhead sweep runner
|   |-- run_eval_range.m              # Parallel batch worker for Monte Carlo evaluation
|   `-- run_phase3_full_eval.m        # Phase 3 MATLAB evaluation pipeline
|-- tests/
|   |-- test_phase1.py                # Unit tests for channel physics and pilot orthogonality
|   |-- test_phase3.py                # Unit tests for GNN components
|   `-- test_deep_beamforming.py      # Unit tests for DeepBeamformingNet pipeline
`-- results/
    |-- gnn/                          # Saved models, training history, and loss curves
    |   |-- best_beamforming_model.pt # Frozen optimal DeepBeamformingNet checkpoint
    |   `-- beamforming_training_history.json
    |-- fig_phase3_sumrate_vs_snr.png # Achievable sum-rate comparison across noise floors
    |-- fig_phase3_gain_vs_snr.png    # Absolute & percentage gains over classical AO
    |-- fig_phase3_independent_cdf.png# Empirical CDF distribution of sum rates
    |-- fig_phase3_runtime_comparison.png # Latency profile (AO vs Deep Beamforming)
    |-- phase3_independent_summary.csv# Summary statistics across 500 held-out realizations
    |-- phase3_independent_per_realization.csv # Granular per-trial comparative results
    |-- generate_independent_500_data.py   # Test dataset generator (seeds 100000-100499)
    `-- process_independent_500_results.py  # Statistical testing, tables, and figure plotting
```

---

## System Parameters

Default system configurations from `config/system_config.py`:

| Parameter | Symbol | Default Value | Description |
|---|---|---|---|
| BS Antennas | $N$ | 8 | Uniform Linear Array elements at Base Station |
| IRS Elements | $M$ | 64 | Passive reflecting elements with discrete/continuous phase |
| Users | $K$ | 4 | Single-antenna ground mobile stations |
| BS-IRS Distance | $d_{\text{BI}}$ | 50.0 m | Fixed separation distance |
| IRS-User Distance | $d_{\text{IU}}$ | 5.0 to 30.0 m | Uniformly distributed user range |
| BS-IRS Path Loss Exponent | $\alpha_{\text{BI}}$ | 2.2 | Near line-of-sight propagation environment |
| IRS-User Path Loss Exponent | $\alpha_{\text{IU}}$ | 3.5 | Non line-of-sight multi-path scattering |
| Rician Factor (BS-IRS) | $K_{\text{Rician}}$ | 10.0 (10 dB) | LoS-to-multipath linear power ratio |
| Reference Path Loss | $C_0$ | -30.0 dB | Path loss at reference distance $d_0 = 1.0\text{ m}$ |
| Element Directional Gain | $G_{\text{ant}}$ | 19.6 dB | Combined antenna and reflection directional gain |
| Pilot Length | $\tau$ | 16 symbols | Pilot sequence length per IRS configuration |
| Pilot Configurations | $Q$ | 64 | Number of linearly independent reflection patterns |
| Coherence Block | $T_{\text{coherence}}$ | 200 symbols | Coherence interval in symbol durations |
| BS Max Transmit Power | $P_{\max}$ | 10.0 W (40 dBm) | Linear transmit power budget |
| Noise Floor Sweep | $\sigma^2$ | -100 to -65 dBm | Thermal noise range across evaluation scenarios |

---

## Installation & Prerequisites

### Python Environment
- Python 3.9+
- Recommended virtual environment:
```bash
python -m venv venv
# Windows:
.\venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

pip install numpy scipy matplotlib torch pandas
```

### MATLAB Environment
- MATLAB R2020a or later
- Optimization Toolbox
- [CVX](http://cvxr.com/cvx/): Convex programming package for MATLAB (configured with SeDuMi or SDPT3 solver). Run `cvx_setup` before running MATLAB routines.

---

## Reproduction & Usage Guide

### 1. Verify Simulation Core (Phase 1)
Run the automated test suite verifying channel dimensions, matrix orthogonality, path-loss scaling, and LS estimation:
```bash
python -m tests.test_phase1
```

### 2. Classical AO Baselines (Phase 2)
Export synthetic channels and run the classical Alternating Optimization algorithm in MATLAB:
```bash
# Export test instances
python -m matlab.export_channels

# In MATLAB:
# run('matlab/run_phase2.m')
```

### 3. Deep Beamforming Pipeline (Phase 3)
Verify neural network modules and test forward inference:
```bash
python -m tests.test_deep_beamforming
```

To retrain `DeepBeamformingNet` from scratch:
```bash
python -m gnn.train_beamforming
```

### 4. Full 500-Instance Independent Monte Carlo Evaluation
To reproduce the 500-realization independent verification study:

```bash
# Step 1: Generate the independent 500-instance dataset (seeds 100000-100499)
python -m results.generate_independent_500_data

# Step 2: Run MATLAB AO optimization across batches (or via single runner)
matlab -batch "addpath('matlab'); run_eval_range(1, 125, 1)"
matlab -batch "addpath('matlab'); run_eval_range(126, 250, 2)"
matlab -batch "addpath('matlab'); run_eval_range(251, 375, 3)"
matlab -batch "addpath('matlab'); run_eval_range(376, 500, 4)"

# Step 3: Compute paired statistical tests, summary tables, and publication plots
python -m results.process_independent_500_results
```

All plots will be updated in the `results/` folder:
- `results/fig_phase3_sumrate_vs_snr.png`
- `results/fig_phase3_gain_vs_snr.png`
- `results/fig_phase3_independent_cdf.png`
- `results/fig_phase3_runtime_comparison.png`

---

## Review-III Research Extensions (Completed)

### 1. Phase 5 — 1-bit & 2-bit Quantization-Aware Beamforming (Task 1)
Addresses hardware non-idealities by constraining IRS phase shifts to discrete phase alphabets:
* **1-bit IRS:** $\{+1, -1\}$ (phase choices $\{0, \pi\}$).
* **2-bit IRS:** $\{+1, +j, -1, -j\}$ (phase choices $\{0, \pi/2, \pi, 3\pi/2\}$).
* **Quantization-Aware Training (QAT):** Implemented via a differentiable Straight-Through Estimator (STE) in `gnn/quantization.py`.
* **Results (500 Held-Out Realizations, Seeds 200,000–200,499):**
  - **Continuous Baseline:** **21.39 bps/Hz**
  - **2-bit QAT:** **21.28 bps/Hz** (Retains **99.5%** of continuous performance, with only a $-0.11\text{ bps/Hz}$ degradation).
  - **2-bit PTQ:** **21.16 bps/Hz** (QAT recovers $+0.12\text{ bps/Hz}$ over post-training quantization).
  - **1-bit QAT:** **20.46 bps/Hz** (Retains **95.6%** of continuous performance with binary phase shifters).
  - **1-bit PTQ:** **20.45 bps/Hz**
  - **Inference Latency:** $\approx 1.94\text{--}2.05\text{ ms}$ on CPU.

Reproduce:
```bash
python -m gnn.train_quantization --bits 1 --epochs 35
python -m gnn.train_quantization --bits 2 --epochs 35
python -m results.evaluate_phase5_quantization
```

### 2. Robustness Across Varied Fading Environments (Task 2)
Evaluated across 5 physical propagation regimes:
* Default ($K_{\text{rice}}=10, \alpha=2.2$): Zero-Shot = **21.70 bps/Hz** | Retrained = **21.88 bps/Hz**
* Weak LOS ($K_{\text{rice}}=2.0, \alpha=2.2$): Zero-Shot = **20.58 bps/Hz** | Retrained = 17.81 bps/Hz
* Pure Rayleigh ($K_{\text{rice}}=0.0$, complete LOS blockage): Zero-Shot = **17.22 bps/Hz** (vs 1.87 bps/Hz baseline)
* Strong LOS ($K_{\text{rice}}=20.0, \alpha=2.0$): Zero-Shot = **23.11 bps/Hz** | Retrained = **23.59 bps/Hz**
* High Path Loss ($\alpha=2.8$): Zero-Shot = **17.00 bps/Hz** | Retrained = **17.21 bps/Hz**
* **Finding:** Zero-shot transfer delivers **17.00 to 23.11 bps/Hz**, outperforming the baseline by 8× to 11× without retraining.

Reproduce:
```bash
python -m gnn.train_and_eval_robustness
```

### 3. Dimension Generalization Across N, M, K (Task 3)
* **IRS Elements Scaling ($M \in \{32, 64, 128\}$):** Permutation-equivariant ElementBlocks allow zero-shot transfer across surface dimensions ($M=32$: 19.36 bps/Hz, $M=64$: 21.46 bps/Hz, $M=128$: 23.87 bps/Hz), matching retrained performance.
* **BS Antenna Scaling ($N \in \{4, 8, 16\}$):** Precoders scale from 21.34 bps/Hz ($N=4$, 5.33 bps/Hz/ant) to 22.37 bps/Hz ($N=8$, 2.80 bps/Hz/ant) and 22.30 bps/Hz ($N=16$, 1.39 bps/Hz/ant).
* **User Load Scaling ($K \in \{2, 4, 8\}$):** Total sum-rate remains constant around $\approx 21.8\text{--}22.0\text{ bps/Hz}$ under total power constraint $P_{\max}=10\text{ W}$, with per-user spectral efficiency dividing equitably ($10.95 \to 5.50 \to 2.72\text{ bps/Hz/user}$).

Reproduce:
```bash
python -m gnn.train_and_eval_generalization
```

### 4. Controlled Noise-Floor Input Feature Ablation (Task 4)
* **Noise-Aware Model:** **21.39 bps/Hz**
* **Ablated Model (Noise Feature Removed):** **21.42 bps/Hz**
* **Net Difference:** $-0.03\text{ bps/Hz}$ ($-0.14\%$), $p_{\text{paired-}t} = 0.5818$, Wilcoxon $p = 0.7446$, 95% CI = $[-0.13, +0.07]\text{ bps/Hz}$.
* **Rigorous Conclusion:** Statistically indistinguishable. $\hat{C}_{\text{LS}}$ already implicitly carries noise variance through estimation residuals, rendering an external scalar noise-floor input redundant.

Reproduce:
```bash
python -m gnn.train_and_eval_noise_ablation
```

---

## Citation & Contact

If you utilize this framework, codebase, or results in your research, please cite:

```bibtex
@misc{irs_adaptive_beamforming_2026,
  author = {Dhruv Rathi},
  title = {Noise-Aware Adaptive Beamforming for IRS-Assisted Wireless Communications},
  year = {2026},
  publisher = {GitHub},
  howpublished = {\url{https://github.com/dhruv-rathi-tech/irs-adaptive-beamforming}}
}
```

For questions or contributions, please open an issue or pull request at [https://github.com/dhruv-rathi-tech/irs-adaptive-beamforming](https://github.com/dhruv-rathi-tech/irs-adaptive-beamforming).