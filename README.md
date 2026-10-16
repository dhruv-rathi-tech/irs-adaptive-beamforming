# Noise-Resilient Intelligent Reflecting Surface (IRS) Adaptive Beamforming

An end-to-end research, simulation, and benchmark framework for Intelligent Reflecting Surface (IRS) assisted multi-user MISO wireless communication systems under practical, noisy channel state information (CSI).

This repository implements the complete pipeline from physical channel modeling and multi-phase pilot estimation to classical Alternating Optimization (AO) with Semidefinite Relaxation (SDR) and end-to-end unsupervised **Deep Beamforming (`DeepBeamformingNet`)**. The deep beamformer directly maps noisy cascaded channel estimates to joint active transmit precoders and passive IRS reflection phase shifts, achieving decisive sum-rate gains in moderate-to-severe noise regimes with a **4,430.1x algorithmic optimization speedup** over numerical solvers.

---

## Documentation Links

- **Research Manuscript:** [`results/manuscript_draft.md`](results/manuscript_draft.md)
- **Comprehensive Project Documentation:** [`PROJECT_DOCUMENTATION.md`](PROJECT_DOCUMENTATION.md) (Full system architecture, derivations, statistical methodology, claims matrix, and reproduction guide)
- **Cryptographic Experiment Manifest:** [`results/experiment_manifest.json`](results/experiment_manifest.json) (SHA-256 hashes of all checkpoints, datasets, and reports)

---

## Key Highlights & Scientific Findings

- **Decoupled Cascaded Channel Architecture:** Eliminates high-dimensional active RF receive chains at the IRS by estimating the cascaded user-IRS-BS link matrix $C_k \in \mathbb{C}^{M \times N}$ via multi-phase least-squares (LS) estimation ($Q = M = 64$ DFT pilot phases, $\tau = 16$ symbols).
- **Physics-Grounded Propagation:** Rician fading for BS-IRS link ($K_{\text{Rician}} = 10$), Rayleigh fading for IRS-User scattering links, 3D geometric positioning, and distance-dependent path loss with directional antenna gains ($19.6\text{ dB}$).
- **Direct Unsupervised Deep Beamforming:** `DeepBeamformingNet` features element-permutation-equivariant `ElementBlock` layers that map noisy LS estimates $\hat{C}_{\text{LS}}$ directly to passive phases $\boldsymbol{\theta}_{\text{ML}}$, power allocations $\mathbf{p}$, and learned regularizations $\alpha$, synthesizing active precoders in closed form via regularized Gram matrix inversion:
  - **Production Model (`use_noise_floor=True`):** Input dimension $d_{\text{in}} = 66$, **42,439** parameters.
  - **Noise-Ablated Model (`use_noise_floor=False`):** Input dimension $d_{\text{in}} = 65$, **42,375** parameters.
- **Large-Scale Independent Verification (500 Held-Out Realizations):**
  - **Overall Average Sum-Rate:** Deep ML achieves **$21.80 \pm 3.38\text{ bps/Hz}$**, outperforming classical AO (**$13.01 \pm 10.31\text{ bps/Hz}$**) by **+8.79 bps/Hz** (win rate **80.0% (400/500)**, $p = 1.48 \times 10^{-65}$) and matched Random RZF (**$15.69 \pm 10.83\text{ bps/Hz}$**) by **+6.11 bps/Hz** (win rate **73.4% (367/500)**, $p = 9.13 \times 10^{-35}$).
  - **Severe Pilot Noise Dominance (−70 dBm):** In severe pilot noise, classical AO collapses to $3.76\text{ bps/Hz}$ and matched RZF to $4.88\text{ bps/Hz}$, while Deep ML maintains **$20.14\text{ bps/Hz}$** (gains up to **+16.38 bps/Hz**, 100/100 wins).
  - **Operating Regime Boundaries Identified:** Classical AO ($27.40\text{ bps/Hz}$) and matched RZF ($30.34\text{ bps/Hz}$) win at high SNR (−100 dBm) due to surplus BS antennas ($N=8 > K=4$). In pure isotropic Rayleigh fading ($K_{\text{rice}}=0$), matched RZF wins ($24.72\text{ vs }17.22\text{ bps/Hz}$) because rich scattering provides full channel rank.
- **Quantization-Aware Training (QAT):** 2-bit discrete phase QAT retains **99.5%** of continuous sum-rate ($21.28\text{ vs }21.39\text{ bps/Hz}$), and 1-bit QAT retains **95.6%** ($20.46\text{ bps/Hz}$).
- **Measured Latency Acceleration (Standard Workstation CPU):** Optimization execution time drops from **10.49 s** (MATLAB CVX AO) to **2.37 ms** (PyTorch CPU), an algorithmic speedup of **4,430.1x** (**223.5x** end-to-end acceleration including multi-phase LS channel estimation).
- **Noise Feature Redundancy (Negative Finding):** Controlled ablation proves scalar noise floor input is statistically redundant ($p = 0.5818$), confirming the network regularizes directly on channel estimation residuals.

---

## System Model

```
                    +-----------------------------+
                    |  Intelligent Reflecting     |
                    |  Surface (IRS, M = 64)      |
                    +-----------------------------+
                               ^         \
       BS-IRS Link (G)         |          \  IRS-User Link (h_r,k)
       Rician Fading (K=10)    |           \ Rayleigh Fading
                               |            v
    +-----------------------------+      +-------------------+
    | Base Station (BS, N = 8)    |      | User k (K = 4)    |
    | Transmit Power P_max = 10 W |      | Single antenna    |
    +-----------------------------+      +-------------------+
             Direct Links Blocked by Urban Obstacles
```

### Network Configuration
- **Base Station (BS):** $N = 8$ transmit antennas configured as a Uniform Linear Array (ULA).
- **IRS:** $M = 64$ passive reflecting elements configured as an $8 \times 8$ Uniform Planar Array (UPA).
- **Users:** $K = 4$ single-antenna receivers randomly distributed at distances $d_{\text{IU}} \in [5, 30]\text{ m}$.
- **Transmit Power:** $P_{\max} = 10\text{ W}$ ($40\text{ dBm}$) total across all users.

---

## Primary Performance Summary (500 Independent Realizations)

Evaluation conducted across 500 independent test channel realizations (seeds `100,000`–`100,499`, 100 trials per noise floor across 5 noise tiers) using strictly matched channel and noise instances:

| Pilot Noise Level | Full AO (CVX) [bps/Hz] | Matched Random RZF [bps/Hz] | DeepBeamformingNet [bps/Hz] | Net Gain vs. AO | Net Gain vs. RZF | ML Win Rate vs. RZF |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **−100 dBm** | $27.40 \pm 9.04$ | $\mathbf{30.34 \pm 5.48}$ | $22.58 \pm 2.45$ | −4.82 bps/Hz | −7.76 bps/Hz | 18% (18/100) |
| **−90 dBm** | $18.52 \pm 6.13$ | $21.57 \pm 4.39$ | $\mathbf{22.42 \pm 2.37}$ | +3.90 bps/Hz | +0.85 bps/Hz | 62% (62/100) |
| **−85 dBm** | $11.08 \pm 3.65$ | $14.12 \pm 3.52$ | $\mathbf{22.17 \pm 2.76}$ | +11.09 bps/Hz | +8.05 bps/Hz | 99% (99/100) |
| **−80 dBm** | $4.29 \pm 1.47$ | $7.55 \pm 2.12$ | $\mathbf{21.70 \pm 3.25}$ | +17.41 bps/Hz | +14.15 bps/Hz | 100% (100/100) |
| **−70 dBm** | $3.76 \pm 2.08$ | $4.88 \pm 2.12$ | $\mathbf{20.14 \pm 4.29}$ | +16.38 bps/Hz | +15.26 bps/Hz | 100% (100/100) |
| **Overall Mean** | **13.01 ± 10.31** | **15.69 ± 10.83** | **21.80 ± 3.38** | **+8.79 bps/Hz** | **+6.11 bps/Hz** | **73.4% (367/500)** |

### Paired Statistical Significance
- **Gain vs. Full AO:** $+8.79\text{ bps/Hz}$ ($t = 19.95, df = 499, p = 1.48 \times 10^{-65}$; win rate $80.0\%$, $400/500$).
- **Gain vs. Matched RZF:** $+6.11\text{ bps/Hz}$ ($t = 13.30, df = 499, p = 9.13 \times 10^{-35}$; 95% CI: $[+5.19, +6.98]$; win rate $73.4\%$, $367/500$).

### Computational Latency Profile (Workstation CPU)
- **Classical AO (CVX / SeDuMi):** $10.49 \pm 4.24\text{ s}$ per realization.
- **Deep Beamforming Inference (`DeepBeamformingNet` CPU):** $2.37 \pm 0.77\text{ ms}$ (median: $2.25\text{ ms}$).
- **Active RZF Precoder Inversion (PyTorch):** $0.45 \pm 0.18\text{ ms}$.
- **Multi-Phase LS Channel Estimation ($Q=64, \tau=16$):** $44.78 \pm 5.32\text{ ms}$.
- **Algorithmic Optimization Acceleration:** **4,430.1x**.
- **End-to-End Latency Acceleration:** **223.5x** ($47.15\text{ ms}$ vs. $10.54\text{ s}$).

---

## Repository Structure

```
irs_project/
|-- README.md                         # Project overview and quickstart guide
|-- PROJECT_DOCUMENTATION.md          # Master documentation & technical reference
|-- LICENSE                           # MIT License
|-- config/
|   `-- system_config.py              # Central simulation parameters
|-- simulation/
|   |-- channel.py                    # Rician BS-IRS & Rayleigh IRS-User generator
|   |-- pilot.py                      # Orthogonal pilot generation & noise modeling
|   `-- multi_phase_estimation.py     # Multi-phase LS cascaded channel estimation
|-- gnn/
|   |-- beamforming_model.py          # DeepBeamformingNet architecture (42,439 params)
|   |-- beamforming_dataset.py        # Dataset loader for complex channel tensors
|   |-- quantization.py               # 1-bit & 2-bit STE quantization layers
|   |-- train_beamforming.py          # Unsupervised training pipeline
|   |-- train_quantization.py         # Quantization-aware training pipeline
|   |-- train_and_eval_robustness.py  # Fading robustness study (Task 2)
|   |-- train_and_eval_generalization.py # Dimension scaling study (Task 3)
|   `-- train_and_eval_noise_ablation.py # Noise-floor feature ablation (Task 4)
|-- matlab/
|   |-- Opt_func_perfectCSI.m         # Classical AO baseline (Perfect CSI)
|   |-- Opt_func_Ck.m                 # Classical AO baseline (Estimated CSI)
|   |-- run_eval_range.m              # Parallel batch worker for Monte Carlo evaluation
|   `-- run_phase3_full_eval.m        # Phase 3 MATLAB evaluation pipeline
|-- tests/
|   |-- test_phase1.py                # Channel physics & pilot orthogonality checks
|   |-- test_phase3.py                # GNN module & dataset checks
|   |-- test_quantization.py          # 1-bit & 2-bit STE quantization checks
|   `-- test_deep_beamforming.py      # Precoder bounds & phase-conjugation checks
`-- results/
    |-- manuscript_draft.md           # Research manuscript draft
    |-- experiment_manifest.json      # Cryptographic SHA-256 hash manifest
    |-- gnn/best_beamforming_model.pt # Frozen optimal DeepBeamformingNet checkpoint
    |-- confirmatory_validation_summary.csv # Confirmatory validation results (500 virgin channels)
    |-- phase3_independent_summary_corrected.csv # Phase 3 corrected baseline results
    |-- benchmark_runtime_summary.json# Standardized latency measurements
    `-- noise_floor_expanded/         # Expanded 15-point pilot noise sweep (1,500 evaluations)
```

---

## Installation & Setup

### 1. Python Environment Setup
```bash
# Clone the repository
git clone https://github.com/dhruv-rathi-tech/irs-adaptive-beamforming.git
cd irs-adaptive-beamforming

# Create and activate virtual environment
python -m venv venv
.\venv\Scripts\activate       # Windows PowerShell
# source venv/bin/activate    # Linux / macOS

# Install dependencies
pip install torch numpy scipy pandas matplotlib
```

### 2. Verify Installation with Test Suite
Run the automated test suite covering physical channel modeling, GNN layers, STE quantization, and strict beamforming power constraints:
```bash
python -X utf8 -m tests.test_phase1
python -X utf8 -m tests.test_phase3
python -X utf8 -m tests.test_quantization
python -X utf8 -m tests.test_deep_beamforming
```
*Expected Outcome:* All 4 test modules pass with exit code 0 (48 passed checks across four test modules: Phase 1: 21 checks, Phase 3: 12 checks, Quantization: 5 checks, Deep Beamforming: 10 checks).

---

## Reproduction Quickstart

To reproduce the core experimental findings:

```bash
# 1. Run confirmatory out-of-sample validation on 500 virgin channel realizations
python -m results.run_confirmatory_validation

# 2. Evaluate 1-bit and 2-bit discrete phase quantization (Task 1)
python -m results.evaluate_phase5_quantization

# 3. Evaluate channel robustness across 5 fading distributions (Task 2)
python -m gnn.train_and_eval_robustness

# 4. Evaluate dimension generalization across M, N, K (Task 3)
python -m gnn.train_and_eval_generalization

# 5. Evaluate noise-floor feature ablation (Task 4)
python -m gnn.train_and_eval_noise_ablation

# 6. Run reproducible computational latency profiling
python -m results.benchmark_runtime

# 7. Run expanded 15-point pilot noise sweep (1,500 evaluations)
python -m results.run_noise_floor_expansion
```

For complete instructions, dataset generation scripts, and MATLAB CVX baseline details, see [`PROJECT_DOCUMENTATION.md`](PROJECT_DOCUMENTATION.md).
