# Intelligent Reflecting Surface (IRS) Assisted Adaptive Beamforming

## Overview

This repository provides a simulation and evaluation framework for Intelligent Reflecting Surface (IRS) assisted multi-user Multiple-Input Single-Output (MISO) wireless communication systems. The project focuses on practical transmission conditions where channel state information (CSI) must be acquired from noisy pilot observations rather than assumed to be perfectly known.

The codebase implements physics-grounded channel models, an orthogonal pilot signaling framework, a multi-phase cascaded channel estimator, and classical Alternating Optimization (AO) beamforming algorithms using Semidefinite Relaxation (SDR). Performance is benchmarked by evaluating both ideal channel conditions (Perfect-CSI) and realistic conditions (Estimated-CSI evaluated on true channels).

---

## System Model

### Network Configuration
- **Base Station (BS):** Equipped with a Uniform Linear Array (ULA) of $N$ antennas.
- **IRS:** Planar/linear array of $M$ passive reflecting elements with phase shifts $\theta = [e^{j\theta_1}, \dots, e^{j\theta_M}]^T$, satisfying $|\theta_m| = 1$.
- **Users:** $K$ single-antenna mobile stations distributed in the service area.

### Channel Modeling
- **BS-to-IRS Link ($G \in \mathbb{C}^{M \times N}$):** Modeled using Rician fading ($K_{\text{Rician}} = 10$) to capture the line-of-sight (LoS) dominant link resulting from engineered IRS deployment.
- **IRS-to-User Link ($H \in \mathbb{C}^{M \times K}$):** Modeled using Rayleigh fading to reflect multi-path scattering environments.
- **Path Loss:** Distance-dependent log-distance path loss with distinct exponents:
  - BS-to-IRS exponent: $\alpha_{\text{BI}} = 2.2$
  - IRS-to-User exponent: $\alpha_{\text{IU}} = 3.5$
  - Element and antenna directional power gains calibrated at $19.6\text{ dB}$.

### Cascaded Channel Formulation
Rather than separately estimating the high-dimensional individual links $G$ and $H$ (which requires active RF chains at the IRS or ill-conditioned matrix division), the system formulates the effective channel for user $k$ as:

$$h_{\text{eff}, k} = h_k^H \text{diag}(\theta) G = \theta^T C_k$$

where $C_k = \text{diag}(h_k^H) G \in \mathbb{C}^{M \times N}$ is the cascaded channel matrix. 

During the pilot phase, the IRS cycles through $Q \ge M$ linearly independent reflection configurations. Each user observes noisy pilot sequences across $N$ orthogonal pilot vectors, and the cascaded matrix $\hat{C}_k$ is solved via batched Least Squares (LS).

---

## Implementation Status

### Completed and Validated
- **Physics-Based Channel Simulation:** Generation of BS-IRS Rician and IRS-User Rayleigh channels with distance-dependent path loss and directional antenna gains (`simulation/channel.py`).
- **Pilot Signaling & Noise Modeling:** Orthogonal pilot matrices via QR decomposition, absolute receiver thermal noise floor modeling, and emergent received SNR calculation (`simulation/pilot.py`).
- **Multi-Phase Cascaded Channel Estimation:** Least-squares recovery of cascaded matrices $C_k$ across configurable pilot configurations (`simulation/multi_phase_estimation.py`).
- **Unit & Sanity Testing:** Automated suite validating channel dimensions, matrix orthogonality, path-loss scaling, and noise-floor consistency (`tests/test_phase1.py`).
- **Classical Alternating Optimization (AO) Baselines:**
  - `matlab/Opt_func_perfectCSI.m`: Reference benchmark assuming perfect instantaneous CSI (theoretical upper bound).
  - `matlab/Opt_func_Ck.m`: Realistic benchmark using estimated cascaded channel $\hat{C}_k$ for joint active transmit beamforming and passive IRS phase optimization.
- **Realistic Performance Evaluation:** True achievable sum rate computed by evaluating the beamformers derived from $\hat{C}_k$ against the true channel realizations ($H_{\text{true}}, G_{\text{true}}$).
- **Sweep & Analysis Workflows:**
  - Cascaded channel Normalized Mean Square Error (NMSE) versus received SNR (`results/generate_nmse_vs_snr.py`).
  - Achievable sum-rate baseline comparison over SNR sweeps (`matlab/run_snr_sweep.m`).
  - Effective Spectral Efficiency versus pilot overhead accounting for coherence block length penalty: $\text{SE}_{\text{eff}} = \left(1 - \frac{\tau_{\text{total}}}{T_{\text{coherence}}}\right) R_{\text{true}}$ (`matlab/run_tau_sweep.m`).

### Planned Work (Not Yet Implemented)
- **Data-Driven Beamforming (Phase 3):** Implementation of neural network models (e.g., Graph Neural Networks / deep learning beamformers) for direct mapping from noisy pilot observations to beamforming configurations.

---

## Repository Structure

```
irs_project/
|-- config/
|   `-- system_config.py            # Central simulation and system parameters
|-- simulation/
|   |-- channel.py                  # BS-IRS and IRS-User channel generators
|   |-- pilot.py                    # Orthogonal pilot generation & noisy reception
|   `-- multi_phase_estimation.py   # Multi-phase LS cascaded channel estimation
|-- tests/
|   `-- test_phase1.py              # Channel and pilot verification tests
|-- matlab/
|   |-- Opt_func_perfectCSI.m       # AO baseline with perfect CSI
|   |-- Opt_func_Ck.m               # AO baseline with estimated cascaded CSI
|   |-- Bisection.m                 # Bisection search subroutine for transmit power
|   |-- export_channels.py          # Exports Python channel data to .mat format
|   |-- run_phase2.m                # Phase 2 classical baseline test runner
|   |-- run_snr_sweep.m             # Multi-realization SNR sweep runner
|   |-- run_tau_sweep.m             # Pilot length and overhead sweep runner
|   `-- verify_Ck_against_reference.m # Numerical equivalence verification
`-- results/
    |-- generate_nmse_vs_snr.py     # NMSE vs SNR Monte Carlo data generator
    |-- plot_nmse_vs_snr.py         # Generates NMSE vs SNR plots
    |-- export_channels_snr_sweep.py# Exports SNR sweep test vectors for MATLAB
    `-- export_channels_tau_sweep.py# Exports pilot sweep test vectors for MATLAB
```

---

## System Parameters

Default system configurations defined in `config/system_config.py`:

| Parameter | Symbol | Default Value | Description |
|---|---|---|---|
| BS Antennas | $N$ | 8 | Uniform linear array elements at BS |
| IRS Elements | $M$ | 64 | Reflective phase shift elements |
| Users | $K$ | 4 | Single-antenna ground receivers |
| BS-IRS Distance | $d_{\text{BI}}$ | 50.0 m | Fixed separation distance |
| IRS-User Distance | $d_{\text{IU}}$ | 5.0 to 30.0 m | Uniformly distributed user range |
| BS-IRS Path Loss Exponent | $\alpha_{\text{BI}}$ | 2.2 | Near-LoS propagation environment |
| IRS-User Path Loss Exponent | $\alpha_{\text{IU}}$ | 3.5 | Non-LoS scattering environment |
| Rician Factor (BS-IRS) | $K_{\text{Rician}}$ | 10.0 | LoS-to-multipath ratio (linear scale) |
| Reference Path Loss | $C_0$ | -30.0 dB | Path loss at $d_0 = 1.0\text{ m}$ |
| Element Directional Gain | - | 19.6 dB | Combined antenna/element gain per hop |
| Pilot Length | $\tau$ | 16 symbols | Pilot length per phase pattern |
| Coherence Block | $T_{\text{coherence}}$ | 200 symbols | Total symbols per channel coherence block |
| BS Transmit Power Budget | $P_{\max}$ | 10.0 W (40 dBm) | Maximum linear transmit power |
| Noise Floor | $\sigma^2$ | -85.0 dBm | Receiver thermal noise power |

---

## Prerequisites and Setup

### Python Environment
- Python 3.9 or newer
- Required libraries:
  - `numpy`
  - `scipy`
  - `matplotlib`

Install dependencies via pip:
```bash
pip install numpy scipy matplotlib
```

### MATLAB Environment
- MATLAB R2020a or later
- Optimization Toolbox
- [CVX](http://cvxr.com/cvx/): Convex programming package for MATLAB (configured with SeDuMi or SDPT3 solver). Ensure `cvx_setup` has been executed in your MATLAB session.

---

## Usage Instructions

### 1. Verify Simulation Pipeline
Run the Phase 1 automated sanity test suite:
```bash
python -m tests.test_phase1
```

### 2. Generate Cascaded Channel NMSE Curves
Simulate channel estimation accuracy over a range of noise floors and plot the resulting NMSE versus received SNR:
```bash
python results/generate_nmse_vs_snr.py
python results/plot_nmse_vs_snr.py
```
Output plot will be saved to `results/fig1_nmse_vs_snr.png`.

### 3. Run Classical AO Baselines (MATLAB)

#### Single-Point Baseline Comparison:
Generate test channel data in Python:
```bash
python -m matlab.export_channels
```
In MATLAB, execute:
```matlab
run('matlab/run_phase2.m')
run('matlab/plot_phase2_comparison.m')
```

#### SNR Sweep Evaluation:
Export multi-realization channel matrices across SNR points:
```bash
python results/export_channels_snr_sweep.py
```
In MATLAB, execute the optimization sweep:
```matlab
run('matlab/run_snr_sweep.m')
```
Results and baseline plots are saved to `results/snr_sweep_results.mat` and `results/fig_sumrate_baseline.png`.

#### Pilot Overhead Sweep Evaluation:
Export channel matrices across pilot lengths ($\tau \in \{8, 16, 32, 64\}$):
```bash
python results/export_channels_tau_sweep.py
```
In MATLAB, compute the effective spectral efficiency:
```matlab
run('matlab/run_tau_sweep.m')
```
Results and curves are saved to `results/tau_sweep_results.mat` and `results/fig3_effSE_vs_overhead.png`.