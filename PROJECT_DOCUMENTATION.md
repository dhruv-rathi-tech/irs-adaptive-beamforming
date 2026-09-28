# Noise-Aware Adaptive Beamforming for IRS-Assisted Wireless Communications
## Full Project Documentation & Handoff Guide

**Purpose of this document:** a complete, standalone account of the project —
what it is, what has been built, what was tried and failed (and why), what
works, and exactly what to do next. Written so that someone with *zero prior
context* — a new AI agent, a new teammate, or you in six months — can read
this, open the code, and continue without re-deriving anything from scratch.

**Status as of this document:**
- Phase 1 (Channel & IRS Simulation) — **COMPLETE**.
- Phase 2 (Multi-Phase LS Estimation & AO Baselines) — **COMPLETE**.
- Phase 3 (Noise-Aware Deep Beamforming) — **COMPLETE AND INDEPENDENTLY VERIFIED**.
  - The preliminary GNN channel-estimation (NMSE regression) approach failed to beat LS and is archived as historical work.
  - The project decisively pivoted to end-to-end Deep Beamforming (`DeepBeamformingNet`), which directly optimizes true-channel achievable sum-rate.
  - On the frozen checkpoint `results/gnn/best_beamforming_model.pt`, Deep Beamforming achieves an overall paired gain of **+8.44 bps/Hz (+64.9%)** across 500 independent held-out realizations, winning **81.6% (408/500)** overall and **99.3% (298/300)** in the moderate-to-severe noise regime ($-85$ to $-70$ dBm).
- Phase 4 (Large-Scale Monte Carlo Evaluation) — **COMPLETE** via the 500-realization independent verification study.
- Phase 5 (Quantization-Aware Extension) — **GROUNDWORK READY** (discrete phase rounding implemented in `matlab/Discerete.m`, reserved for future work).

---

## 1. What this project is

**Title:** Noise-Aware Adaptive Beamforming for Intelligent Reflecting
Surface (IRS)-Assisted Wireless Communications.
**Context:** Final-year B.Tech research project, VIT Chennai, School of
Electronics Engineering.

**The problem.** An IRS is a surface of many small passive elements that
each reflect an incoming radio wave with a controllable phase shift. By
choosing those phases well, an IRS can redirect a base station's (BS)
signal around obstacles toward users who wouldn't otherwise get good
coverage. Doing this — "beamforming" through the IRS — requires knowing the
wireless channel (how the signal propagates from BS → IRS → user). In
practice, this channel is never known exactly: it must be *estimated* from
short, noisy "pilot" transmissions, and that estimate degrades as noise
increases or as the pilot sequence is shortened (shorter pilots = less
overhead but worse estimates).

**The research question (Current & Active).**
> **"Given the SAME noisy pilot observations, can an ML/GNN-based beamforming method produce an active and passive configuration $(W, \theta)$ that achieves higher achievable SUM-RATE on the SAME underlying true channel than the conventional LS $\to$ AO pipeline?"**

**Primary Metric:**
* **TRUE-CHANNEL ACHIEVABLE SUM-RATE (bps/Hz)** evaluated on the identical underlying physical channel realizations.

**Secondary Metrics:**
* End-to-end and optimization latency, paired win rates, 95% confidence intervals, and rigorous hypothesis testing (paired $t$-test and Wilcoxon signed-rank test).

**What must NOT be claimed** (per the project's own ground rules):
- "Our method always outperforms classical optimization under perfect CSI."
  *(False: in the near-perfect CSI regime at $-100$ dBm, conventional LS $\to$ Full AO outperforms ML Direct by $4.90$ bps/Hz because it converges against near-exact channel estimates).*
- "Our method wins 100% of all trials across all noise levels."
  *(False: ML Direct wins 408/500 trials (81.6%) overall, and 298/300 trials (99.3%) in the moderate-to-severe noise regime from $-85$ to $-70$ dBm).*
- Any fabricated, un-run, or cherry-picked result.
- Any claim of improvement that isn't backed by an actual, reproducible experiment in this repository.

This document reports real, independently audited numbers only.

---

## 2. High-level system model (the physics/math, in plain terms)

- **BS**: `N = 8` antennas.
- **IRS**: `M = 64` passive reflecting elements, each with phase shift
  `θ_m`, constrained to `|θ_m| = 1` (it can only change phase, not
  amplitude — this is what makes it "passive").
- **Users**: `K = 4` single-antenna users.
- **Two channel links:**
  - `G` (BS→IRS), shape `(M, N)` = `(64, 8)`, complex. Modeled as Rician
    fading (a strong line-of-sight component + scattering) because the
    BS-IRS link is usually a deliberately placed, near-LOS deployment.
  - `H` (IRS→user), shape `(M, K)` = `(64, 4)`, complex, one column per
    user. Modeled as Rayleigh fading (no dominant LOS path — users are
    scattered around, often behind obstacles) — this is exactly the
    "blockage" scenario IRS is meant to solve.
- **Path loss**: standard log-distance model, `PL(d) = C0 · (d/d0)^(−α)`,
  with different exponents for the two link types (BS-IRS more LOS-like,
  IRS-user more NLOS/scattered), plus a fixed antenna/element gain term.
  All of this lives in `config/system_config.py` and `simulation/channel.py`.
- **The cascaded channel.** Instead of trying to separately estimate `G`
  and `H` — which requires dividing by small channel coefficients and
  amplifies noise badly — this project follows the literature and estimates, per user `k`, the
  **cascaded channel**:
  ```
  C_k = diag(conj(h_k)) @ G        shape (M, N), complex
  ```
  This is the single quantity that matters for beamforming: the effective
  channel a user sees under IRS phase configuration `θ` is
  `h_eff_k = θᵀ · C_k` (a simple linear map).
- **Pilot protocol (multi-phase).** To estimate `C_k` (an `M×N` unknown),
  the IRS cycles through `Q = M = 64` different random phase
  configurations. At each configuration, the BS sends `τ = 16` pilot
  symbols; each user's noisy received pilot is used to get a per-phase
  estimate of the effective channel, and stacking all `Q` measurements
  gives a linear system `Θ_stack @ C_k = (per-phase estimates)` that is
  solved by Least Squares.
  $$\text{Total pilot overhead} = Q \cdot \tau = 64 \times 16 = 1024 \text{ pilot symbols per realization.}$$
- **CRITICAL DISTINCTION ON NEURAL NETWORK INPUT:**
  The neural network does **NOT** consume the 1024 raw pilot symbols directly.
  The 1024 pilot-symbol observations are first processed by the multi-phase LS estimator (`multi_phase_pilot_estimate_C`) to produce the cascaded channel estimate $\hat{C}_{\text{LS}} \in \mathbb{C}^{K \times M \times N}$.
  The neural network receives $\hat{C}_{\text{LS}}$ and the scalar noise floor $\sigma_{\text{dBm}}$.
- **Noise.** A *fixed absolute* receiver noise floor in dBm (not a
  transmit-relative SNR) — this matches how real receivers work (thermal
  noise floor + noise figure). The *received* SNR is an **emergent,
  measured** quantity that comes out of this fixed noise floor combined
  with path loss and fading.

---

## 3. Repository layout

```
project/
├── config/
│   └── system_config.py          All tunable parameters (N, M, K, geometry,
│                                  path-loss exponents, noise floor, pilot
│                                  length, seed). Single source of truth.
│
├── simulation/                   PHASE 1 — channel/pilot/noise simulation
│   ├── channel.py                 Generates G (Rician) and H (Rayleigh).
│   ├── pilot.py                   Pilot matrix generation, noise floor
│   │                               conversion (dBm→W), single-phase noisy
│   │                               pilot reception, received-SNR measurement.
│   └── multi_phase_estimation.py  THE key Phase 1/2 function:
│                                   multi_phase_pilot_estimate_C() — runs the
│                                   full Q=64-phase pilot protocol and solves
│                                   for C_hat[k] via LS (1024 pilot symbols).
│
├── matlab/                       PHASE 2 & PHASE 3 EVALUATION — classical AO & runners
│   ├── Opt_func_perfectCSI.m      AO baseline using TRUE H, G (upper bound).
│   ├── Opt_func_Ck.m               AO baseline using ESTIMATED C_hat (realistic baseline).
│   │                               Supports warm-start theta_init and max_iters cap.
│   ├── Bisection.m                  Helper: bisection search for AO power allocation.
│   ├── compute_sum_rate.m          Downlink multi-user sum-rate evaluation utility.
│   ├── run_eval_range.m            Batch multi-worker Monte Carlo evaluation function
│   │                               evaluating all 6 competing pipelines.
│   ├── run_phase3_full_eval.m      Preliminary 50-realization evaluation driver.
│   ├── run_phase2.m                 Single-run Phase 2 evaluation driver.
│   ├── run_snr_sweep.m              Noise-floor sweep driver.
│   ├── run_tau_sweep.m              Pilot-length sweep driver.
│   ├── Discerete.m, DiscreteSum.m   Quantized phase rounding helpers (Phase 5 groundwork).
│   └── verify_Ck_against_reference.m  Numerically verifies Opt_func_Ck.m equivalence.
│
├── gnn/                          PHASE 3 — Neural Beamforming Architecture & Training
│   ├── beamforming_model.py        CURRENT ACTIVE: DeepBeamformingNet architecture
│   │                               (unit-modulus phase head, adaptive RZF head,
│   │                               differentiable true sum-rate loss function).
│   ├── beamforming_dataset.py      CURRENT ACTIVE: PyTorch Dataset generating paired
│   │                               C_hat_LS observations and true channels.
│   ├── train_beamforming.py        CURRENT ACTIVE: End-to-end true sum-rate training script.
│   ├── model.py                    HISTORICAL: NoiseAwareGNN (NMSE regression, failed).
│   ├── dataset.py                  HISTORICAL: Graph dataset for NMSE regression.
│   ├── train.py                    HISTORICAL: NMSE-loss training script.
│   └── evaluate.py                 HISTORICAL: NMSE evaluation script.
│
├── results/                      Generated datasets, checkpoints, and figures
│   ├── gnn/
│   │   ├── best_beamforming_model.pt CURRENT FROZEN CHECKPOINT (DeepBeamformingNet).
│   │   ├── best_model.pt            HISTORICAL CHECKPOINT (NoiseAwareGNN NMSE model).
│   │   └── training_history.json    Training metrics log.
│   ├── phase3_independent_eval_data.mat 500-instance held-out independent test set.
│   ├── phase3_independent_evaluation_results.mat Merged Monte Carlo results (500 trials).
│   ├── phase3_independent_per_realization.csv Detailed per-trial log (500 rows).
│   ├── phase3_independent_summary.csv Statistical summary table across noise levels.
│   ├── fig_phase3_independent_sumrate.png Sum-rate vs noise floor (with error bands).
│   ├── fig_phase3_independent_gain.png Net gain vs noise floor (with 95% CI error bars).
│   ├── fig_phase3_independent_paired.png Paired scatter plot (500 realizations).
│   ├── fig_phase3_independent_distribution.png Boxplot distributions of paired gain.
│   ├── fig_phase3_independent_cdf.png Empirical CDF of paired improvement.
│   ├── phase3_eval_data.mat         Preliminary 50-realization test dataset.
│   ├── phase3_evaluation_results.mat Preliminary 50-realization evaluation results.
│   └── phase3_evaluation_summary.csv Preliminary 50-realization summary table.
│
└── tests/
    ├── test_phase1.py              17 sanity checks on channel/pilot/noise physics.
    ├── test_phase3.py              12 checks on historical GNN pipeline.
    └── test_deep_beamforming.py    Sanity checks on DeepBeamformingNet, unit-modulus
                                    phase constraints, and differentiable RZF beamformer.
```

**How to run the current verified pipeline:**
```bash
# 1. Activate environment
./venv/Scripts/activate

# 2. Run unit tests
python -X utf8 -m tests.test_phase1
python -X utf8 -m tests.test_deep_beamforming

# 3. Train the Deep Beamforming Network (optional: checkpoint is already trained and frozen)
python -m gnn.train_beamforming --train_samples 300 --val_samples 60 --epochs 35

# 4. Generate the independent 500-realization test dataset (seeds 100,000–100,499)
python -m results.generate_independent_500_data

# 5. Run the full Monte Carlo evaluation in MATLAB (parallel workers or sequential)
matlab -batch "cd('d:/VIT/SEM 7/project/irs_project'); addpath('matlab'); run_eval_range(1, 125, 1)"
matlab -batch "cd('d:/VIT/SEM 7/project/irs_project'); addpath('matlab'); run_eval_range(126, 250, 2)"
matlab -batch "cd('d:/VIT/SEM 7/project/irs_project'); addpath('matlab'); run_eval_range(251, 375, 3)"
matlab -batch "cd('d:/VIT/SEM 7/project/irs_project'); addpath('matlab'); run_eval_range(376, 500, 4)"

# 6. Process results, run paired hypothesis tests, export CSVs, and generate publication plots
python -m results.process_independent_500_results
```

---

## 4. Phase 1 — IRS/Channel Simulation Environment (COMPLETE)

**What it does:** generates realistic BS-IRS and IRS-user channels,
simulates pilot transmission and reception under configurable noise, and
validates all of it with sanity tests.

**Key files:** `config/system_config.py`, `simulation/channel.py`,
`simulation/pilot.py`.

**Key design decisions worth knowing:**
- Path loss and fading are physically motivated (Rician for BS-IRS,
  Rayleigh for IRS-user), not arbitrary.
- IRS phases are always exactly unit-modulus (`|θ_m| = 1`), enforced by
  construction (`exp(iφ)`), matching the passive-IRS hardware constraint.
- The noise floor is **fixed and absolute** (dBm), and received SNR is
  **measured, not set** — see §2. `compute_received_snr_db()` in
  `simulation/pilot.py` is how you check what SNR a given noise floor
  actually produces for a given channel realization.

**Verification:** `python3 -m tests.test_phase1` — 17 checks, all passing
as of this document (matrix shapes, orthogonality of the pilot matrix,
dBm↔Watts conversion correctness, no NaN/Inf anywhere, SNR-vs-noise-floor
trend sanity, etc.).

---

## 5. Phase 2 — Classical Estimation + AO Baselines (COMPLETE)

**What it does:** (a) estimates the cascaded channel `C_k` from noisy
multi-phase pilots via Least Squares, and (b) runs classical Alternating
Optimization (AO) with Semidefinite Relaxation (SDR) to jointly optimize
the BS transmit beamformer and the IRS phase configuration, both under
perfect CSI (upper bound) and under the *estimated* CSI (the realistic,
honest baseline).

**Key files:** `simulation/multi_phase_estimation.py` (Python, the LS
estimator), `matlab/Opt_func_perfectCSI.m` and `matlab/Opt_func_Ck.m`
(MATLAB, the two AO variants), `matlab/run_phase2.m` /
`run_snr_sweep.m` / `run_tau_sweep.m` (drivers/sweeps).

**How the LS estimator works** (`multi_phase_pilot_estimate_C`):
1. Cycle through `Q = M = 64` random IRS phase configurations.
2. At each one, transmit `τ = 16` pilot symbols, receive a noisy signal,
   and compute a per-phase LS estimate of the effective channel
   `h_eff_hat[q, k, :]` for every user `k`.
3. Stack all `Q` per-phase estimates and solve
   `Θ_stack @ C_k = h_eff_hat_stack[:, k, :]` via `np.linalg.lstsq` — one
   joint LS solve per user, using all 64 measurements at once.
4. Returns `C_hat` (the estimate) alongside `C_true` (ground truth, known
   because this is a simulation) so accuracy can be measured directly.

**How the AO baseline works** (`Opt_func_Ck.m`): takes the `C_hat` cell
array directly (never reconstructs `H`/`G` separately — this was a
deliberate design choice to avoid noise amplification from dividing by
small channel coefficients) and alternates between (a) fixing the IRS
phases and solving for the BS beamformer via zero-forcing + power
allocation, and (b) fixing the beamformer and solving for IRS phases via
SDR, until the sum rate converges. `verify_Ck_against_reference.m` confirms
this cascaded-channel formulation is numerically identical to a reference
that separately reconstructs `H`, `G` — i.e., no accuracy was traded away
for the cascaded formulation's noise-robustness benefit.

**Results already generated and saved (`results/*.mat`, `*.png`):**
- **NMSE vs received SNR**: from ≈ −32.9 dB NMSE at ~48 dB SNR down to
  ≈ +7.1 dB NMSE at ~8 dB SNR (50 Monte Carlo realizations per point).
  LS is very accurate at low noise, degrades sharply as noise increases —
  exactly the expected, physically sane trend.
- **Sum rate vs SNR**: Perfect-CSI AO achieves ≈ 24–48 bps/Hz across the
  swept range; Estimated-CSI AO (true channels evaluated with the
  estimated-CSI-derived beamformer) achieves only ≈ 2.5–25 bps/Hz over the
  same range — a large, well-documented CSI-mismatch gap. This gap *is*
  the problem Phase 3 is meant to help close.
- **τ (pilot length) sweep**: effective spectral efficiency peaks around
  τ = 16–32, illustrating the overhead/accuracy trade-off. The saved
  result (`results/tau_sweep_results.mat`) was generated with
  `T_coherence = 5000` symbols; note `matlab/run_tau_sweep.m`'s own
  in-file default is `T_coherence = 200` — the saved results used an
  overridden/larger value, so check which `T_coherence` is actually set
  before re-running or comparing against the saved numbers.

**Verification:** Phase 1's 17 tests still pass after Phase 2's additions
(confirmed as part of Stage 3 work); `verify_Ck_against_reference.m`
cross-checks the cascaded formulation numerically.

---

## 6. Phase 3 (Historical) — GNN Channel Estimation NMSE Regression (FAILED / SUPERSEDED)

> [!NOTE]
> **HISTORICAL ARCHIVE:** This section documents the *initial, failed* Phase 3 formulation (channel estimation NMSE regression). It is preserved strictly for scientific record and to prevent future regression to failed ideas. The actual active and successful Phase 3 implementation is **Deep Beamforming** (detailed in Sections 10–14).

**Initial Framing (Old):**
In early development, Phase 3 attempted to train a Graph Neural Network (`NoiseAwareGNN` in `gnn/model.py`) to minimize the Frobenius Normalized Mean Squared Error (NMSE) of the cascaded channel:
$$\text{Noisy Pilots } Y \xrightarrow{\text{LS}} \hat{C}_{\text{LS}} \xrightarrow{\text{GNN}} \hat{C}_{\text{GNN}} \quad [\text{Target: minimize } \|\hat{C}_{\text{GNN}} - C_{\text{true}}\|_F^2]$$

**Why It Failed:**
At $Q = M = 64$ pilot phases, the linear observation system is determined and subject to i.i.d. Gaussian noise. Under this classical linear-Gaussian model, Least Squares is the Minimum Variance Unbiased Estimator (MVUE). Unweighted mean-pooling across nodes cannot invert random phase matrices better than pseudoinversion. Consequently, the GNN learned a near-zero residual correction and remained statistically tied with LS (within 0.1–0.3% NMSE margin, see §6.4). This negative result motivated the decisive pivot in Section 10 to end-to-end Deep Beamforming.

### 6.1 What was decided in the historical NMSE formulation and why

**Target:** predict `C_k` (same quantity the LS estimator produces), not
beamforming weights or IRS phases directly. Reason: it plugs into the
*existing, verified* `Opt_func_Ck.m` with zero changes, so GNN-vs-LS is a
clean apples-to-apples swap of one estimator for another, upstream of
beamforming.

**Formulation — residual correction, not from-scratch prediction.** The
GNN does **not** predict `C_k` from nothing. It predicts a *correction* to
the LS estimate: `prediction = C_hat_LS + learned_correction`. This was
not the first thing tried — see §6.3 for why the naive from-scratch
formulation failed outright.

**Graph design (per user, per channel realization):**
- **Nodes:** `Q = M = 64`, one per IRS phase measurement used during
  pilot training (the same 64 measurements the LS solve consumes).
- **Node features** (dim = `2M + 2N + 1 = 145`):
  `[Re(θ_q), Im(θ_q), Re(h_eff_hat_q), Im(h_eff_hat_q), snr_db]` — the
  phase configuration used at that measurement, the noisy per-phase
  channel estimate at that measurement, and the (normalized) received
  SNR for this sample, broadcast to every node so the network can
  explicitly condition on noise level.
- **Edges:** fully connected (every node attends to every other node) —
  cheap at 64 nodes, and mirrors what the joint LS solve already does
  (combining all 64 measurements at once).
- **Readout:** mean-pool over all 64 nodes → small MLP head → predicted
  correction, reshaped back to `C_k`'s real+imaginary flattened form.

**Normalization:** channel-magnitude values are tiny (~1e-6). Rather than
fit a normalization to the data (which would leak target statistics into
the input), a **fixed, physics-derived** scale constant
(`CHANNEL_SCALE = 1e-5`, derived analytically from the path-loss/gain
model in `config/system_config.py`) is used to bring both inputs and
targets to an O(1) range. See `channel_magnitude_scale()` in
`gnn/dataset.py` for the derivation.

**Noise-aware training:** the noise floor for each training sample is
drawn uniformly at random from the same range used in the Phase 2 sweep
(−100 to −70 dBm), so the model is exposed to many different noise
conditions rather than one fixed level. This is what "noise-aware" means
concretely in this implementation.

### 6.2 Architecture (`gnn/model.py`)

`NoiseAwareGNN`: a node encoder (Linear→ReLU), 1 "GraphConv" block (each
node's features are combined with the mean of all other nodes' features
via two learned linear maps + residual + LayerNorm — a hand-rolled,
minimal mean-aggregation message-passing layer; no PyTorch Geometric
dependency, by deliberate choice — see §6.5), mean-pool readout, and a
small MLP head. **88,512 parameters** — deliberately small for a
1500–5000-sample regime and to allow retraining in minutes on CPU.

### 6.3 What was tried and failed (important — don't repeat this)

1. **From-scratch prediction of `C_k`** (no residual connection): failed
   to beat even a trivial "always predict zero" baseline, even on a tiny
   16-example set the model could otherwise easily memorize (confirmed:
   it *could* memorize the tiny set — NMSE 1.03→0.03 over 300 epochs —
   but generalization on real validation data never moved off ~1.0, the
   zero-baseline). Root cause diagnosed: at low noise, LS is *already*
   near-exact (NMSE ≈ 0.008 at −85 dBm), so nearly all of `C_k`'s value is
   "linear-algebra-recoverable," and asking a mean-pooled GNN to
   rediscover that linear algebra from raw measurements, with no
   inductive bias for it, is an unnecessarily hard learning problem.
2. **Fresh-random-data-every-epoch training:** an early dataset
   implementation generated a brand-new random channel realization on
   every access, so the model never got repeated exposure to the same
   example across epochs. With early stopping kicking in after ~20
   epochs, the optimizer never had enough signal to learn anything.
   **Fixed** by pre-generating a fixed pool of samples once and reusing it
   across epochs (`IRSGraphDataset` in `gnn/dataset.py` now does this).
3. **Raw mean-NMSE loss with mixed-SNR training:** at the noisiest end of
   the training range (~−70 dBm), the per-phase LS solve is occasionally
   severely ill-conditioned for a small fraction of random channel
   realizations — individual per-sample NMSE values were measured as high
   as ~1400 (median at −70 dBm is a much more modest ~1.8–2.0). These
   outliers dominated a plain mean-NMSE loss and destabilized training for
   every other, well-behaved sample in the same batch. **Fixed** with (a)
   a clipped training loss (`robust_nmse_loss`, clips each sample's
   contribution at 5.0 before averaging — chosen because it's well above
   the *typical* LS NMSE across the sweep, so it only engages for genuine
   outliers) and (b) gradient-norm clipping, and (c) reporting **median**
   NMSE alongside mean for anything evaluated on this heavy-tailed data,
   since the mean alone is not representative of "typical" performance.

All of this is documented directly in code comments in `gnn/dataset.py`
and `gnn/train.py` (search for "ARCHITECTURE REFORMULATION" and "Stage 3
debugging finding").

### 6.4 Final trained result (honest, reproducible)

Training config used for the final checkpoint in `results/gnn/`:
300–400 epoch budget (early-stopped at epoch 125 on no val-median
improvement for 60 epochs), 3000 training realizations (12,000 user-graphs,
pre-generated pool), 400 validation realizations, batch size 32, Adam
lr=5e-4, hidden_dim=64, 1 GraphConv block, mixed noise floor uniform
−100 to −70 dBm, robust-loss clip=5.0, gradient-norm clip=1.0.
Total training time: **170.5 s on CPU** (no GPU in the dev environment).

**Per-noise-level evaluation** (`results/gnn/gnn_vs_ls_table.csv`, 50 fresh
Monte Carlo realizations per noise level, seeds disjoint from training):

| Noise floor (dBm) | LS median NMSE | GNN median NMSE | Verdict |
|---:|---:|---:|---|
| −100 | 0.00097 | 0.00117 | LS better |
| −95  | 0.00308 | 0.00327 | LS better |
| −90  | 0.00974 | 0.00990 | LS better |
| −85  | 0.03080 | 0.03088 | LS better |
| −80  | 0.09739 | 0.09732 | **GNN better** |
| −75  | 0.30796 | 0.30735 | **GNN better** |
| −70  | 0.97385 | 0.97107 | **GNN better** |

**Honest interpretation.** The GNN has essentially learned to reproduce
the LS estimate (the residual correction it learned is very close to
zero everywhere). It ties or marginally beats LS (by roughly 0.1–0.3%,
i.e. a few thousandths in absolute NMSE) only at the three *highest*-noise
levels tested — which is at least *directionally* consistent with where a
noise-aware model should help, since that's where LS itself is weakest —
but the margin is far too small to claim the core research hypothesis
("noise-aware learning improves robustness") has been demonstrated. See
`results/gnn/training_curve.png`: training and validation NMSE both
converge within ~5 epochs and then sit flat for 120 more epochs at
essentially the LS baseline level, which is consistent with "the model
found a near-identity solution and got stuck there," not "the model is
undertrained."

**What has been verified as correct** (so the *next* person doesn't waste
time re-suspecting the pipeline itself):
- Zero-prediction baseline loss is exactly `1.0` (confirms the NMSE loss
  function is implemented correctly).
- The residual connection is confirmed near-identity at initialization
  (confirms the architecture correctly starts from "LS plus a small
  learned nudge," not from scratch).
- A tiny fixed dataset can be memorized to near-zero loss (confirms the
  model has enough capacity and the optimizer/backprop path is not
  broken).
- Train/val pools use disjoint RNG seeds — no data leakage.
- The intermediate quantities used to build GNN inputs
  (`h_eff_hat_stack`, `thetas`) were cross-checked to produce
  **bit-for-bit identical** `C_true` and LS-recomputed `C_hat` as the
  original Phase 2 estimator, when fed the same RNG stream — so the
  GNN-vs-LS comparison is on identical underlying data, not an
  apples-to-oranges setup.
- All 12 of `tests/test_phase3.py`'s checks pass; all 17 of
  `tests/test_phase1.py`'s checks still pass (nothing in Phase 1/2 broke).

**In short: the pipeline is correct and the negative result is real, not a
bug.** The problem is that the current architecture (mean-pooling) and/or
training recipe genuinely isn't extracting a useful noise-dependent signal
beyond what LS already provides.

### 6.5 Known limitations / deliberate simplifications

- **No PyTorch Geometric.** The sandbox this was developed in had very
  limited disk space (~2.8 GB free at one point after installing plain
  PyTorch). A 64-node fully-connected graph is small enough to express as
  plain batched tensor ops, so PyG was skipped rather than risking a
  fragile, space-constrained install. If disk space is not a constraint in
  your environment, switching to PyG is *not required* to fix the current
  result, but could make trying more sophisticated GNN layers (e.g.
  attention) faster to implement.
- **No GPU** was available in the development sandbox; everything above
  was trained and timed on CPU. Training is currently fast enough (170s)
  that this hasn't been a blocker, but a larger/deeper model would
  benefit from GPU.
- **Mean-pooling readout.** This is very likely the single biggest reason
  the model hasn't beaten LS — mean-pooling forces every node
  (measurement) to be weighted equally, which throws away exactly the
  kind of "trust this measurement more/less" signal a noise-aware
  combiner needs. This is the top candidate to change next (§7).

---

## 7. Project Progression: Resolution of Phase 3 & Completion of Phase 4

### 7.1 How the Original Phase 3 Limitation Was Overcome
As documented in §6, attempting to beat Least Squares in raw channel NMSE under an i.i.d. Gaussian linear observation model was fundamentally bottlenecked by the Gauss-Markov theorem (LS is already the MVUE). Rather than continuing to pursue marginal NMSE gains via GNN feature engineering:
1. **Direct True Sum-Rate Optimization:** The network was redesigned as `DeepBeamformingNet` (`gnn/beamforming_model.py`) to bypass channel estimation error entirely, outputting beamforming configurations $(W_{\text{ML}}, \theta_{\text{ML}})$ trained end-to-end to maximize achievable true-channel sum-rate.
2. **Phase 3 Beamforming Integration Completed:** The neural beamformer was directly evaluated on true channels and benchmarked against conventional LS $\to$ Full AO and Perfect-CSI AO.

### 7.2 Completion of Phase 4 (Large-Scale Monte Carlo Verification)
Phase 4 (Large-Scale Monte Carlo Evaluation) has been **fully executed and completed**:
* Evaluated across **500 completely independent channel and noise realizations** (100 realizations each at $-100, -90, -85, -80, -70$ dBm) with disjoint seeds `100,000 – 100,499`.
* Evaluated all 6 competing branches on identical channel realizations.
* Rigorous paired statistical hypothesis testing performed (paired $t$-test $p = 1.06 \times 10^{-62}$, Wilcoxon $p = 2.86 \times 10^{-48}$).
* All five required publication-grade plots generated and saved in `results/`.

### 7.3 Active Research Horizon: Phase 5 & Experimental Testbed (Future Work)
With Phases 1 through 4 complete and verified, the remaining future directions are:
1. **Phase 5 — Quantization-Aware Extension (Groundwork Available):**
   Real-world IRS hardware frequently constrains phase shifts to 1-bit ($\{0, \pi\}$) or 2-bit ($\{0, \pi/2, \pi, 3\pi/2\}$) resolution. The quantization groundwork already exists in `matlab/Discerete.m` and `matlab/DiscreteSum.m`. A quantization-aware neural head (e.g. using Straight-Through Estimators) can be integrated into `DeepBeamformingNet`.
2. **Hardware/Testbed Prototyping:**
   Evaluating the frozen Deep Beamforming network on real over-the-air SDR testbeds to validate performance under mutual coupling and non-ideal RF impairments.

---

## 8. Key numbers to remember (quick reference)

| Parameter / Metric | Value |
|---|---|
| BS antennas ($N$) | 8 |
| IRS elements ($M$) | 64 |
| Single-antenna users ($K$) | 4 |
| Pilot length per configuration ($\tau$) | 16 symbols |
| Multi-phase IRS configurations ($Q$) | 64 ($= M$) |
| Total pilot overhead per realization ($Q \cdot \tau$) | **1024 symbols** |
| Active Model Architecture | `DeepBeamformingNet` ($M=64$ elements, 2 `ElementBlock` layers, unit-modulus phase head, adaptive RZF head) |
| Active Frozen Checkpoint | `results/gnn/best_beamforming_model.pt` |
| Training Dataset | **300 channel realizations** ($307,200$ pilot symbols), seeds `1000–1299` |
| Validation Dataset | **60 channel realizations** ($61,440$ pilot symbols), seeds `50000–50059` |
| Preliminary Test Dataset | **50 channel realizations** ($51,200$ pilot symbols), seeds `80000–80049` |
| Final Independent Verification Test Set | **500 channel realizations** ($512,000$ pilot symbols), seeds `100000–100499` |
| Overall Paired Win Rate (500 trials) | **81.6% (408 wins / 500 total)** |
| Moderate-to-Severe Noise Win Rate ($-85$ to $-70$ dBm) | **99.3% (298 wins / 300 total)** |
| Overall Mean Sum-Rate Gain ($R_{\text{ML}} - R_{\text{LS}}$) | **+8.44 bps/Hz (+64.9% relative gain)** |
| Overall 95% Confidence Interval | **[+7.59, +9.30] bps/Hz** |
| Statistical Significance ($p$-values) | Paired $t$-test: **$1.06 \times 10^{-62}$**, Wilcoxon: **$2.86 \times 10^{-48}$** |
| Neural Inference Latency | **2.73 ms ± 4.47 ms** (CPU forward pass) |
| ML Total End-to-End Pipeline Latency | **44.94 ms ± 8.98 ms** (LS Pilot Estimation + Neural Inference) |
| Conventional LS $\to$ Full AO Optimization Latency | **10.49 s ± 4.24 s** (average 6.4 iterations) |
| Conventional Total End-to-End Latency | **10.54 s ± 4.24 s** (LS Pilot Estimation + Full AO Solver) |
| Optimization Stage Speedup | **3,846.2× faster** ($10.49$ s vs $2.73$ ms) |
| End-to-End System Speedup | **234.4× faster** ($10.54$ s vs $44.94$ ms) |
| Operating Boundary | Conventional LS $\to$ AO is superior at $-100$ dBm (by $4.90$ bps/Hz); Deep Beamforming is superior at $-90$ to $-70$ dBm (by $+6.55$ to $+15.60$ bps/Hz). |

---

## 9. Guide for a new agent or researcher

Read in the following order:
1. This document: gives the complete context, the failure of channel-NMSE regression, and the success of direct Deep Beamforming.
2. `config/system_config.py`: defines all system geometry and parameters.
3. `simulation/multi_phase_estimation.py`: the multi-phase LS pilot protocol ($Q=64, \tau=16$).
4. `gnn/beamforming_model.py`: the active `DeepBeamformingNet` architecture and differentiable sum-rate loss function.
5. `results/generate_independent_500_data.py` and `matlab/run_eval_range.m`: the independent 500-instance evaluation pipeline.
6. `results/process_independent_500_results.py`: statistical processing, paired tests, and publication figure generation.

Confirm your environment by running the test suite:
```bash
python -X utf8 -m tests.test_phase1
python -X utf8 -m tests.test_deep_beamforming
```

---

## 10. Phase 3 Transformation: From Channel Estimation to Deep Beamforming

### 10.1 The Decisive Pivot
In earlier iterations of this project, Phase 3 was framed as neural channel estimation:
$$\text{Noisy Pilots } Y \xrightarrow{\text{LS}} \hat{C}_{\text{LS}} \xrightarrow{\text{GNN}} \hat{C}_{\text{GNN}} \quad [\text{Evaluated by Frobenius NMSE vs } C_{\text{true}}]$$
This framing produced negative results: at $Q = M = 64$, the linear observation system is determined and corrupted by complex circular symmetric Gaussian noise. Under the Gauss-Markov theorem, the Least Squares (LS) estimator is the Minimum Variance Unbiased Estimator (MVUE). No unweighted message-passing or spatial convolution could consistently outperform the pseudoinverse in NMSE across channel realizations.

More critically, in communication systems with imperfect CSI, **channel NMSE decouples from beamforming utility**. Even an estimate with moderate NMSE causes conventional Alternating Optimization (AO) to overfit its semidefinite relaxation against estimation artifacts, leading to severe beam misalignment and sum-rate collapse on the actual channel.

The research question was therefore decisively pivoted from NMSE regression to end-to-end communication performance:
> **"Given the SAME noisy pilot observations, can an ML-based beamforming method produce an active and passive configuration $(W, \theta)$ that achieves higher achievable SUM-RATE on the SAME underlying true channel than the conventional LS $\to$ AO pipeline?"**

* **Primary Metric:** TRUE-CHANNEL ACHIEVABLE SUM-RATE ($R_{\text{true}}$ in bps/Hz).
* **Secondary Metrics:** Runtime / latency, paired win rate, 95% confidence intervals, and parametric/non-parametric hypothesis tests ($p$-values).
* **Strict Rule:** The research objective is NOT "GNN beating LS in channel NMSE."

### 10.2 Dual Pipeline Architecture: Direct ML vs Conventional LS $\to$ AO
Both pipelines start from the **identical noisy pilot observations** and their performance is evaluated on the **identical true channel realizations**:

```
                       RAW NOISY PILOTS
             (Q = 64 configurations x tau = 16 = 1024 symbols)
                                |
                                v
               Multi-Phase LS Estimator (pinv)
                                |
                                v
               LS Cascaded Channel Estimate C_hat_LS
                                |
          +---------------------+---------------------+
          |                                           |
          v                                           v
[Conventional Baseline]                     [Proposed Deep Beamforming]
   C_hat_LS only                               C_hat_LS + noise floor sigma_dBm
          |                                           |
          v                                           v
Alternating Optimization (AO)                   DeepBeamformingNet
  - Step 1: W via SOCP/SDR/FP                     (2 ElementBlocks + Global Context)
  - Step 2: theta via SDR/Gaussian Rand               |
  - Iterate until convergence                         +--> IRS phase theta_ML (|theta_m|=1)
          |                                           +--> adaptive RZF reg alpha > 0
          v                                           +--> user power allocation p
  W_LS, theta_LS                                      |
          |                                           v
          |                                     Differentiable RZF W_ML
          |                                           |
          +---------------------+---------------------+
                                |
                                v
                      SAME TRUE CHANNEL
                   (H_true, G_true, C_true)
                                |
                                v
                     Achievable Sum-Rate
              R = sum_k log2(1 + SINR_k,true)
```

**Key Methodological Safeguard:**
Both pipelines receive strictly the noisy LS cascaded channel estimate $\hat{C}_{\text{LS}}$ (with Deep Beamforming also receiving the scalar pilot noise floor $\sigma_{\text{dBm}}$). Neither pipeline receives the true channels $G_{\text{true}}, H_{\text{true}}$, perfect CSI, or downstream optimization solutions during inference.

---

## 11. Candidate ML Formulations & Architectural Analysis

| Candidate | Architecture & Constraints | Training Target & Loss | Computational Complexity | Scientific Merit & Outcome |
|---|---|---|---|---|
| **A. Deep Beamforming (Direct ML)** | Input: $\hat{C}_{\text{LS}} \in \mathbb{C}^{K \times M \times N}$, $\sigma_{\text{dBm}}$.<br>Output: $\theta_{\text{ML}} \in \mathbb{C}^M$ ($|\theta_m|=1$) via complex normalization; $W_{\text{ML}} \in \mathbb{C}^{N \times K}$ via differentiable RZF with learned power allocation. | Differentiable true-channel sum-rate loss: $\mathcal{L} = -\mathbb{E}[R_{\text{true}}(H_{\text{true}}, G_{\text{true}}, \theta_{\text{ML}}, W_{\text{ML}})]$. | $2.73$ ms inference (ZERO optimization iterations). | **SELECTED & DEMONSTRATED**: Substantially outperforms LS $\to$ Full AO in the noisy regime ($-85$ to $-70$ dBm) by up to $+15.60$ bps/Hz. |
| **B. Joint Unstructured $(W, \theta)$** | Direct MLP/GNN output of unconstrained $W$ and $\theta$. | Direct sum-rate or supervised from AO. | $< 5$ ms. | **REJECTED**: Discards the optimal spatial filtering inductive bias of RZF; fails multi-user interference nulling. |
| **C. ML Warm-Start + Truncated AO** | GNN predicts $\theta^{(0)}$ fed to `Opt_func_Ck.m`, capped at $N_{\text{iter}} = 1$. | End-to-end true sum-rate or supervised phase. | ML inference ($2.73$ ms) + 1 AO iteration ($\approx 1.5$ s). | **TESTED**: AO optimization against noisy $\hat{C}_{\text{LS}}$ actively overfits to estimation noise, degrading performance relative to direct ML in severe noise. |
| **D. Task-Oriented Channel $\tilde{C}$** | GNN predicts regularized surrogate channel $\tilde{C}$ fed to existing CVX AO. | Backprop through CVX SDP is non-differentiable without intractable unrolling. | $> 10$ s (full AO). | **REJECTED**: Non-differentiability of CVX SDP prevents end-to-end learning. |

---

## 12. Implemented Architecture & Dataset Protocol

### 12.1 Neural Model: `DeepBeamformingNet` (`gnn/beamforming_model.py`)
* **Input:**
  1. LS cascaded channel estimate $\hat{C}_{\text{LS}} \in \mathbb{C}^{K \times M \times N}$
  2. Pilot noise floor $\sigma_{\text{dBm}} \in \mathbb{R}$
* **Strict Data Leakage Prohibition:**
  During inference, the model **MUST NOT receive**:
  * True channels $G_{\text{true}}$ (BS-to-IRS) or $H_{\text{true}}$ (IRS-to-users)
  * True cascaded channels $C_{k, \text{true}}$
  * Perfect CSI
  * Downstream AO solution $(W_{\text{AO}}, \theta_{\text{AO}})$
  * True beamformer or optimal power allocation
* **Input Feature Representation:**
  For each IRS element $m \in \{1, \dots, M\}$ ($M=64$ nodes):
  * Real and imaginary components of $\hat{C}_{k, m, n}$ for all $K=4$ users and $N=8$ BS antennas: $2 \times K \times N = 64$ dimensions.
  * Average element power: $\frac{1}{KN} \sum_{k,n} |\hat{C}_{k,m,n}|^2$ (1 dimension).
  * Normalized noise floor: $(\sigma_{\text{dBm}} + 85.0) / 15.0$ (1 dimension).
  * Total input feature per node: $\mathbf{x}_m \in \mathbb{R}^{66}$.
* **Encoder & Message Passing:**
  * Node Encoder: $\text{Linear}(66 \to 64) \to \text{ReLU} \to \text{Linear}(64 \to 64) \to \text{ReLU}$.
  * 2 `ElementBlock` layers: Combines self features with global mean-pooled context across all $M=64$ elements:
    $$\mathbf{h}_m^{(l+1)} = \text{LayerNorm}\left(\mathbf{h}_m^{(l)} + \text{FC}_2\left(\text{ReLU}\left(\text{FC}_1(\mathbf{h}_m^{(l)}) + \text{FC}_{\text{global}}\left(\frac{1}{M}\sum_{j=1}^M \mathbf{h}_j^{(l)}\right)\right)\right)\right)$$
* **Dual Output Heads:**
  1. **Phase Head (Per-Element):** $\text{Linear}(64 \to 64) \to \text{ReLU} \to \text{Linear}(64 \to 2)$ producing $(u_m, v_m)$.
     Unit-modulus projection strictly satisfies the IRS hardware constraint:
     $$\theta_m = \frac{u_m + j v_m}{\sqrt{u_m^2 + v_m^2 + 10^{-8}}}, \quad |\theta_m| \equiv 1.0$$
  2. **Global Power & Regularization Head:** Operates on global graph pool $\mathbf{h}_{\text{glob}} = \frac{1}{M}\sum_{m=1}^M \mathbf{h}_m$:
     * User power allocation logits $\to \text{Softmax} \to \mathbf{p} \in \mathbb{R}_+^K$ with $\sum_{k=1}^K p_k = 1.0$.
     * Adaptive RZF regularizer $\to \text{Softplus} \to \alpha > 0$.
* **Differentiable Active Beamformer Construction:**
  * Effective estimated channel: $\hat{H}_{\text{eff}} = \theta^T \hat{C} \in \mathbb{C}^{K \times N}$.
  * Unnormalized RZF: $W_{\text{unnorm}} = \hat{H}_{\text{eff}}^H (\hat{H}_{\text{eff}} \hat{H}_{\text{eff}}^H + \alpha I_K)^{-1} \in \mathbb{C}^{N \times K}$.
  * Unit column normalization: $\bar{\mathbf{w}}_k = \mathbf{w}_k / \|\mathbf{w}_k\|_2$.
  * Power scaling: $W_{\text{ML}} = [\sqrt{P_{\max} p_1}\,\bar{\mathbf{w}}_1, \dots, \sqrt{P_{\max} p_K}\,\bar{\mathbf{w}}_K]$ satisfying $\|W_{\text{ML}}\|_F^2 = P_{\max}$.
* **Parameter Count:** **42,439 trainable parameters** (verified via `torch.load`).
* **Active Frozen Checkpoint:** `results/gnn/best_beamforming_model.pt`.

### 12.2 Pilot Signaling Protocol
* BS Antennas: $N = 8$
* IRS Elements: $M = 64$
* Users: $K = 4$
* IRS Phase Configurations: $Q = 64$ orthogonal DFT/Hadamard configurations
* Pilot Symbols per Configuration: $\tau = 16$ orthogonal QR pilot sequences
* **Total Pilot Symbols per Channel Realization:**
  $$Q \times \tau = 64 \times 16 = 1,024 \text{ pilot symbols}$$
* **Critical Protocol Distinction:**
  The neural network **does NOT consume the 1,024 raw pilot symbols directly**. The $1,024$ noisy pilot observations are first processed by the multi-phase Least Squares estimator to yield the cascaded channel matrix $\hat{C}_{\text{LS}} \in \mathbb{C}^{K \times M \times N}$. The neural network consumes $\hat{C}_{\text{LS}}$ together with the noise floor $\sigma_{\text{dBm}}$.

### 12.3 Dataset-Size Summary Table

| Stage | Channel Realizations | Noise Levels | Pilot Symbols / Realization | Total Pilot Symbols | Purpose & Seed Range |
|---|:---:|:---:|:---:|:---:|---|
| **Training** | 300 | Mixed ($-100$ to $-70$ dBm) | 1,024 | 307,200 | Model training (`seeds 1000–1299`, $K=4$ users/realization) |
| **Validation** | 60 | Mixed ($-100$ to $-70$ dBm) | 1,024 | 61,440 | Model selection & early stopping (`seeds 50000–50059`) |
| **Preliminary Test** | 50 | 5 levels (10 / level) | 1,024 | 51,200 | Initial sanity validation (`seeds 80000–80049`, superseded) |
| **Final Independent Test** | **500** | 5 levels (100 / level) | 1,024 | **512,000** | Authoritative held-out verification (`seeds 100000–100499`) |

*Note: All 500 test channel realizations are completely disjoint from training, validation, and preliminary seeds. The model checkpoint was completely frozen with zero post-test adjustments.*

---

## 13. Preliminary 50-Realization Test (Historical / Superseded)

As an initial proof of concept, a 50-realization Monte Carlo evaluation (10 instances per noise level) was conducted (`results/phase3_evaluation_summary.csv`):

| Noise Floor (dBm) | Conventional LS $\to$ Full AO (bps/Hz) | Proposed Deep ML Direct (bps/Hz) | Absolute Gain: $R_{\text{ML}} - R_{\text{LS}}$ | Win Rate ($R_{\text{ML}} > R_{\text{LS}}$) |
|:---:|:---:|:---:|:---:|:---:|
| **−70.0** | 3.36 ± 1.28 | **21.20 ± 4.58** | **+17.84 bps/Hz** | **100.0% (10/10)** |
| **−80.0** | 7.38 ± 3.33 | **21.85 ± 3.40** | **+14.47 bps/Hz** | **100.0% (10/10)** |
| **−85.0** | 8.22 ± 3.34 | **23.27 ± 2.30** | **+15.05 bps/Hz** | **100.0% (10/10)** |
| **−90.0** | 17.11 ± 9.08 | **22.40 ± 1.91** | **+5.29 bps/Hz** | **60.0% (6/10)** |
| **−100.0** | **27.79 ± 6.47** | 21.81 ± 2.63 | −5.98 bps/Hz | 20.0% (2/10) |

*Preliminary Runtime Observation:* Average full AO runtime was observed at $14.85$ s vs $2.8$ ms for direct ML inference ($\approx 5,300\times$ optimization speedup).

> [!NOTE]
> **Status:** The 50-realization test served as preliminary evidence. It is now **formally superseded** by the 500-realization independent verification in Section 14, which provides exact confidence intervals, paired hypothesis tests, and rigorous apples-to-apples latency profiling.

---

## 14. Final Independent Verification (500 Realizations)

### 14.1 Independent Experimental Design
To definitively verify whether the Deep Beamforming gain is statistically robust and reproducible:
* **Frozen Checkpoint:** `results/gnn/best_beamforming_model.pt` evaluated in `eval()` mode. Zero retraining, zero hyperparameter tuning, zero test-set cherry-picking.
* **Strict Seed Partitions (100 independent realizations per noise level):**
  * $-100$ dBm: Seeds `100000 – 100099` (100 realizations)
  * $-90$ dBm: Seeds `100100 – 100199` (100 realizations)
  * $-85$ dBm: Seeds `100200 – 100299` (100 realizations)
  * $-80$ dBm: Seeds `100300 – 100399` (100 realizations)
  * $-70$ dBm: Seeds `100400 – 100499` (100 realizations)
  * **Total Test Volume:** 500 realizations $\times$ 1,024 pilot symbols = **512,000 pilot symbols**.
* **Identical Observation Model:** For every trial, both conventional and ML pipelines operated on the exact same noisy pilot observations and were scored on the exact same true channel matrices.

### 14.2 Final Numerical Results Table (True-Channel Sum-Rate in bps/Hz)
Derived directly from verified test artifacts `results/phase3_independent_summary.csv` and `results/phase3_independent_per_realization.csv`:

| Noise (dBm) | $N$ | Rand RZF (bps/Hz) | Conventional LS $\to$ Full AO (bps/Hz) | Proposed Deep ML Direct (bps/Hz) | Net Gain ($R_{\text{ML}} - R_{\text{LS}}$) | Paired Win Rate | 95% Confidence Interval | Paired $t$-test $p$-value | Wilcoxon $p$-value |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **−70.0** | 100 | 2.20 ± 0.69 | 3.76 ± 2.08 (med: 2.96) | **19.36 ± 4.55 (med: 19.14)** | **+15.60 bps/Hz (+415.3%)** | **100/100 (100.0%)** | [+14.68, +16.53] | **$9.64 \times 10^{-56}$** | **$3.90 \times 10^{-18}$** |
| **−80.0** | 100 | 1.93 ± 0.64 | 7.41 ± 4.39 (med: 6.07) | **21.55 ± 3.13 (med: 21.57)** | **+14.15 bps/Hz (+191.0%)** | **100/100 (100.0%)** | [+13.26, +15.03] | **$1.60 \times 10^{-53}$** | **$3.90 \times 10^{-18}$** |
| **−85.0** | 100 | 2.22 ± 0.81 | 10.97 ± 6.42 (med: 9.80) | **21.77 ± 3.21 (med: 21.95)** | **+10.81 bps/Hz (+98.5%)** | **98/100 (98.0%)** | [+9.63, +11.98] | **$1.95 \times 10^{-33}$** | **$5.43 \times 10^{-18}$** |
| **−90.0** | 100 | 1.98 ± 0.60 | 15.50 ± 7.20 (med: 14.44) | **22.05 ± 2.57 (med: 21.98)** | **+6.55 bps/Hz (+42.3%)** | **83/100 (83.0%)** | [+5.13, +7.97] | **$7.36 \times 10^{-15}$** | **$2.62 \times 10^{-12}$** |
| **−100.0** | 100 | 2.00 ± 0.62 | **27.40 ± 9.04 (med: 29.49)** | 22.51 ± 2.42 (med: 22.25) | −4.90 bps/Hz (−17.9%) | 27/100 (27.0%) | [−6.64, −3.15] | $2.32 \times 10^{-7}$ | $5.86 \times 10^{-7}$ |

*Reference Benchmarks:*
* Random-IRS + RZF baseline achieves $\approx 1.93 - 2.22$ bps/Hz across all noise regimes.
* Perfect-CSI AO upper benchmark achieves $34.86 - 36.14$ bps/Hz.

### 14.3 Overall Aggregate Across All 500 Realizations
* **Conventional LS $\to$ Full AO Mean Sum-Rate:** $13.01$ bps/Hz
* **Proposed Deep ML Direct Mean Sum-Rate:** **$21.45$ bps/Hz**
* **Overall Mean Paired Gain ($R_{\text{ML}} - R_{\text{LS}}$):** **$+8.44$ bps/Hz (+64.9%)**
* **Overall Paired Win Rate:** **81.6% (408 wins out of 500 trials)**
* **Moderate-to-Severe Noise Win Rate ($-85$ to $-70$ dBm):** **298 wins out of 300 trials = 99.3%**
  *(Important distinction: ML does NOT win 100% across $-85$ to $-70$ dBm; at $-85$ dBm, 2 realizations favored AO, yielding exactly $99.3\%$.)*
* **Overall 95% Confidence Interval for Gain:** $[+7.59, +9.30]$ bps/Hz
* **Overall Paired $t$-test $p$-value:** **$1.0644 \times 10^{-62}$**
* **Overall Wilcoxon signed-rank $p$-value:** **$2.8595 \times 10^{-48}$**

### 14.4 Apples-to-Apples Runtime & Latency Profile
Measured consistently across all 500 realizations on identical hardware:

| Pipeline Stage | Conventional LS $\to$ Full AO | Proposed Deep Beamforming (Direct ML) | Stage Speedup |
|---|:---:|:---:|:---:|
| **Pilot Channel Estimation (LS)** | $42.21 \pm 5.98$ ms | $42.21 \pm 5.98$ ms | $1.0\times$ (identical) |
| **Optimization / Inference Stage** | $10.49 \pm 4.24$ s (avg 6.4 iterations) | **$2.73 \pm 4.47$ ms** (single forward pass) | **3,846.2× faster** |
| **Total End-to-End Latency** | **$10.54 \pm 4.24$ s** | **$44.94 \pm 8.98$ ms** | **234.4× faster** |

*Note on Historical Speedup:* Earlier preliminary documentation reported an estimated "5,300× speedup" based on a single iteration count. The authoritative, measured optimization-stage speedup across all 500 instances is **3,846.2×**, with an end-to-end speedup of **234.4×**.

### 14.5 Scientific Interpretation & Operating Boundary
The 500-instance verification reveals an essential physical operating boundary:
1. **Near-Perfect CSI Regime ($-100$ dBm):**
   At $-100$ dBm, the pilot SNR is high and $\hat{C}_{\text{LS}}$ is very close to ground truth. Conventional AO exploits this accurate CSI via fine-grained semidefinite relaxation and achieves $27.40$ bps/Hz, outperforming ML direct inference by $4.90$ bps/Hz.
2. **Moderate-to-Severe Noise Regime ($-85$ to $-70$ dBm):**
   As pilot noise increases, conventional AO continues to solve its mathematical program against corrupted channel matrices. This causes severe beam misalignment on the true channel, collapsing AO sum-rate from $10.97$ bps/Hz down to $3.76$ bps/Hz. In contrast, `DeepBeamformingNet` was trained directly on true achievable sum-rate under noisy inputs, learning noise-robust phase alignment and adaptive regularization $\alpha$. ML achieves $19.36 - 21.77$ bps/Hz, beating AO in **298 of 300 trials (99.3%)** with gains up to $+15.60$ bps/Hz ($+415\%$).
3. **Core Scientific Conclusion:**
   The claim is NOT that "ML unconditionally beats AO everywhere." Rather:
   > **"The proposed Deep Beamforming method substantially outperforms conventional LS $\to$ AO under moderate-to-severe pilot noise ($-85$ to $-70$ dBm), while providing a 234.4× end-to-end latency reduction (44.9 ms vs 10.5 s), whereas conventional LS $\to$ AO remains superior in the near-perfect-CSI regime ($-100$ dBm)."**

---

# Current Project Handoff

### 10-Point Orientation for Researchers and AI Agents

1. **Historical GNN Failure:** The original attempt to make a GNN beat Least Squares in channel-estimation Frobenius NMSE failed because LS is the mathematical MVUE for the linear-Gaussian observation model. NMSE optimization has been formally superseded and preserved as historical work.
2. **Current Pivot:** The project pivoted to direct Deep Beamforming: mapping noisy LS channel estimates $\hat{C}_{\text{LS}}$ directly to active beamformer $W$ and IRS phase $\theta$ to maximize true-channel achievable sum-rate.
3. **Active Model & Checkpoint:** The current model is `DeepBeamformingNet` (`gnn/beamforming_model.py`, 42,439 parameters) with frozen checkpoint `results/gnn/best_beamforming_model.pt`.
4. **Primary Evaluation Metric:** All claims and comparisons are based on **true-channel achievable sum-rate** on identical channels and identical noisy pilots. Channel NMSE is diagnostic only.
5. **Authoritative Benchmark:** The authoritative result is the **500-realization independent Monte Carlo evaluation** across 5 noise levels (100 trials each, seeds 100000–100499, total 512,000 pilot symbols).
6. **Overall Performance:** ML achieves an overall mean sum-rate of **$21.45$ bps/Hz** vs **$13.01$ bps/Hz** for conventional LS $\to$ AO, yielding an average paired gain of **$+8.44$ bps/Hz (+64.9%)** with a win rate of **$81.6\%$ (408/500)** ($p = 1.06 \times 10^{-62}$).
7. **Moderate-to-Severe Noise Dominance:** Across $-85$ to $-70$ dBm, ML wins **298 out of 300 trials ($99.3\%$)** with gains up to $+15.60$ bps/Hz ($+415.3\%$). Do not claim 100% win rate across this range.
8. **Operating Boundary:** Conventional LS $\to$ AO outperforms ML at $-100$ dBm ($27.40$ vs $22.51$ bps/Hz) because AO can exploit near-perfect CSI. The negative gain at $-100$ dBm is preserved and documented as an honest operating boundary.
9. **Latency Profile:** Total end-to-end latency is **$44.94$ ms** for ML (42.2 ms LS est + 2.73 ms ML infer) vs **$10.54$ s** for conventional AO, representing a **$234.4\times$ end-to-end speedup** and a **$3,846.2\times$ optimization-stage speedup**.
10. **Documentation Rule:** The current model, checkpoint, and 500-instance evaluation results are frozen. Do NOT claim retraining, architecture changes, or altered metrics unless an explicit new experiment is executed.

### Reproduction Commands
```bash
# 1. Activate virtual environment
./venv/Scripts/activate

# 2. Run unit tests
python -X utf8 -m tests.test_phase1
python -X utf8 -m tests.test_phase3

# 3. Train the Deep Beamforming Network (reproduces best_beamforming_model.pt)
python -m gnn.train_beamforming --train_samples 300 --val_samples 60 --epochs 35

# 4. Generate 500-instance independent test dataset (seeds 100000-100499)
python -m results.generate_independent_500_data

# 5. Run independent 500-instance Monte Carlo evaluation across MATLAB AO and ML branches
matlab -batch "cd('d:/VIT/SEM 7/project/irs_project'); addpath('matlab'); run_phase3_independent_eval"

# 6. Process independent results, compute statistical tests, and generate publication figures
python -m results.process_independent_500_results
```


