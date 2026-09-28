"""
Phase 3 -- Independent 500-Realization Test Data Generator.

Generates 500 completely independent test realizations:
  - 100 realizations per noise level x 5 noise levels: [-100, -90, -85, -80, -70] dBm
  - Seeds: 100,000 to 100,499 (strictly disjoint from all training, val, and prior test pools)
  - Model checkpoint is FROZEN: results/gnn/best_beamforming_model.pt
  - Generates true channels, multi-phase noisy pilots, LS estimate C_hat_LS
  - Runs frozen ML model inference (receiving ONLY C_hat_LS and noise_dbm)
  - Measures pure PyTorch inference time and LS estimation time
  - Evaluates true achievable sum-rate for ML direct
  - Saves all instances into results/phase3_independent_eval_data.mat for MATLAB baseline evaluation
"""
from __future__ import annotations

import os
import time
import numpy as np
import scipy.io as sio
import torch

from config.system_config import SystemConfig
from simulation.channel import generate_bs_irs_channel, generate_irs_user_channels
from simulation.multi_phase_estimation import multi_phase_pilot_estimate_C
from gnn.beamforming_model import DeepBeamformingNet, compute_true_sum_rate_torch

NOISE_LEVELS = [-100.0, -90.0, -85.0, -80.0, -70.0]
N_TRIALS_PER_NOISE = 100
BASE_SEED = 100000
MODEL_PATH = "results/gnn/best_beamforming_model.pt"
OUT_PATH = "results/phase3_independent_eval_data.mat"


def main():
    cfg = SystemConfig()
    sigma_watts = 3.16e-12  # -85 dBm data noise
    P_max = cfg.pilot_power  # 10.0 W

    device = torch.device("cpu")
    print(f"Loading frozen checkpoint: {MODEL_PATH}")
    model = DeepBeamformingNet(M=cfg.M, N=cfg.N, K=cfg.K, hidden_dim=64, n_blocks=2).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    total_instances = len(NOISE_LEVELS) * N_TRIALS_PER_NOISE
    print(f"Generating independent test set: {len(NOISE_LEVELS)} noise levels x {N_TRIALS_PER_NOISE} trials = {total_instances} realizations.")
    print(f"Seed range: {BASE_SEED} to {BASE_SEED + total_instances - 1} (strictly disjoint).")

    records = []
    trial_counter = 0

    t_start_all = time.perf_counter()
    for n_idx, noise_dbm in enumerate(NOISE_LEVELS):
        t_noise_start = time.perf_counter()
        for trial in range(N_TRIALS_PER_NOISE):
            seed = BASE_SEED + trial_counter
            trial_counter += 1
            rng = np.random.default_rng(seed)

            # 1. Physics-based channels
            G = generate_bs_irs_channel(cfg, rng)
            H, _ = generate_irs_user_channels(cfg, rng)

            # 2. Multi-phase pilot observation and LS estimation
            t_ls_0 = time.perf_counter()
            est = multi_phase_pilot_estimate_C(G, H, cfg, rng, noise_power_dbm=noise_dbm)
            t_ls_est = time.perf_counter() - t_ls_0

            c_hat_np = np.stack(est["C_hat"], axis=0)  # (K, M, N)

            # 3. ML Inference (Input strictly: C_hat_LS and noise_dbm)
            c_hat_t = torch.from_numpy(c_hat_np).unsqueeze(0).to(torch.complex64)
            noise_t = torch.tensor([noise_dbm], dtype=torch.float32)

            t_inf_0 = time.perf_counter()
            with torch.no_grad():
                theta_ml, W_ml = model(c_hat_t, noise_t, P_max=P_max)
            t_infer = time.perf_counter() - t_inf_0

            # 4. True sum-rate evaluation on true channel
            G_t = torch.from_numpy(G).unsqueeze(0).to(torch.complex64)
            H_t = torch.from_numpy(H).unsqueeze(0).to(torch.complex64)
            rate_ml = compute_true_sum_rate_torch(H_t, G_t, theta_ml, W_ml, sigma_watts).item()

            theta_ml_np = theta_ml.squeeze(0).numpy()  # (M,)
            W_ml_np = W_ml.squeeze(0).numpy()          # (N, K)

            # Format C_hat as 1xK cell array for MATLAB
            C_hat_cell = np.empty((1, cfg.K), dtype=object)
            for k in range(cfg.K):
                C_hat_cell[0, k] = est["C_hat"][k]

            records.append({
                "noise_dbm": float(noise_dbm),
                "trial": int(trial),
                "seed": int(seed),
                "G_true": G,
                "H_true": H,
                "C_hat": C_hat_cell,
                "theta_ml": theta_ml_np,
                "W_ml": W_ml_np,
                "rate_ml_direct": float(rate_ml),
                "time_ml_infer": float(t_infer),
                "time_ls_est": float(t_ls_est),
                "P_max": float(P_max),
                "sigma": float(sigma_watts),
                "M": int(cfg.M),
                "N": int(cfg.N),
                "K": int(cfg.K),
            })

        print(f"Noise {noise_dbm:6.1f} dBm: 100 trials generated in {time.perf_counter() - t_noise_start:.2f} s.")

    total_gen_time = time.perf_counter() - t_start_all
    print(f"Total dataset generation time: {total_gen_time:.2f} s.")

    # Save to MATLAB .mat file
    sio.savemat(OUT_PATH, {
        "records": records,
        "noise_levels": NOISE_LEVELS,
        "n_trials": N_TRIALS_PER_NOISE,
        "base_seed": BASE_SEED,
    })
    print(f"Saved {OUT_PATH} successfully ({len(records)} realizations).")


if __name__ == "__main__":
    main()
