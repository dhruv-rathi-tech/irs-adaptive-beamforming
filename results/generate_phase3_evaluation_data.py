"""
Phase 3 -- Paired Monte Carlo Test Data Generator.

Generates paired test sets across noise levels:
  [-100, -90, -85, -80, -70] dBm
Seeds are strictly disjoint from all training and validation pools (base_seed=80000).

For each test realization, generates:
  - G_true, H_true
  - noisy observations and LS estimate C_hat_LS
  - runs ML inference to obtain theta_ML and W_ML (recording inference time)
  - evaluates ML true sum-rate in Python
  - saves all channels and configurations into results/phase3_eval_data.mat
    so MATLAB can run the conventional baselines on the EXACT SAME instances.
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
N_TRIALS = 10
MODEL_PATH = "results/gnn/best_beamforming_model.pt"
OUT_PATH = "results/phase3_eval_data.mat"


def main():
    cfg = SystemConfig()
    sigma_watts = 3.16e-12  # -85 dBm data noise
    P_max = cfg.pilot_power  # 10.0 W

    device = torch.device("cpu")
    model = DeepBeamformingNet(M=cfg.M, N=cfg.N, K=cfg.K, hidden_dim=64, n_blocks=2).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    total_instances = len(NOISE_LEVELS) * N_TRIALS
    print(f"Preparing paired test dataset: {len(NOISE_LEVELS)} noise levels x {N_TRIALS} trials = {total_instances} realizations.")

    records = []
    trial_counter = 0

    for n_idx, noise_dbm in enumerate(NOISE_LEVELS):
        for trial in range(N_TRIALS):
            seed = 80000 + trial_counter
            trial_counter += 1
            rng = np.random.default_rng(seed)

            G = generate_bs_irs_channel(cfg, rng)
            H, _ = generate_irs_user_channels(cfg, rng)

            est = multi_phase_pilot_estimate_C(G, H, cfg, rng, noise_power_dbm=noise_dbm)
            c_hat_np = np.stack(est["C_hat"], axis=0)  # (K, M, N)

            # PyTorch ML Inference
            c_hat_t = torch.from_numpy(c_hat_np).unsqueeze(0).to(torch.complex64)
            noise_t = torch.tensor([noise_dbm], dtype=torch.float32)

            t0 = time.perf_counter()
            with torch.no_grad():
                theta_ml, W_ml = model(c_hat_t, noise_t, P_max=P_max)
            t_infer = time.perf_counter() - t0

            # Evaluate ML direct sum-rate on true channel
            G_t = torch.from_numpy(G).unsqueeze(0).to(torch.complex64)
            H_t = torch.from_numpy(H).unsqueeze(0).to(torch.complex64)
            rate_ml = compute_true_sum_rate_torch(H_t, G_t, theta_ml, W_ml, sigma_watts).item()

            theta_ml_np = theta_ml.squeeze(0).numpy()  # (M,)
            W_ml_np = W_ml.squeeze(0).numpy()          # (N, K)

            # MATLAB cell format for C_hat: 1xK cell array of (M, N)
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
                "P_max": float(P_max),
                "sigma": float(sigma_watts),
                "M": int(cfg.M),
                "N": int(cfg.N),
                "K": int(cfg.K),
            })

    # Save to MATLAB format
    # Export as struct array
    sio.savemat(OUT_PATH, {
        "records": records,
        "noise_levels": NOISE_LEVELS,
        "n_trials": N_TRIALS,
    })
    print(f"Saved {OUT_PATH} successfully.")


if __name__ == "__main__":
    main()
