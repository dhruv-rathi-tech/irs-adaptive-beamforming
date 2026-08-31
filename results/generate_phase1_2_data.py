"""
Data generation for Phase 1/2 verification plots (results/make_phase1_2_plots.py).

This script was missing from the repo -- the plotting script expects
results/pathloss_curve.npz and results/recovery_vs_noise.npz but no generator
for them existed anywhere in the project (confirmed by grep across the repo).

Both .npz files below are produced entirely from the already-implemented
simulation code (simulation/channel.py, simulation/pilot.py,
simulation/multi_phase_estimation.py) -- no hand-typed/fabricated numbers.

Run with: python -m results.generate_phase1_2_data   (from repo root)
"""

import os
import numpy as np

from config.system_config import SystemConfig
from simulation.channel import (
    path_loss_linear,
    generate_bs_irs_channel,
    generate_irs_user_channels,
)
from simulation.multi_phase_estimation import multi_phase_pilot_estimate_C


def generate_pathloss_curve(cfg: SystemConfig, out_path: str):
    """
    Phase 1 sanity check data: path loss (linear) vs distance for both links,
    using the project's actual log-distance model (path_loss_linear) and the
    actual configured exponents (alpha_BI, alpha_IU).
    """
    d = np.linspace(1.0, 100.0, 200)  # meters
    pl_bi = path_loss_linear(d, cfg.C0_dB, cfg.d0, cfg.alpha_BI)
    pl_iu = path_loss_linear(d, cfg.C0_dB, cfg.d0, cfg.alpha_IU)

    np.savez(out_path, d=d, pl_bi=pl_bi, pl_iu=pl_iu)
    print(f"Saved {out_path} | d.shape={d.shape}")


def generate_recovery_vs_noise(cfg: SystemConfig, out_path: str):
    """
    Phase 2 sanity check data: mean C_hat relative recovery error vs pilot-phase
    noise floor, using the actual multi-phase LS estimator
    (multi_phase_pilot_estimate_C) against actual generated channels G, H.

    A single fixed realization of G/H is used (matching the "single-realization
    sanity check" framing already stated in the plotting script's docstring/
    title -- Phase 4 is where proper Monte Carlo averaging happens).
    """
    rng = np.random.default_rng(cfg.seed)
    G = generate_bs_irs_channel(cfg, rng)
    H, _ = generate_irs_user_channels(cfg, rng)

    noise_floors = np.linspace(-100.0, -60.0, 15)  # dBm
    errs = np.zeros_like(noise_floors)

    for i, noise_dbm in enumerate(noise_floors):
        out = multi_phase_pilot_estimate_C(
            G, H, cfg, rng, noise_power_dbm=float(noise_dbm)
        )
        errs[i] = np.mean(out["recovery_err"])  # mean over K users

    np.savez(out_path, noise_floors=noise_floors, errs=errs)
    print(f"Saved {out_path} | noise_floors={noise_floors}")
    print(f"    mean recovery errs: {np.round(errs, 4)}")


if __name__ == "__main__":
    cfg = SystemConfig()
    os.makedirs("results", exist_ok=True)
    generate_pathloss_curve(cfg, "results/pathloss_curve.npz")
    generate_recovery_vs_noise(cfg, "results/recovery_vs_noise.npz")