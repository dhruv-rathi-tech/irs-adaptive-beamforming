"""
Sum Rate Baseline data prep: exports .mat files across a noise sweep
for MATLAB to run Perfect-CSI AO and Estimated-CSI AO.

Run from project root:
    py results/export_channels_snr_sweep.py
"""

import os
import sys
import numpy as np
from scipy.io import savemat

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config.system_config import SystemConfig
from simulation.channel import (
    generate_bs_irs_channel,
    generate_irs_user_channels
)
from simulation.multi_phase_estimation import multi_phase_pilot_estimate_C


# Reduced sweep for faster Phase 2 experimentation
NOISE_DBM_SWEEP = [-100, -92.5, -85, -77.5, -70]

# Reduced number of channel realizations
N_REALIZATIONS = 3

SEED = SystemConfig().seed

out_dir = os.path.join(
    os.path.dirname(__file__),
    "snr_sweep"
)

os.makedirs(out_dir, exist_ok=True)

manifest = []

for noise_dbm in NOISE_DBM_SWEEP:

    for trial in range(N_REALIZATIONS):

        cfg = SystemConfig()

        # Same channel realization for a given trial
        # across different noise levels.
        rng = np.random.default_rng(
            SEED * 1000 + trial
        )

        G = generate_bs_irs_channel(cfg, rng)
        H, _ = generate_irs_user_channels(cfg, rng)

        # Estimate cascaded channel from noisy pilots
        est = multi_phase_pilot_estimate_C(
            G,
            H,
            cfg,
            rng,
            noise_power_dbm=float(noise_dbm)
        )

        # Convert Python list of C_hat matrices into
        # MATLAB-compatible 1 x K cell array.
        C_hat_cell = np.empty(
            (1, cfg.K),
            dtype=object
        )

        for k in range(cfg.K):
            C_hat_cell[0, k] = est["C_hat"][k]

        # Convert dBm -> Watts
        sigma_watts = 10 ** (
            (noise_dbm - 30) / 10.0
        )

        fname = (
            f"phase2_data_snr{noise_dbm}_trial{trial}.mat"
        )

        out_path = os.path.join(
            out_dir,
            fname
        )

        savemat(
            out_path,
            {
                "M": cfg.M,
                "N": cfg.N,
                "K": cfg.K,
                "P_max": cfg.pilot_power,
                "sigma": sigma_watts,
                "H_true": H,
                "G_true": G,
                "C_hat": C_hat_cell,
                "noise_power_dbm": noise_dbm,
            }
        )

        manifest.append(
            {
                "file": fname,
                "noise_dbm": noise_dbm,
                "trial": trial
            }
        )

    print(
        f"noise_floor={noise_dbm:6.1f} dBm | "
        f"{N_REALIZATIONS} trials exported"
    )


savemat(
    os.path.join(out_dir, "manifest.mat"),
    {
        "noise_dbm_sweep": NOISE_DBM_SWEEP,
        "n_realizations": N_REALIZATIONS,
    }
)

print(
    f"\nSaved {len(manifest)} files to {out_dir}/"
)