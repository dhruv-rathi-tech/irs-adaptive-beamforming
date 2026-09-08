"""
Graph 1 data generator: Average Cascaded-Channel NMSE vs Received SNR.

For each noise floor (i.e. each resulting SNR point), runs N_REALIZATIONS
independent (channel, pilot noise) trials, computes NMSE for each, then
averages -> removes the single-realization jaggedness in recovery_vs_noise.npz.

Lives in results/. Run from results/ folder:  python generate_nmse_vs_snr.py
(also runnable from project root: python results/generate_nmse_vs_snr.py)
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))  # project root, for config/ simulation/
from config.system_config import SystemConfig
from simulation.channel import generate_bs_irs_channel, generate_irs_user_channels
from simulation.multi_phase_estimation import multi_phase_pilot_estimate_C

N_REALIZATIONS = 50          # independent channel+noise trials per SNR point
NOISE_DBM_SWEEP = np.arange(-100, -59, 5)   # -100 to -60 dBm -> spans low to high SNR


def run():
    cfg = SystemConfig()
    snr_points = []
    nmse_mean = []
    nmse_std = []

    for noise_dbm in NOISE_DBM_SWEEP:
        trial_nmse = []
        trial_snr = []
        for trial in range(N_REALIZATIONS):
            rng = np.random.default_rng(cfg.seed * 1000 + trial)  # fresh channel + noise each trial
            G = generate_bs_irs_channel(cfg, rng)
            H, _ = generate_irs_user_channels(cfg, rng)

            est = multi_phase_pilot_estimate_C(G, H, cfg, rng, noise_power_dbm=float(noise_dbm))

            # NMSE in dB, aggregated (Frobenius) over all K users' C_k at once
            num = sum(np.linalg.norm(est["C_hat"][k] - est["C_true"][k]) ** 2 for k in range(cfg.K))
            den = sum(np.linalg.norm(est["C_true"][k]) ** 2 for k in range(cfg.K))
            nmse_db = 10 * np.log10(num / den)

            trial_nmse.append(nmse_db)
            trial_snr.append(np.mean(est["avg_received_snr_db"]))

        snr_points.append(np.mean(trial_snr))
        nmse_mean.append(np.mean(trial_nmse))
        nmse_std.append(np.std(trial_nmse))
        print(f"noise_floor={noise_dbm:6.1f} dBm | avg SNR={snr_points[-1]:7.2f} dB | "
              f"NMSE mean={nmse_mean[-1]:7.2f} dB | std={nmse_std[-1]:5.2f} dB "
              f"({N_REALIZATIONS} realizations)")

    out_dir = os.path.dirname(__file__)  # save alongside this script, i.e. results/
    np.savez(
        os.path.join(out_dir, "nmse_vs_snr.npz"),
        snr_db=np.array(snr_points),
        nmse_db_mean=np.array(nmse_mean),
        nmse_db_std=np.array(nmse_std),
        n_realizations=N_REALIZATIONS,
    )
    print(f"\nSaved {os.path.join(out_dir, 'nmse_vs_snr.npz')}")


if __name__ == "__main__":
    run()