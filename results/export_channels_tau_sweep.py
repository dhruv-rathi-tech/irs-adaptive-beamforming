"""
Graph 3 data prep: exports one .mat file per pilot length tau, at a FIXED
SNR (fixed noise floor), so MATLAB can run AO on each and we can compute
Effective Spectral Efficiency vs pilot overhead.

Lives in results/. Run from results/ folder:  python export_channels_tau_sweep.py
(also runnable from project root: python results/export_channels_tau_sweep.py)
Requires cfg.M <= min(tau) (Q=M phases, so tau*Q must stay <= T_coherence
for meaningful overhead point; smallest tau here is intentionally small
to show the tradeoff even though Q=M=64 is fixed by identifiability).
"""
import os
import sys
import numpy as np
from scipy.io import savemat
from dataclasses import replace

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))  # project root, for config/ simulation/
from config.system_config import SystemConfig
from simulation.channel import generate_bs_irs_channel, generate_irs_user_channels
from simulation.multi_phase_estimation import multi_phase_pilot_estimate_C

TAU_SWEEP = [8, 16, 32, 64]        # pilot length per phase; must be >= N=8 (BS antennas)
FIXED_NOISE_DBM = -85.0            # fixed pilot+data noise floor for a fair sweep
T_COHERENCE_OVERRIDE = 5000        # realistic coherence block (symbols) so tau_total/T stays in (0,1)
                                    # for all tau in TAU_SWEEP (max tau_total = 64*64 = 4096 < 5000)
SEED = SystemConfig().seed

out_dir = os.path.join(os.path.dirname(__file__), "tau_sweep")  # results/tau_sweep
os.makedirs(out_dir, exist_ok=True)

for tau in TAU_SWEEP:
    cfg = replace(SystemConfig(), tau=tau, T_coherence=T_COHERENCE_OVERRIDE)
    rng = np.random.default_rng(SEED)
    G = generate_bs_irs_channel(cfg, rng)
    H, _ = generate_irs_user_channels(cfg, rng)

    est = multi_phase_pilot_estimate_C(G, H, cfg, rng, noise_power_dbm=FIXED_NOISE_DBM)

    C_hat_cell = np.empty((1, cfg.K), dtype=object)
    for k in range(cfg.K):
        C_hat_cell[0, k] = est["C_hat"][k]

    data_sigma_watts = 10 ** ((FIXED_NOISE_DBM - 30) / 10.0)
    out_path = os.path.join(out_dir, f"phase2_data_tau{tau}.mat")

    savemat(out_path, {
        "M": cfg.M, "N": cfg.N, "K": cfg.K,
        "P_max": cfg.pilot_power,
        "sigma": data_sigma_watts,
        "H_true": H, "G_true": G,
        "C_hat": C_hat_cell,
        "tau_total": est["tau_total"],   # = Q * tau
        "Q": est["Q"],
        "tau": tau,
        "T_coherence": cfg.T_coherence,
        "recovery_err": est["recovery_err"],
    })
    print(f"Saved {out_path} | tau={tau} | tau_total={est['tau_total']} "
          f"(overhead frac={est['tau_total']/cfg.T_coherence:.3f})")
