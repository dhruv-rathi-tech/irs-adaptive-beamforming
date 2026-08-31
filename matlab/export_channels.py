"""
Export channels + estimated cascaded channels to .mat for MATLAB AO (Phase 2).

Two exports:
  1. Perfect-CSI case: raw H, G (ground truth) -> matlab/run_phase2.m calls
     Opt_func_perfectCSI.m directly (unmodified reference algorithm, upper bound).
  2. Estimated-CSI case: C_hat (cell-compatible list of (M,N) arrays) from the
     multi-phase LS pilot protocol -> matlab/run_phase2.m calls Opt_func_Ck.m.

Both exports also include recovery_err / avg_received_snr_db for reporting,
and tau_total for pilot-overhead accounting (SE_eff calculations later).
"""
import numpy as np
from scipy.io import savemat
from config.system_config import SystemConfig
from simulation.channel import generate_bs_irs_channel, generate_irs_user_channels
from simulation.multi_phase_estimation import multi_phase_pilot_estimate_C


def export_phase2_data(cfg: SystemConfig, pilot_noise_power_dbm: float, data_noise_power_dbm: float,
                        seed: int, out_path: str):
    """
    pilot_noise_power_dbm : noise floor during pilot/estimation phase (affects C_hat quality)
    data_noise_power_dbm  : noise floor during data transmission phase (affects AO's `sigma`,
                             i.e. the receiver noise used when computing sum-rate/SINR).
    These are logically separate quantities even if numerically equal by default;
    kept explicit here to avoid silently conflating pilot-phase and data-phase noise.
    """
    rng = np.random.default_rng(seed)
    G = generate_bs_irs_channel(cfg, rng)
    H, d_IU = generate_irs_user_channels(cfg, rng)

    est = multi_phase_pilot_estimate_C(G, H, cfg, rng, noise_power_dbm=pilot_noise_power_dbm)

    # MATLAB cell array equivalent: object array of (M,N) matrices
    C_hat_cell = np.empty((1, cfg.K), dtype=object)
    for k in range(cfg.K):
        C_hat_cell[0, k] = est["C_hat"][k]

    data_sigma_watts = 10 ** ((data_noise_power_dbm - 30) / 10.0)

    savemat(out_path, {
        "M": cfg.M, "N": cfg.N, "K": cfg.K,
        "P_max": cfg.pilot_power,  # NOTE: reference AO's P_max is the DATA-phase transmit power
                                     # budget (separate concept from pilot_power, reused here for
                                     # simplicity since both represent BS transmit power capability).
        "sigma": data_sigma_watts,   # data-phase noise power (Watts), used by AO's SINR calc
        "H_true": H, "G_true": G,
        "C_hat": C_hat_cell,
        "tau_total": est["tau_total"],
        "Q": est["Q"],
        "recovery_err": est["recovery_err"],
        "avg_received_snr_db": est["avg_received_snr_db"],
        "pilot_noise_power_dbm": pilot_noise_power_dbm,
        "data_noise_power_dbm": data_noise_power_dbm,
    })
    print(f"Saved {out_path} | tau_total={est['tau_total']} | "
          f"recovery_err={np.round(est['recovery_err'], 4)}")


if __name__ == "__main__":
    import os
    cfg = SystemConfig()
    os.makedirs("results", exist_ok=True)
    export_phase2_data(cfg, pilot_noise_power_dbm=-95.0, data_noise_power_dbm=-85.0,
                        seed=cfg.seed, out_path="results/phase2_data_good_snr.mat")
    export_phase2_data(cfg, pilot_noise_power_dbm=-85.0, data_noise_power_dbm=-85.0,
                        seed=cfg.seed, out_path="results/phase2_data_mid_snr.mat")