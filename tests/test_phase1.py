"""
Phase 1 sanity checks. Run with: python -m tests.test_phase1  (from irs_project/)
"""
import numpy as np
from config.system_config import SystemConfig
from simulation.channel import generate_bs_irs_channel, generate_irs_user_channels, path_loss_linear
from simulation.pilot import receive_noisy_pilots, orthogonal_pilot_matrix, dbm_to_watts, compute_received_snr_db


def check(name, cond):
    status = "PASS" if cond else "FAIL"
    print(f"[{status}] {name}")
    assert cond, f"FAILED: {name}"


def main():
    cfg = SystemConfig()
    rng = np.random.default_rng(cfg.seed)

    # --- Channel shape & type checks ---
    G = generate_bs_irs_channel(cfg, rng)
    H, d_IU = generate_irs_user_channels(cfg, rng)

    check("G shape == (M, N)", G.shape == (cfg.M, cfg.N))
    check("H shape == (M, K)", H.shape == (cfg.M, cfg.K))
    check("G dtype complex", np.iscomplexobj(G))
    check("H dtype complex", np.iscomplexobj(H))
    check("d_IU within [d_IU_min, d_IU_max]",
          np.all(d_IU >= cfg.d_IU_min) and np.all(d_IU <= cfg.d_IU_max))

    # --- Path loss sanity: farther distance -> lower linear path loss ---
    pl_near = path_loss_linear(np.array([5.0]), cfg.C0_dB, cfg.d0, cfg.alpha_IU)
    pl_far = path_loss_linear(np.array([30.0]), cfg.C0_dB, cfg.d0, cfg.alpha_IU)
    check("path loss decreases with distance", pl_far[0] < pl_near[0])

    # --- Pilot matrix orthogonality: X_p @ X_p^H ~ (tau/N) * I_N ---
    X_p = orthogonal_pilot_matrix(cfg.N, cfg.tau, rng)
    gram = X_p @ X_p.conj().T
    expected = (cfg.tau / cfg.N) * np.eye(cfg.N)
    check("pilot matrix shape == (N, tau)", X_p.shape == (cfg.N, cfg.tau))
    check("pilot matrix orthogonality (X_p X_p^H ≈ (tau/N) I)",
          np.allclose(gram, expected, atol=1e-8))

    # --- dBm -> Watts conversion sanity ---
    check("0 dBm == 1 mW", np.isclose(dbm_to_watts(0.0), 1e-3))
    check("-85 dBm noise floor is tiny but positive", 0 < dbm_to_watts(-85.0) < 1e-6)
    check("higher dBm -> higher linear power", dbm_to_watts(-80.0) > dbm_to_watts(-90.0))

    # --- Full pilot reception pipeline (fixed noise-floor convention) ---
    out = receive_noisy_pilots(G, H, cfg, rng, noise_power_dbm=-85.0)
    check("h_eff shape == (K, N)", out["h_eff"].shape == (cfg.K, cfg.N))
    check("Y_noisy shape == (K, tau)", out["Y_noisy"].shape == (cfg.K, cfg.tau))
    check("theta unit modulus (IRS passive constraint)",
          np.allclose(np.abs(out["theta"]), 1.0, atol=1e-10))
    check("sigma_n2 matches dbm_to_watts(-85)", np.isclose(out["sigma_n2"], dbm_to_watts(-85.0)))
    check("received_snr_db has shape (K,)", out["received_snr_db"].shape == (cfg.K,))

    print(f"    emergent received SNR at -85 dBm noise floor: "
          f"{np.round(out['received_snr_db'], 1)} dB")

    # --- Noise floor sanity: worse (higher) noise floor -> lower received SNR ---
    out_noisy = receive_noisy_pilots(G, H, cfg, rng, theta=out["theta"], X_p=out["X_p"],
                                      noise_power_dbm=-60.0)  # much worse noise floor
    check("worse noise floor (-60dBm) -> lower received SNR than -85dBm",
          np.all(out_noisy["received_snr_db"] < out["received_snr_db"]))

    # --- Received SNR should scale ~10*log10 with noise floor changes (physical sanity) ---
    delta_dbm = 10.0
    out_a = receive_noisy_pilots(G, H, cfg, rng, theta=out["theta"], X_p=out["X_p"], noise_power_dbm=-85.0)
    out_b = receive_noisy_pilots(G, H, cfg, rng, theta=out["theta"], X_p=out["X_p"], noise_power_dbm=-85.0 + delta_dbm)
    snr_drop = out_a["received_snr_db"] - out_b["received_snr_db"]
    print(f"    SNR drop for +{delta_dbm}dBm noise floor increase: {np.round(snr_drop, 2)} dB (expected ~{delta_dbm} dB)")
    check("received SNR drops by ~delta_dbm when noise floor increases by delta_dbm",
          np.allclose(snr_drop, delta_dbm, atol=0.5))

    # --- No NaN/Inf anywhere ---
    for name_, arr in [("G", G), ("H", H), ("Y_noisy", out["Y_noisy"])]:
        check(f"{name_} has no NaN/Inf", np.all(np.isfinite(arr)))

    print("\nAll Phase 1 sanity checks passed.")


if __name__ == "__main__":
    main()