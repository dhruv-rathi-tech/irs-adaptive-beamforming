"""
Pilot transmission and noisy reception.

During the pilot phase, the IRS holds a FIXED (e.g. random or codebook) phase
configuration theta, and the BS sends tau known pilot symbols. Each user k
receives a noisy observation of the cascaded channel (h_k^H diag(theta) G).

This module produces exactly the "noisy pilot" data that:
  - the classical baseline (Phase 2) will feed into an LS channel estimator
  - the noise-aware GNN (Phase 3) will consume directly (implicit estimation)
"""
import numpy as np
from config.system_config import SystemConfig


def random_irs_phase(M: int, rng: np.random.Generator) -> np.ndarray:
    """Random unit-modulus IRS phase vector, shape (M,), complex128."""
    phases = rng.uniform(0, 2 * np.pi, size=M)
    return np.exp(1j * phases)


def orthogonal_pilot_matrix(N: int, tau: int, rng: np.random.Generator) -> np.ndarray:
    """
    Generate a pilot symbol matrix X_p of shape (N, tau): tau known pilot
    vectors sent across N BS antennas.

    Requires tau >= N to have (semi-)orthogonal pilots across antennas
    (standard requirement for identifiability of an N-antenna transmitter).
    We construct it via QR decomposition of a random complex Gaussian matrix,
    then scale columns to unit power per symbol.

    Returns X_p with X_p @ X_p^H proportional to identity (when tau >= N).
    """
    if tau < N:
        raise ValueError(
            f"Pilot length tau={tau} must be >= N={N} BS antennas for "
            f"an identifiable (orthogonal) pilot design."
        )
    A = rng.standard_normal((N, tau)) + 1j * rng.standard_normal((N, tau))
    Q, _ = np.linalg.qr(A.T)  # Q: (tau, N) orthonormal columns
    X_p = Q.T  # (N, tau), rows orthonormal -> X_p @ X_p^H = I_N
    # normalize so each column (each time-symbol) has unit total power summed over antennas
    X_p = X_p * np.sqrt(tau / N)
    return X_p.astype(np.complex128)


def dbm_to_watts(dbm: float) -> float:
    """Convert dBm (power relative to 1 mW) to linear Watts."""
    return 10 ** ((dbm - 30) / 10.0)


def compute_received_snr_db(signal_power_linear: float, sigma_n2: float) -> float:
    """
    Compute the actual RECEIVED SNR (dB) given received signal power and noise
    variance. This is an EMERGENT/measured quantity (depends on path loss,
    fading, IRS phase), not something set directly -- see noise convention
    note on noise_power_dbm in SystemConfig.
    """
    return 10 * np.log10(signal_power_linear / sigma_n2)


def receive_noisy_pilots(
    G,
    H,
    cfg: SystemConfig,
    rng: np.random.Generator,
    theta=None,
    X_p=None,
    noise_power_dbm=None,
) -> dict:
    """
    Simulate pilot transmission and noisy reception for all K users.

    Cascaded (effective) channel for user k under phase theta:
        h_eff_k = h_k^H @ diag(theta) @ G          shape: (N,)   [row vector as (N,) array]

    Received pilot signal for user k over tau symbols:
        y_k = h_eff_k @ X_p + n_k                  shape: (tau,)
        n_k ~ CN(0, sigma_n^2 I_tau)

    Noise variance is set from a FIXED absolute noise floor (dBm), not from a
    transmit-referenced SNR -- see SystemConfig.noise_power_dbm docstring.
    Received SNR is an emergent output; use compute_received_snr_db() to
    measure it after the fact (per-user, per-realization).

    Returns a dict with all intermediate quantities (needed for later phases:
    LS estimation, GNN input, and ground truth for supervised training/checks).
    """
    M, N = cfg.M, cfg.N
    K = cfg.K
    tau = cfg.tau

    if theta is None:
        theta = random_irs_phase(M, rng)
    if X_p is None:
        X_p = orthogonal_pilot_matrix(N, tau, rng)
    if noise_power_dbm is None:
        noise_power_dbm = cfg.noise_power_dbm

    sigma_n2 = dbm_to_watts(noise_power_dbm)

    Phi = np.diag(theta)  # (M, M)
    h_eff = np.zeros((K, N), dtype=np.complex128)
    for k in range(K):
        h_eff[k, :] = np.conj(H[:, k]) @ Phi @ G  # (N,)

    Y_clean = h_eff @ X_p  # (K, tau)

    noise = np.sqrt(sigma_n2 / 2) * (
        rng.standard_normal((K, tau)) + 1j * rng.standard_normal((K, tau))
    )
    Y_noisy = Y_clean + noise

    # emergent received SNR per user, for reporting/verification
    sig_power_per_user = np.mean(np.abs(Y_clean) ** 2, axis=1)  # (K,)
    received_snr_db = compute_received_snr_db(sig_power_per_user, sigma_n2)

    return {
        "theta": theta,
        "X_p": X_p,
        "h_eff": h_eff,        # ground truth cascaded channel (K, N) - for labels/checks only
        "Y_clean": Y_clean,    # noiseless received pilots (K, tau)
        "Y_noisy": Y_noisy,    # noisy received pilots (K, tau) - actual model input
        "sigma_n2": sigma_n2,
        "noise_power_dbm": noise_power_dbm,
        "received_snr_db": received_snr_db,  # (K,) emergent, measured quantity
    }


if __name__ == "__main__":
    from simulation.channel import generate_bs_irs_channel, generate_irs_user_channels

    cfg = SystemConfig()
    rng = np.random.default_rng(cfg.seed)
    G = generate_bs_irs_channel(cfg, rng)
    H, _ = generate_irs_user_channels(cfg, rng)
    out = receive_noisy_pilots(G, H, cfg, rng)
    print("h_eff shape:", out["h_eff"].shape)
    print("Y_noisy shape:", out["Y_noisy"].shape)
    print("sigma_n2:", out["sigma_n2"], "W  (noise floor:", out["noise_power_dbm"], "dBm)")
    print("emergent received SNR per user (dB):", np.round(out["received_snr_db"], 2))