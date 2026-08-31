"""
Multi-phase pilot protocol for estimating the CASCADED channel
    C_k = diag(conj(h_k)) @ G   in C^{M x N}   (per user k)
directly, instead of separately recovering H and G.

Derivation (see conversation record):
    h_eff_k = h_k^H diag(theta) G = theta^T C_k        (theta: (M,), C_k: (M,N))
Stacking over Q>=M distinct known IRS phases theta^(1..Q):
    H_all_stack_k = Theta_stack @ C_k      Theta_stack in C^{Q x M}, C_k in C^{M x N}
Solved by per-user least squares (batched over the N columns) -- no division
by any small channel entry, avoiding the noise blow-up of explicit H,G recovery.

This matches exactly the V{k,j} = diag(H(:,k)')*G construction already used
inside the reference Opt_func.m, so C_k plugs directly into the AO/SDR step.
"""
import numpy as np
from config.system_config import SystemConfig
from simulation.pilot import (
    random_irs_phase,
    orthogonal_pilot_matrix,
    dbm_to_watts,
    compute_received_snr_db,
)


def multi_phase_pilot_estimate_C(
    G: np.ndarray,
    H_true: np.ndarray,
    cfg: SystemConfig,
    rng: np.random.Generator,
    noise_power_dbm: float,
    Q: int | None = None,
) -> dict:
    """
    Estimate C_k = diag(conj(h_k)) @ G for all K users via multi-phase pilots.

    Returns dict with:
      C_hat        : list of K arrays, each (M, N) -- estimated cascaded channel per user
      C_true       : list of K arrays, each (M, N) -- ground truth (for verification only)
      thetas       : (Q, M) the phase configurations used (needed by AO to reconstruct H_all)
      tau_total    : int, total pilot symbols consumed (Q * tau)
      recovery_err : (K,) relative Frobenius error of C_hat vs C_true (verification only)
      avg_received_snr_db : (K,) emergent received SNR, for reporting
    """
    M, N, K = cfg.M, cfg.N, cfg.K
    tau = cfg.tau
    Q = M if Q is None else Q
    if Q < M:
        raise ValueError(f"Q={Q} must be >= M={M} for identifiability of C_k.")

    X_p = orthogonal_pilot_matrix(N, tau, rng)
    sigma_n2 = dbm_to_watts(noise_power_dbm)

    thetas = np.zeros((Q, M), dtype=np.complex128)
    h_eff_hat_stack = np.zeros((Q, K, N), dtype=np.complex128)
    sig_power_accum = np.zeros(K)

    C_true = [np.diag(np.conj(H_true[:, k])) @ G for k in range(K)]  # ground truth, for verification only

    for q in range(Q):
        theta_q = random_irs_phase(M, rng)
        thetas[q] = theta_q
        Phi = np.diag(theta_q)

        h_eff_true = np.zeros((K, N), dtype=np.complex128)
        for k in range(K):
            h_eff_true[k, :] = np.conj(H_true[:, k]) @ Phi @ G

        Y_clean = h_eff_true @ X_p
        noise = np.sqrt(sigma_n2 / 2) * (
            rng.standard_normal((K, tau)) + 1j * rng.standard_normal((K, tau))
        )
        Y_noisy = Y_clean + noise
        sig_power_accum += np.mean(np.abs(Y_clean) ** 2, axis=1)

        # LS estimate of cascaded channel for this phase block:
        # hhat_eff = (N/tau) * Y @ X_p^H   (since X_p X_p^H = (tau/N) I_N)
        h_eff_hat_stack[q] = (N / tau) * (Y_noisy @ X_p.conj().T)

    # Solve Theta_stack @ C_k = h_eff_hat_stack[:, k, :]  for each user k (all N columns jointly)
    C_hat = []
    recovery_err = np.zeros(K)
    for k in range(K):
        rhs = h_eff_hat_stack[:, k, :]  # (Q, N)
        C_k_hat, *_ = np.linalg.lstsq(thetas, rhs, rcond=None)  # (M, N)
        C_hat.append(C_k_hat)
        recovery_err[k] = np.linalg.norm(C_k_hat - C_true[k]) / np.linalg.norm(C_true[k])

    avg_received_snr_db = compute_received_snr_db(sig_power_accum / Q, sigma_n2)

    return {
        "C_hat": C_hat,
        "C_true": C_true,
        "thetas": thetas,
        "tau_total": Q * tau,
        "Q": Q,
        "recovery_err": recovery_err,
        "noise_power_dbm": noise_power_dbm,
        "avg_received_snr_db": avg_received_snr_db,
    }


if __name__ == "__main__":
    from simulation.channel import generate_bs_irs_channel, generate_irs_user_channels

    cfg = SystemConfig()
    rng = np.random.default_rng(cfg.seed)
    G = generate_bs_irs_channel(cfg, rng)
    H, _ = generate_irs_user_channels(cfg, rng)

    for noise_dbm in [-95.0, -85.0, -70.0]:
        out = multi_phase_pilot_estimate_C(G, H, cfg, rng, noise_power_dbm=noise_dbm)
        print(f"noise_floor={noise_dbm:6.1f} dBm | tau_total={out['tau_total']:4d} | "
              f"avg received SNR={np.round(out['avg_received_snr_db'], 1)} dB | "
              f"C_k relative error: {np.round(out['recovery_err'], 4)}")