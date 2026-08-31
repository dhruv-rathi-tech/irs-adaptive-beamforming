"""
Channel generation for IRS-assisted system.

Two links:
  G : BS -> IRS channel, shape (M, N), complex.  Rician (LOS-dominant).
  H : IRS -> User channel, shape (M, K), complex. Rayleigh (NLoS-rich), one column per user.

Convention:
  - N = number of BS antennas
  - M = number of IRS elements
  - K = number of users
  - All channels are narrowband (single subcarrier), consistent with the
    project handout's scope (no OFDM / frequency-selectivity in Phase 1).
"""
import numpy as np
from config.system_config import SystemConfig


def path_loss_linear(d: np.ndarray, C0_dB: float, d0: float, alpha: float) -> np.ndarray:
    """
    Standard log-distance path loss model, returned in LINEAR scale (not dB).
    PL(d) = C0 * (d/d0)^(-alpha)   where C0 = 10^(C0_dB/10)
    d: distance(s) in meters (scalar or array)
    Returns: linear path loss (multiplicative power attenuation factor, <=1)
    """
    C0 = 10 ** (C0_dB / 10.0)
    return C0 * (d / d0) ** (-alpha)


def ula_steering_vector(n_elements: int, theta: float) -> np.ndarray:
    """
    Uniform Linear Array steering vector.
    theta: angle in [-0.5, 0.5], representing normalized spatial angle sin(phi)/2
           (standard convention: a(n) = exp(-j*pi*n*sin(phi)), here folded into theta)
    Returns: (n_elements,) complex unit-modulus vector.
    """
    n = np.arange(n_elements) - (n_elements - 1) / 2.0
    return np.exp(-2j * np.pi * theta * n)


def generate_bs_irs_channel(cfg: SystemConfig, rng: np.random.Generator) -> np.ndarray:
    """
    Generate BS-IRS channel G, shape (M, N), complex128.

    Rician model:
        G = sqrt(PL(d_BI)) * ( sqrt(K/(K+1)) * G_LOS + sqrt(1/(K+1)) * G_NLoS )

    G_LOS  = a_IRS(theta_aoa) @ a_BS(theta_aod)^H   (rank-1, deterministic angles, unit-modulus entries)
    G_NLoS ~ CN(0, 1) i.i.d. entries (rank M, scattering)

    Returns G with average per-entry power ≈ PL(d_BI) (so total channel "gain"
    scales correctly with distance/path-loss).
    """
    M, N = cfg.M, cfg.N
    K_rice = cfg.rician_K_BI

    # random but fixed-per-call AoA/AoD (single dominant LOS path)
    theta_aoa = rng.uniform(-0.5, 0.5)  # at IRS
    theta_aod = rng.uniform(-0.5, 0.5)  # at BS

    a_irs = ula_steering_vector(M, theta_aoa)  # (M,)
    a_bs = ula_steering_vector(N, theta_aod)   # (N,)
    G_LOS = np.outer(a_irs, np.conj(a_bs))     # (M, N), unit-modulus entries

    G_NLoS = (rng.standard_normal((M, N)) + 1j * rng.standard_normal((M, N))) / np.sqrt(2)

    pl = path_loss_linear(cfg.d_BI, cfg.C0_dB, cfg.d0, cfg.alpha_BI)
    gain_linear = 10 ** (cfg.element_gain_db / 20.0)  # amplitude gain (power gain = this^2)

    G = gain_linear * np.sqrt(pl) * (
        np.sqrt(K_rice / (K_rice + 1)) * G_LOS
        + np.sqrt(1 / (K_rice + 1)) * G_NLoS
    )
    return G.astype(np.complex128)


def generate_irs_user_channels(cfg: SystemConfig, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate IRS-User channels H, shape (M, K), complex128 — one Rayleigh column per user.
    Also returns per-user distances (K,) for reference/logging.

    H[:, k] = sqrt(PL(d_IU_k)) * g_k,   g_k ~ CN(0, I_M)
    """
    M, K = cfg.M, cfg.K
    d_IU = rng.uniform(cfg.d_IU_min, cfg.d_IU_max, size=K)

    g = (rng.standard_normal((M, K)) + 1j * rng.standard_normal((M, K))) / np.sqrt(2)
    pl = path_loss_linear(d_IU, cfg.C0_dB, cfg.d0, cfg.alpha_IU)  # (K,)
    gain_linear = 10 ** (cfg.element_gain_db / 20.0)

    H = gain_linear * g * np.sqrt(pl)[np.newaxis, :]  # broadcast per-column scaling
    return H.astype(np.complex128), d_IU


if __name__ == "__main__":
    cfg = SystemConfig()
    rng = np.random.default_rng(cfg.seed)
    G = generate_bs_irs_channel(cfg, rng)
    H, d_IU = generate_irs_user_channels(cfg, rng)
    print("G shape:", G.shape, "avg power:", np.mean(np.abs(G) ** 2))
    print("H shape:", H.shape, "avg power per user:", np.mean(np.abs(H) ** 2, axis=0))
    print("d_IU:", d_IU)