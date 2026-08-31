"""
System configuration for Noise-Aware IRS Beamforming project.
All tunable parameters live here so later phases don't hardcode values.
"""
from dataclasses import dataclass


@dataclass
class SystemConfig:
    # --- BS antenna array ---
    N: int = 8          # number of BS antennas (ULA)

    # --- IRS array ---
    M: int = 64         # number of IRS elements (ULA for simplicity in Phase 1)

    # --- Users ---
    K: int = 4           # number of single-antenna users

    # --- Geometry (meters) ---
    d_BI: float = 50.0   # BS-IRS distance
    d_IU_min: float = 5.0   # min IRS-user distance
    d_IU_max: float = 30.0  # max IRS-user distance

    # --- Path loss model: PL(d) [linear] = C0 * (d/d0)^(-alpha) ---
    C0_dB: float = -30.0   # path loss at reference distance d0 (dB), typical mmWave value
    d0: float = 1.0        # reference distance (m)
    alpha_BI: float = 2.2  # path loss exponent, BS-IRS (near-LOS, engineered placement)
    alpha_IU: float = 3.5  # path loss exponent, IRS-user (more scattering/NLoS)

    # --- Rician K-factor for BS-IRS link (linear, not dB) ---
    rician_K_BI: float = 10.0   # LOS-dominant, since BS-IRS link is typically deployed with LOS

    # --- Antenna/element gain (linear amplitude), applied per hop ---
    # Matches reference baseline (BS_IRS.m / FF_User.m: lambda_gain ~ 9.6, ~19.6dB
    # power gain) representing directional antenna/element gain not captured by
    # the isotropic path-loss model above. Applied once per link (BS-IRS, IRS-user).
    element_gain_db: float = 19.6

    # --- Pilot transmission ---
    tau: int = 16          # pilot length (symbols per coherence block)
    T_coherence: int = 200  # coherence block length (symbols) -> tau/T is overhead
    pilot_power: float = 10.0  # linear pilot transmit power (W; 10W = 40dBm, realistic mmWave BS w/ PA)

    # --- Noise ---
    # FIXED absolute receiver noise floor (dBm), NOT relative to transmit power.
    # This matches standard practice and the reference MATLAB baseline
    # (sigma = 3.16e-12 W = -85 dBm). Physically: thermal noise -174 dBm/Hz +
    # noise figure + 10*log10(bandwidth); -85 dBm is a typical resulting value.
    # RECEIVED SNR (the quantity actually swept in Phase 4 plots) is an
    # EMERGENT result of this fixed noise floor combined with path loss/fading
    # -- it is NOT set directly. Use compute_received_snr_db() to measure it.
    noise_power_dbm: float = -85.0

    # --- Reproducibility ---
    seed: int = 42


DEFAULT_CONFIG = SystemConfig()