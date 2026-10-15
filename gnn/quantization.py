"""
Phase 5 -- Differentiable IRS Phase Quantization & Straight-Through Estimator (STE).

Permitted Phase Alphabets:
  1-bit IRS (bits=1):
    Phases: {0, pi}
    Reflection coefficients: {+1, -1}
  2-bit IRS (bits=2):
    Phases: {0, pi/2, pi, 3*pi/2}
    Reflection coefficients: {+1, +j, -1, -j}

Straight-Through Estimator (STE):
  Forward pass: discrete quantized phases exactly satisfying hardware constraint.
  Backward pass: gradients pass through continuously to unquantized pre-activations.
"""
from __future__ import annotations

import numpy as np
import torch


def quantize_phase_ste(theta: torch.Tensor, bits: int | None) -> torch.Tensor:
    """
    Quantize complex unit-modulus phase vector theta to discrete alphabet using STE.
    
    Args:
      theta: complex tensor of shape (..., M) with |theta_m| == 1.
      bits: 1, 2, or None (continuous).
      
    Returns:
      theta_q: complex tensor with entries strictly in the discrete alphabet.
    """
    if bits is None or bits <= 0:
        return theta

    # Extract continuous phase in [0, 2*pi)
    phi = torch.atan2(theta.imag, theta.real)
    phi = torch.remainder(phi, 2.0 * np.pi)

    n_levels = 2 ** bits
    delta = 2.0 * np.pi / n_levels

    # Nearest discrete phase bin
    k = torch.round(phi / delta)
    phi_q = torch.remainder(k * delta, 2.0 * np.pi)

    # Exact discrete complex reflection coefficient
    q_real = torch.cos(phi_q)
    q_imag = torch.sin(phi_q)

    # Snap tiny numerical noise (e.g. sin(pi) ~ 1e-16) to exact integer values for 1-bit / 2-bit
    if bits == 1:
        q_real = torch.where(q_real >= 0.0, torch.tensor(1.0, device=theta.device, dtype=q_real.dtype),
                             torch.tensor(-1.0, device=theta.device, dtype=q_real.dtype))
        q_imag = torch.zeros_like(q_imag)
    elif bits == 2:
        # Snap real and imag to {-1, 0, 1}
        q_real = torch.round(q_real)
        q_imag = torch.round(q_imag)

    theta_discrete = torch.complex(q_real, q_imag)

    # Straight-Through Estimator: forward is theta_discrete, backward flows to theta
    return theta + (theta_discrete - theta).detach()


def quantize_phase_numpy(theta: np.ndarray, bits: int | None) -> np.ndarray:
    """
    NumPy phase quantization for evaluation and baseline comparisons.
    """
    if bits is None or bits <= 0:
        return theta / np.abs(theta)

    phi = np.angle(theta)
    phi = np.remainder(phi, 2.0 * np.pi)

    n_levels = 2 ** bits
    delta = 2.0 * np.pi / n_levels

    k = np.round(phi / delta)
    phi_q = np.remainder(k * delta, 2.0 * np.pi)

    if bits == 1:
        # {+1, -1}
        real_part = np.where(np.cos(phi_q) >= 0.0, 1.0, -1.0)
        return real_part.astype(np.complex128)
    elif bits == 2:
        # {+1, +j, -1, -j}
        real_part = np.round(np.cos(phi_q))
        imag_part = np.round(np.sin(phi_q))
        return (real_part + 1j * imag_part).astype(np.complex128)
    else:
        return (np.cos(phi_q) + 1j * np.sin(phi_q)).astype(np.complex128)
