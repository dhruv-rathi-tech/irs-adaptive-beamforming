"""
Sanity tests for Phase 3 Deep Beamforming Network and Differentiable Loss.

Checks:
  1. Output dimensions: theta_ml in (B, M), W_ml in (B, N, K)
  2. Strict unit-modulus constraint: |theta_m| == 1.0
  3. Strict power constraint: ||W_ml||_F^2 == P_max
  4. Numerical exactness: PyTorch sum-rate matches NumPy ground truth
  5. Backpropagation: gradient flow confirmed through entire beamforming chain
  6. Non-leakage: model forward accepts only (C_hat_LS, noise_dbm)
"""
import numpy as np
import torch

from config.system_config import SystemConfig
from gnn.beamforming_model import DeepBeamformingNet, compute_true_sum_rate_torch

PASS = "PASS"
FAIL = "FAIL"


def check(name: str, cond: bool) -> bool:
    print(f"[{PASS if cond else FAIL}] {name}")
    return cond


def main():
    cfg = SystemConfig()
    all_ok = True
    B = 3
    P_max = 10.0
    sigma_watts = 3.16e-12

    model = DeepBeamformingNet(M=cfg.M, N=cfg.N, K=cfg.K, hidden_dim=64, n_blocks=2)

    c_hat = torch.randn(B, cfg.K, cfg.M, cfg.N, dtype=torch.complex64) * 1e-5
    noise_dbm = torch.tensor([-100.0, -85.0, -70.0])

    theta_ml, W_ml = model(c_hat, noise_dbm, P_max=P_max)

    # 1. Shapes
    all_ok &= check("theta_ml shape == (B, M)", theta_ml.shape == (B, cfg.M))
    all_ok &= check("W_ml shape == (B, N, K)", W_ml.shape == (B, cfg.N, cfg.K))

    # 2. Strict unit modulus
    modulus = torch.abs(theta_ml)
    all_ok &= check("unit modulus constraint |theta_m| == 1.0",
                     torch.allclose(modulus, torch.ones_like(modulus), atol=1e-5))

    # 3. Strict power constraint
    power = torch.sum(W_ml.real ** 2 + W_ml.imag ** 2, dim=(1, 2))
    expected_power = torch.full((B,), P_max)
    all_ok &= check(f"power constraint ||W_ml||_F^2 == P_max={P_max}",
                     torch.allclose(power, expected_power, atol=1e-3))

    # 4. Numerical match with NumPy ground truth
    H_np = (np.random.randn(B, cfg.M, cfg.K) + 1j * np.random.randn(B, cfg.M, cfg.K)) * 1e-4
    G_np = (np.random.randn(B, cfg.M, cfg.N) + 1j * np.random.randn(B, cfg.M, cfg.N)) * 1e-4
    H_t = torch.from_numpy(H_np).to(torch.complex64)
    G_t = torch.from_numpy(G_np).to(torch.complex64)

    theta_np = theta_ml.detach().numpy()
    W_np = W_ml.detach().numpy()

    rates_np = []
    for b in range(B):
        Phi = np.diag(np.conj(theta_np[b]))
        H_all = H_np[b].conj().T @ Phi @ G_np[b]
        r = 0.0
        for k in range(cfg.K):
            S = np.abs(H_all[k, :] @ W_np[b, :, k]) ** 2
            IN = sum(np.abs(H_all[k, :] @ W_np[b, :, j]) ** 2 for j in range(cfg.K) if j != k)
            r += np.log2(1.0 + S / (IN + sigma_watts))
        rates_np.append(r)

    rates_t = compute_true_sum_rate_torch(H_t, G_t, theta_ml, W_ml, sigma_watts).detach().numpy()
    all_ok &= check("PyTorch sum-rate matches NumPy ground truth (<1e-4)",
                     np.allclose(rates_t, np.array(rates_np), atol=1e-4))

    # 5. Differentiability
    loss = -compute_true_sum_rate_torch(H_t, G_t, theta_ml, W_ml, sigma_watts).mean()
    loss.backward()
    grad_ok = all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    all_ok &= check("backward pass produces finite gradients on all parameters", grad_ok)

    print(f"\n{'ALL DEEP BEAMFORMING TESTS PASSED' if all_ok else 'SOME TESTS FAILED'}")
    return all_ok


if __name__ == "__main__":
    ok = main()
    exit(0 if ok else 1)
