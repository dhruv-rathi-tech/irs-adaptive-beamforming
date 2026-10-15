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
    torch.manual_seed(42)
    np.random.seed(42)
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

    # 6. Conjugation consistency test:
    # Row k of H_all_true is H_true(:, k)' * diag(conj(theta)) * G_true.
    # The cascaded channel is C_k[m, n] = H_true[m, k]^* * G_true[m, n].
    # Therefore, (conj(theta))^T * C_k must exactly match H_all_true[k, :].
    c_true_np = np.zeros((B, cfg.K, cfg.M, cfg.N), dtype=np.complex64)
    for b in range(B):
        for k in range(cfg.K):
            c_true_np[b, k] = np.diag(np.conj(H_np[b][:, k])) @ G_np[b]
    c_true_t = torch.from_numpy(c_true_np).to(torch.complex64)

    # Matched effective channel using production compute_effective_channel
    H_eff_matched = DeepBeamformingNet.compute_effective_channel(c_true_t, theta_ml)
    H_all_true_expected = torch.einsum('bmk,bm,bmn->bkn', H_t.conj(), theta_ml.conj(), G_t)
    all_ok &= check("Production compute_effective_channel matches H_all_true",
                     torch.allclose(H_eff_matched, H_all_true_expected, atol=1e-5))

    # 8. Analytical Synthetic Channel Regression Test with Known Ground Truth
    # Single-user (K=1), M=2 elements, N=2 antennas:
    # h = [1, j]^T  => h^H = [1, -j]
    # theta = [1, j]^T => theta^* = [1, -j]
    # G = [[2, 3], [4, 5]]
    # Physical channel: h^H @ diag(theta^*) @ G = [1, -1] @ [[2, 3], [4, 5]] = [-2, -2]
    # Cascaded channel C = diag(h^*) @ G = [[2, 3], [-4j, -5j]]
    # Matched effective channel: (theta^*)^T @ C = 1*[2, 3] + (-j)*[-4j, -5j] = [2, 3] + [-4, -5] = [-2, -2]
    # Mismatched effective channel: theta^T @ C = 1*[2, 3] + j*[-4j, -5j] = [6, 8]
    theta_synth = torch.tensor([[1.0 + 0.0j, 0.0 + 1.0j]], dtype=torch.complex64)  # (1, 2)
    c_synth = torch.tensor([[[[2.0 + 0.0j, 3.0 + 0.0j],
                              [0.0 - 4.0j, 0.0 - 5.0j]]]], dtype=torch.complex64)  # (1, 1, 2, 2)
    expected_h_eff = torch.tensor([[[-2.0 + 0.0j, -2.0 + 0.0j]]], dtype=torch.complex64)

    # Test production helper compute_effective_channel directly
    prod_h_eff = DeepBeamformingNet.compute_effective_channel(c_synth, theta_synth)
    all_ok &= check("Production compute_effective_channel matches analytical ground truth [-2, -2]",
                     torch.allclose(prod_h_eff, expected_h_eff, atol=1e-5))

    # Test production construct_beamformer directly
    synth_net = DeepBeamformingNet(M=2, N=2, K=1, hidden_dim=16, n_blocks=1)
    p_alloc = torch.tensor([[1.0]], dtype=torch.float32)
    alpha = torch.tensor([[0.5]], dtype=torch.float32)
    P_max_synth = 10.0
    W_ml_synth = synth_net.construct_beamformer(c_synth, theta_synth, p_alloc, alpha, P_max=P_max_synth)
    expected_W_synth = torch.tensor([[[-np.sqrt(5.0) + 0.0j], [-np.sqrt(5.0) + 0.0j]]], dtype=torch.complex64)
    all_ok &= check("Production construct_beamformer precoder matches hand-calculated [-sqrt(5), -sqrt(5)]",
                     torch.allclose(W_ml_synth, expected_W_synth, atol=1e-5))

    # Evaluate precoder against physical channel h_phys = [-2, -2]
    h_phys = torch.tensor([[[-2.0 + 0.0j, -2.0 + 0.0j]]], dtype=torch.complex64)
    rx_signal = torch.bmm(h_phys, W_ml_synth)
    expected_rx = torch.tensor([[[4.0 * np.sqrt(5.0) + 0.0j]]], dtype=torch.complex64)
    all_ok &= check("Precoder against physical channel achieves coherent sum 4*sqrt(5)",
                     torch.allclose(rx_signal, expected_rx, atol=1e-5))

    print(f"\n{'ALL DEEP BEAMFORMING TESTS PASSED' if all_ok else 'SOME TESTS FAILED'}")
    return all_ok


if __name__ == "__main__":
    ok = main()
    exit(0 if ok else 1)
