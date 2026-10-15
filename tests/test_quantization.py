"""
Unit tests for Phase 5 IRS Phase Quantization (1-bit and 2-bit alphabets, STE, unit modulus).
"""
import numpy as np
import torch

from gnn.quantization import quantize_phase_ste, quantize_phase_numpy


def test_1bit_alphabet():
    # Random phases
    angles = torch.linspace(0, 2 * np.pi, 200)
    theta = torch.complex(torch.cos(angles), torch.sin(angles))
    q1 = quantize_phase_ste(theta, bits=1)

    # Check unit modulus
    assert torch.allclose(torch.abs(q1), torch.ones_like(q1.real), atol=1e-6)

    # Check exact alphabet {+1, -1}
    unique_vals = set(np.round(q1.detach().numpy(), 4))
    for val in unique_vals:
        assert val in {1.0 + 0j, -1.0 + 0j}, f"Unexpected 1-bit value: {val}"


def test_2bit_alphabet():
    angles = torch.linspace(0, 2 * np.pi, 200)
    theta = torch.complex(torch.cos(angles), torch.sin(angles))
    q2 = quantize_phase_ste(theta, bits=2)

    # Check unit modulus
    assert torch.allclose(torch.abs(q2), torch.ones_like(q2.real), atol=1e-6)

    # Check exact alphabet {+1, +j, -1, -j}
    expected = {1.0 + 0j, 0.0 + 1j, -1.0 + 0j, 0.0 - 1j}
    unique_vals = set(np.round(q2.detach().numpy(), 4))
    for val in unique_vals:
        assert val in expected, f"Unexpected 2-bit value: {val}"


def test_boundary_behavior():
    # Test specific angles near decision boundaries
    # 1-bit boundary is at pi/2 and 3pi/2
    th_plus1 = torch.tensor([np.exp(1j * 0.1), np.exp(1j * (0.5 * np.pi - 0.05)), np.exp(1j * (1.5 * np.pi + 0.05))], dtype=torch.complex64)
    th_minus1 = torch.tensor([np.exp(1j * (0.5 * np.pi + 0.05)), np.exp(1j * np.pi), np.exp(1j * (1.5 * np.pi - 0.05))], dtype=torch.complex64)

    q_plus = quantize_phase_ste(th_plus1, bits=1)
    q_minus = quantize_phase_ste(th_minus1, bits=1)

    assert torch.allclose(q_plus.real, torch.ones(3))
    assert torch.allclose(q_minus.real, -torch.ones(3))


def test_ste_gradient_flow():
    th = torch.randn(10, 64, dtype=torch.complex64)
    th = th / torch.abs(th)
    th.requires_grad = True

    # 1-bit STE
    q1 = quantize_phase_ste(th, bits=1)
    loss1 = q1.real.sum() + q1.imag.sum()
    loss1.backward()

    assert th.grad is not None
    assert torch.isfinite(th.grad).all()

    # Reset grads and test 2-bit STE
    th.grad.zero_()
    q2 = quantize_phase_ste(th, bits=2)
    loss2 = q2.real.sum() + q2.imag.sum()
    loss2.backward()

    assert th.grad is not None
    assert torch.isfinite(th.grad).all()


def test_numpy_torch_match():
    np_rng = np.random.default_rng(42)
    angles = np_rng.uniform(0, 2 * np.pi, 500)
    th_np = np.exp(1j * angles)
    th_torch = torch.from_numpy(th_np).to(torch.complex64)

    # 1-bit
    q1_np = quantize_phase_numpy(th_np, bits=1)
    q1_torch = quantize_phase_ste(th_torch, bits=1).detach().numpy()
    np.testing.assert_allclose(q1_np, q1_torch, atol=1e-5)

    # 2-bit
    q2_np = quantize_phase_numpy(th_np, bits=2)
    q2_torch = quantize_phase_ste(th_torch, bits=2).detach().numpy()
    np.testing.assert_allclose(q2_np, q2_torch, atol=1e-5)


if __name__ == "__main__":
    test_1bit_alphabet()
    print("[PASS] 1-bit alphabet strictly {+1, -1}")
    test_2bit_alphabet()
    print("[PASS] 2-bit alphabet strictly {+1, +j, -1, -j}")
    test_boundary_behavior()
    print("[PASS] Boundary behavior correct")
    test_ste_gradient_flow()
    print("[PASS] STE gradient flow verified")
    test_numpy_torch_match()
    print("[PASS] NumPy and PyTorch implementations match")
    print("\nALL QUANTIZATION UNIT TESTS PASSED!")
