"""
Phase 5 -- Training Script for Quantization-Aware Deep Beamforming Network.

Trains DeepBeamformingNet with Straight-Through Estimator (STE) for discrete IRS phase constraints:
  - 1-bit: phase alphabet {0, pi} -> reflection coefficients {+1, -1}
  - 2-bit: phase alphabet {0, pi/2, pi, 3pi/2} -> reflection coefficients {+1, +j, -1, -j}

Uses identical training and validation channel realizations as the continuous baseline.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from config.system_config import SystemConfig
from gnn.beamforming_dataset import BeamformingDataset
from gnn.beamforming_model import DeepBeamformingNet, compute_true_sum_rate_torch


def evaluate_model(model: DeepBeamformingNet, loader: DataLoader,
                   sigma_watts: float, P_max: float, device: torch.device) -> dict:
    model.eval()
    all_rates = []
    with torch.no_grad():
        for c_hat_ls, noise_dbm, G_true, H_true in loader:
            c_hat_ls = c_hat_ls.to(device)
            noise_dbm = noise_dbm.to(device)
            G_true = G_true.to(device)
            H_true = H_true.to(device)

            theta_ml, W_ml = model(c_hat_ls, noise_dbm, P_max=P_max)
            rates = compute_true_sum_rate_torch(H_true, G_true, theta_ml, W_ml, sigma_watts)
            all_rates.append(rates.cpu())

    all_rates = torch.cat(all_rates)
    return {
        "mean_sum_rate": float(all_rates.mean().item()),
        "median_sum_rate": float(all_rates.median().item()),
        "std_sum_rate": float(all_rates.std().item()),
    }


def train_quantized_model(bits: int, epochs: int = 35, lr: float = 1e-3,
                          batch_size: int = 16, out_dir: str = "results/phase5_quantization",
                          seed: int = 42) -> str:
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cpu")

    cfg = SystemConfig()
    sigma_watts = 3.16e-12
    P_max = cfg.pilot_power

    os.makedirs(out_dir, exist_ok=True)

    # Re-use identical training/validation realization pools from results/gnn/
    train_cache = "results/gnn/train_data_300.pt"
    val_cache = "results/gnn/val_data_60.pt"

    train_ds = BeamformingDataset(cfg, n_samples=300, base_seed=1000, cache_file=train_cache)
    val_ds = BeamformingDataset(cfg, n_samples=60, base_seed=50000, cache_file=val_cache)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = DeepBeamformingNet(M=cfg.M, N=cfg.N, K=cfg.K, hidden_dim=64, n_blocks=2,
                               quant_bits=bits).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=5)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nTraining {bits}-bit QAT DeepBeamformingNet ({n_params:,} parameters, STE enabled) ...")

    best_val_rate = -float("inf")
    best_model_path = os.path.join(out_dir, f"model_{bits}bit_qat.pt")
    history = {
        "bits": bits,
        "n_params": n_params,
        "train_mean_rate": [],
        "val_mean_rate": [],
        "val_median_rate": [],
    }

    t0 = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        train_rates = []
        for c_hat_ls, noise_dbm, G_true, H_true in train_loader:
            c_hat_ls = c_hat_ls.to(device)
            noise_dbm = noise_dbm.to(device)
            G_true = G_true.to(device)
            H_true = H_true.to(device)

            optimizer.zero_grad()
            theta_ml, W_ml = model(c_hat_ls, noise_dbm, P_max=P_max)
            rates = compute_true_sum_rate_torch(H_true, G_true, theta_ml, W_ml, sigma_watts)
            loss = -rates.mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_rates.append(rates.detach().cpu())

        train_mean = float(torch.cat(train_rates).mean().item())
        val_metrics = evaluate_model(model, val_loader, sigma_watts, P_max, device)
        val_mean = val_metrics["mean_sum_rate"]
        val_median = val_metrics["median_sum_rate"]
        scheduler.step(val_mean)

        history["train_mean_rate"].append(train_mean)
        history["val_mean_rate"].append(val_mean)
        history["val_median_rate"].append(val_median)

        if val_mean > best_val_rate:
            best_val_rate = val_mean
            torch.save(model.state_dict(), best_model_path)
            mark = "(* BEST *)"
        else:
            mark = ""

        if epoch % 5 == 0 or epoch == epochs or mark:
            print(f"Epoch {epoch:2d}/{epochs:2d} | Train Rate: {train_mean:6.3f} | Val Rate: {val_mean:6.3f} (med: {val_median:6.3f}) {mark}")

    elapsed = time.time() - t0
    history["total_time_sec"] = elapsed
    history["best_val_mean_rate"] = best_val_rate

    history_path = os.path.join(out_dir, f"history_{bits}bit_qat.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    print(f"Completed {bits}-bit QAT in {elapsed:.1f}s. Best Val Rate: {best_val_rate:.4f} bps/Hz -> Saved to {best_model_path}\n")
    return best_model_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bits", type=int, choices=[1, 2], required=True)
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--out_dir", type=str, default="results/phase5_quantization")
    args = parser.parse_args()

    train_quantized_model(bits=args.bits, epochs=args.epochs, out_dir=args.out_dir)
