"""
Phase 3 -- Training script for Deep Beamforming Network.

Optimizes achievable true-channel sum-rate directly via PyTorch autograd:
    Loss = - mean( TrueSumRate(H_true, G_true, theta_ML, W_ML) )

Validation evaluates TRUE SUM-RATE on held-out channel realizations.
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


def compute_baseline_rzf_rates(loader: DataLoader, sigma_watts: float, P_max: float) -> dict:
    """
    Compute conventional RZF baseline sum-rate on the exact same dataset using random IRS phases.
    """
    all_rates = []
    rng = np.random.default_rng(999)
    for c_hat_ls, noise_dbm, G_true, H_true in loader:
        B, K, M, N = c_hat_ls.shape
        # Random IRS phase for each item in batch
        rand_angles = rng.uniform(0, 2 * np.pi, size=(B, M))
        theta_rand = torch.from_numpy(np.exp(1j * rand_angles)).to(torch.complex64)

        # Standard RZF from C_hat_LS
        H_eff = torch.einsum("bm,bkmn->bkn", theta_rand, c_hat_ls)
        H_eff_H = H_eff.conj().transpose(-2, -1)
        G_eff = torch.bmm(H_eff, H_eff_H)
        I_K = torch.eye(K, dtype=torch.complex64).unsqueeze(0).expand(B, K, K)
        alpha = (K * sigma_watts / P_max)
        inv_reg = torch.linalg.inv(G_eff + alpha * I_K)
        W_unnorm = torch.bmm(H_eff_H, inv_reg)
        col_norms = torch.sqrt(torch.sum(W_unnorm.real ** 2 + W_unnorm.imag ** 2, dim=1, keepdim=True) + 1e-12)
        W_dir = W_unnorm / col_norms
        W_rzf = W_dir * np.sqrt(P_max / K)

        rates = compute_true_sum_rate_torch(H_true, G_true, theta_rand, W_rzf, sigma_watts)
        all_rates.append(rates)

    all_rates = torch.cat(all_rates)
    return {
        "mean_sum_rate": float(all_rates.mean().item()),
        "median_sum_rate": float(all_rates.median().item()),
        "std_sum_rate": float(all_rates.std().item()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--train_samples", type=int, default=400)
    parser.add_argument("--val_samples", type=int, default=80)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--n_blocks", type=int, default=2)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--out_dir", type=str, default="results/gnn")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cpu")

    cfg = SystemConfig()
    sigma_watts = 3.16e-12  # -85 dBm data noise
    P_max = cfg.pilot_power  # 10 W

    os.makedirs(args.out_dir, exist_ok=True)

    # 1. Datasets
    train_cache = os.path.join(args.out_dir, f"train_data_{args.train_samples}.pt")
    val_cache = os.path.join(args.out_dir, f"val_data_{args.val_samples}.pt")

    train_ds = BeamformingDataset(cfg, n_samples=args.train_samples,
                                  base_seed=1000, cache_file=train_cache)
    val_ds = BeamformingDataset(cfg, n_samples=args.val_samples,
                                base_seed=50000, cache_file=val_cache)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    # 2. Benchmark baseline on val set
    base_val = compute_baseline_rzf_rates(val_loader, sigma_watts, P_max)
    print(f"\n[Baseline Random-IRS + RZF] Validation True Sum-Rate: "
          f"{base_val['mean_sum_rate']:.4f} bps/Hz (median: {base_val['median_sum_rate']:.4f})\n")

    # 3. Model & Optimizer
    model = DeepBeamformingNet(M=cfg.M, N=cfg.N, K=cfg.K,
                               hidden_dim=args.hidden_dim, n_blocks=args.n_blocks).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=5)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"DeepBeamformingNet parameters: {n_params:,}")

    history = {
        "train_mean_rate": [],
        "val_mean_rate": [],
        "val_median_rate": [],
        "baseline_val_mean": base_val["mean_sum_rate"],
        "baseline_val_median": base_val["median_sum_rate"],
        "args": vars(args),
    }

    best_val_rate = -float("inf")
    patience_count = 0
    best_model_path = os.path.join(args.out_dir, "best_beamforming_model.pt")

    start_time = time.time()
    for epoch in range(1, args.epochs + 1):
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

        improved = val_mean > best_val_rate
        if improved:
            best_val_rate = val_mean
            patience_count = 0
            torch.save(model.state_dict(), best_model_path)
            mark = "(* BEST *)"
        else:
            patience_count += 1
            mark = ""

        print(f"Epoch {epoch:2d}/{args.epochs:2d} | "
              f"Train True Sum-Rate: {train_mean:6.3f} bps/Hz | "
              f"Val True Sum-Rate: {val_mean:6.3f} (med: {val_median:6.3f}) {mark}")

        if patience_count >= args.patience:
            print(f"Early stopping triggered at epoch {epoch}")
            break

    elapsed = time.time() - start_time
    history["total_time_sec"] = elapsed
    history["best_val_mean_rate"] = best_val_rate

    history_path = os.path.join(args.out_dir, "beamforming_training_history.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    print(f"\nTraining completed in {elapsed:.1f}s. Best Validation True Sum-Rate: {best_val_rate:.4f} bps/Hz")
    print(f"Model saved to {best_model_path}")
    print(f"History saved to {history_path}")


if __name__ == "__main__":
    main()
