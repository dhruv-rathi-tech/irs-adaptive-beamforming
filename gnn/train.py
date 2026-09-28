"""
Phase 3 -- Training loop for the noise-aware GNN.

------------------------------------------------------------------
NOISE-AWARE TRAINING
------------------------------------------------------------------
`IRSGraphDataset` (gnn/dataset.py) draws a FRESH random noise floor
(uniform over SNR_DBM_LO..SNR_DBM_HI = -100..-70 dBm, matching the
Phase 2 sweep) for every sample, and a fresh random channel realization
too. So across an epoch the model sees many different SNR conditions
mixed together -- this is the "train across multiple noise levels
rather than one fixed noise condition" principle from the project
handout, implemented directly (no separate noise-injection module
needed since the physical pilot simulator already IS the noise model).

------------------------------------------------------------------
LOSS
------------------------------------------------------------------
Per-sample relative Frobenius error (linear-scale NMSE):
    ||pred - target||^2 / ||target||^2
averaged over the batch. This is the SAME metric definition used for
the Phase 2 LS baseline in results/generate_nmse_vs_snr.py, so GNN
and LS are directly comparable in Stage 4. (An earlier raw-MSE version
of this loss let large-magnitude examples dominate the gradient and
failed to beat a trivial zero-prediction baseline -- see Stage 3
report. This relative-error loss replaces it.)

------------------------------------------------------------------
TRAIN / VAL SPLIT
------------------------------------------------------------------
Both datasets generate on-the-fly with NO overlapping seeds
(train uses base_seed=0, val uses base_seed=10_000_000) -- this
guarantees the validation channels/noise realizations are never seen
during training (no test-time leakage).
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
from gnn.dataset import IRSGraphDataset, collate_user_graphs
from gnn.model import NoiseAwareGNN


def per_sample_nmse(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Per-sample relative Frobenius error (NMSE), NOT averaged -- shape (B,)."""
    num = ((pred - target) ** 2).sum(dim=1)   # (B,)
    den = (target ** 2).sum(dim=1) + eps      # (B,)
    return num / den


def relative_frobenius_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """
    Mean per-sample relative Frobenius error (NMSE). This is EXACTLY the
    NMSE definition already used for the LS baseline in
    results/generate_nmse_vs_snr.py (num/den, same squared-Frobenius-norm
    ratio, just per-sample here instead of aggregated across users).
    """
    return per_sample_nmse(pred, target, eps).mean()


def robust_nmse_loss(pred: torch.Tensor, target: torch.Tensor, clip: float = 5.0,
                      eps: float = 1e-12) -> torch.Tensor:
    """
    Mean per-sample NMSE, with each sample's contribution CLIPPED to at
    most `clip` before averaging.

    Why this is needed (found during Stage 3 debugging): at the noisiest
    end of the training range (~-70 dBm), the per-phase LS solve used to
    build C_hat_ls is occasionally severely ill-conditioned for a small
    fraction of random channel realizations, producing per-sample NMSE
    values as large as ~1400 (median at -70dBm is ~1.8, but the tail is
    heavy -- verified empirically). A handful of such outliers in a batch
    dominate a plain-mean loss and destabilize gradient descent for every
    other, well-behaved sample in that batch. Clipping each sample's loss
    contribution (not the gradient, not the data) is the standard, honest
    fix: it still trains on every sample, but prevents pathological
    individual cases from overwhelming the batch. clip=5.0 was chosen
    because it is well above the LS baseline's typical/median NMSE across
    the sweep (<=1 for the large majority of realizations, per the
    diagnostic script), so it only engages for genuine outliers, not for
    ordinary noisy samples.
    """
    per_sample = per_sample_nmse(pred, target, eps)
    return per_sample.clamp(max=clip).mean()


def run_epoch(model, loader, optimizer, device, train: bool, clip_loss: float = 5.0,
              grad_clip_norm: float = 1.0):
    """
    Trains using the ROBUST (outlier-clipped) loss. Reports BOTH the mean
    and the MEDIAN per-sample NMSE (unclipped, over every sample) so
    printed/logged numbers stay directly comparable to the Phase 2 LS-only
    NMSE metric.

    Why median is also reported (found during Stage 3 debugging): the
    per-sample NMSE distribution at high noise floors is heavily
    right-skewed -- a small fraction of channel realizations make the LS
    solve badly ill-conditioned, producing individual NMSE values in the
    hundreds or low thousands (verified empirically, see
    robust_nmse_loss docstring). A handful of such values swamp a MEAN
    computed over a few hundred samples, making the mean uninformative
    about "typical" performance even once training is genuinely working.
    The MEDIAN is far more representative of the typical case and is the
    metric used for the headline comparison; the mean is kept alongside
    it for transparency, not hidden.
    """
    model.train(mode=train)
    all_nmse = []
    for feats, targets, _snr, c_hat_ls in loader:
        feats = feats.to(device)
        targets = targets.to(device)
        c_hat_ls = c_hat_ls.to(device)

        if train:
            optimizer.zero_grad()
        pred = model(feats, c_hat_ls=c_hat_ls) if model.residual else model(feats)
        train_loss = robust_nmse_loss(pred, targets, clip=clip_loss)
        per_sample = per_sample_nmse(pred, targets).detach()  # unclipped, for reporting

        if train:
            train_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            optimizer.step()

        all_nmse.append(per_sample)

    all_nmse = torch.cat(all_nmse)
    return {
        "mean": all_nmse.mean().item(),
        "median": all_nmse.median().item(),
    }


def ls_baseline_nmse(loader, device):
    """Mean AND median NMSE of the raw LS estimate itself (no GNN),
    computed on the same loader/data for a reference point printed
    alongside training. See run_epoch's docstring for why median matters."""
    all_nmse = []
    for _feats, targets, _snr, c_hat_ls in loader:
        targets = targets.to(device)
        c_hat_ls = c_hat_ls.to(device)
        all_nmse.append(per_sample_nmse(c_hat_ls, targets).detach())
    all_nmse = torch.cat(all_nmse)
    return {"mean": all_nmse.mean().item(), "median": all_nmse.median().item()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--train_samples", type=int, default=600,
                         help="number of channel realizations, pre-generated ONCE and reused every epoch (each yields K=4 graphs)")
    parser.add_argument("--val_samples", type=int, default=120)
    parser.add_argument("--batch_size", type=int, default=16,
                         help="channel realizations per batch (=> batch_size*K graphs per step)")
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--clip_loss", type=float, default=5.0,
                         help="clip per-sample NMSE contribution to the TRAINING loss at this value (robustness to outlier LS failures at high noise)")
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--n_blocks", type=int, default=1)
    parser.add_argument("--patience", type=int, default=30,
                         help="stop if val loss doesn't improve for this many epochs")
    parser.add_argument("--out_dir", type=str, default="results/gnn")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--residual", action="store_true", default=True,
                         help="predict a correction on top of the LS estimate (default, recommended)")
    parser.add_argument("--no-residual", dest="residual", action="store_false",
                         help="predict C_k from scratch instead (kept for comparison; known to fail)")
    parser.add_argument("--noise_dbm_lo", type=float, default=-100.0)
    parser.add_argument("--noise_dbm_hi", type=float, default=-70.0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    cfg = SystemConfig()
    F = 2 * cfg.M + 2 * cfg.N + 1
    out_dim = 2 * cfg.M * cfg.N

    noise_range = (args.noise_dbm_lo, args.noise_dbm_hi)
    train_ds = IRSGraphDataset(cfg, n_samples=args.train_samples, noise_dbm_range=noise_range, base_seed=0)
    val_ds = IRSGraphDataset(cfg, n_samples=args.val_samples, noise_dbm_range=noise_range, base_seed=10_000_000)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                               collate_fn=collate_user_graphs)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                             collate_fn=collate_user_graphs)

    model = NoiseAwareGNN(in_dim=F, out_dim=out_dim, hidden_dim=args.hidden_dim,
                           n_blocks=args.n_blocks, residual=args.residual).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,} | residual mode: {args.residual} | "
          f"noise range: [{args.noise_dbm_lo}, {args.noise_dbm_hi}] dBm")

    ls_val = ls_baseline_nmse(val_loader, device)
    print(f"Reference: LS-only NMSE on this val set -- mean {ls_val['mean']:.4f} "
          f"({10*np.log10(ls_val['mean']):.2f} dB), median {ls_val['median']:.4f} "
          f"({10*np.log10(ls_val['median']):.2f} dB). GNN is judged primarily on MEDIAN "
          f"(mean is heavily skewed by rare ill-conditioned LS outliers at high noise -- see docs).")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    os.makedirs(args.out_dir, exist_ok=True)
    history = {"epoch": [], "train_loss": [], "val_loss": []}

    best_val_median = float("inf")
    best_val_mean = None
    epochs_since_improve = 0
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        train_stats = run_epoch(model, train_loader, optimizer, device, train=True, clip_loss=args.clip_loss)
        val_stats = run_epoch(model, val_loader, optimizer, device, train=False, clip_loss=args.clip_loss)

        history["epoch"].append(epoch)
        history["train_loss"].append(train_stats)
        history["val_loss"].append(val_stats)

        print(f"epoch {epoch:3d}/{args.epochs} | "
              f"train NMSE mean {train_stats['mean']:.4f} median {train_stats['median']:.4f} | "
              f"val NMSE mean {val_stats['mean']:.4f} median {val_stats['median']:.4f}")

        if val_stats["median"] < best_val_median - 1e-5:
            best_val_median = val_stats["median"]
            best_val_mean = val_stats["mean"]
            epochs_since_improve = 0
            torch.save(model.state_dict(), os.path.join(args.out_dir, "best_model.pt"))
        else:
            epochs_since_improve += 1
            if epochs_since_improve >= args.patience:
                print(f"Early stopping at epoch {epoch} (no val median-NMSE improvement for {args.patience} epochs).")
                break

    elapsed = time.time() - t0
    print(f"\nTraining complete in {elapsed:.1f}s.")
    print(f"Best val NMSE -- GNN: median {best_val_median:.4f} ({10*np.log10(best_val_median):.2f} dB), "
          f"mean {best_val_mean:.4f} ({10*np.log10(best_val_mean):.2f} dB)")
    print(f"                 LS:  median {ls_val['median']:.4f} ({10*np.log10(ls_val['median']):.2f} dB), "
          f"mean {ls_val['mean']:.4f} ({10*np.log10(ls_val['mean']):.2f} dB)")
    if best_val_median < ls_val["median"]:
        print(f"-> GNN IMPROVES on LS (median) by {10*np.log10(ls_val['median']/best_val_median):.2f} dB.")
    else:
        print(f"-> GNN does NOT beat LS (median) on this val set (worse by "
              f"{10*np.log10(best_val_median/ls_val['median']):.2f} dB).")

    with open(os.path.join(args.out_dir, "training_history.json"), "w") as f:
        json.dump({**history, "n_params": n_params, "elapsed_sec": elapsed,
                    "ls_val": ls_val, "best_val_median": best_val_median,
                    "best_val_mean": best_val_mean, "args": vars(args)}, f, indent=2)
    print(f"Saved training history to {args.out_dir}/training_history.json")
    print(f"Saved best model checkpoint to {args.out_dir}/best_model.pt")


if __name__ == "__main__":
    main()
