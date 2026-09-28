"""
Phase 3 sanity tests -- run after training to confirm the pipeline behaves
as documented. Not a full unit-test suite; checks the properties that
matter for research validity (no NaNs, residual connection wired
correctly, zero-baseline / LS-baseline sanity, no train/val leakage).

Run: python3 -m tests.test_phase3
"""
import numpy as np
import torch

from config.system_config import SystemConfig
from gnn.dataset import IRSGraphDataset, make_graph_sample, collate_user_graphs
from gnn.model import NoiseAwareGNN
from gnn.train import relative_frobenius_loss, per_sample_nmse

PASS = "PASS"
FAIL = "FAIL"


def check(name, cond):
    print(f"[{PASS if cond else FAIL}] {name}")
    return cond


def main():
    cfg = SystemConfig()
    all_ok = True

    # 1. Shapes and finiteness
    rng = np.random.default_rng(cfg.seed)
    nf, tg, snr, chat = make_graph_sample(cfg, rng, noise_power_dbm=-85.0)
    all_ok &= check("node_feats shape correct",
                     nf.shape == (cfg.K, cfg.M, 2 * cfg.M + 2 * cfg.N + 1))
    all_ok &= check("targets shape correct", tg.shape == (cfg.K, 2 * cfg.M * cfg.N))
    all_ok &= check("c_hat_ls shape matches targets", chat.shape == tg.shape)
    all_ok &= check("no NaN/Inf in node_feats", np.all(np.isfinite(nf)))
    all_ok &= check("no NaN/Inf in targets", np.all(np.isfinite(tg)))
    all_ok &= check("no NaN/Inf in c_hat_ls", np.all(np.isfinite(chat)))

    # 2. Zero-prediction baseline NMSE == 1.0 exactly (loss function correctness)
    zero_pred = torch.zeros(4, 2 * cfg.M * cfg.N)
    target_t = torch.from_numpy(tg)
    zero_nmse = relative_frobenius_loss(zero_pred, target_t).item()
    all_ok &= check(f"zero-baseline NMSE == 1.0 (got {zero_nmse:.6f})", abs(zero_nmse - 1.0) < 1e-5)

    # 3. Model residual connection: at init, output should stay close to c_hat_ls
    F = 2 * cfg.M + 2 * cfg.N + 1
    out_dim = 2 * cfg.M * cfg.N
    torch.manual_seed(0)
    model = NoiseAwareGNN(in_dim=F, out_dim=out_dim, hidden_dim=64, n_blocks=1, residual=True)
    feats_t = torch.from_numpy(nf)
    chat_t = torch.from_numpy(chat)
    with torch.no_grad():
        pred = model(feats_t, c_hat_ls=chat_t)
    diff = (pred - chat_t).abs().mean().item()
    all_ok &= check(f"residual connection near-identity at init (mean|diff|={diff:.4f} < 1.0)", diff < 1.0)
    all_ok &= check("model forward output finite", torch.isfinite(pred).all().item())

    # 4. Predicting c_hat_ls exactly should reproduce the LS NMSE exactly (sanity of per_sample_nmse)
    ls_nmse_direct = per_sample_nmse(chat_t, target_t)
    ls_nmse_manual = ((chat - tg) ** 2).sum(axis=1) / ((tg ** 2).sum(axis=1) + 1e-12)
    all_ok &= check("per_sample_nmse matches manual computation",
                     np.allclose(ls_nmse_direct.numpy(), ls_nmse_manual, atol=1e-4))

    # 5. Train/val seed disjointness (no data leakage) -- spot check a few samples
    train_ds = IRSGraphDataset(cfg, n_samples=5, base_seed=0)
    val_ds = IRSGraphDataset(cfg, n_samples=5, base_seed=10_000_000)
    train_targets = torch.cat([train_ds[i][1] for i in range(5)])
    val_targets = torch.cat([val_ds[i][1] for i in range(5)])
    # Different base seeds -> different channel realizations -> targets should differ substantially
    overlap = torch.isclose(train_targets[:cfg.K * 2], val_targets[:cfg.K * 2], atol=1e-6).all().item()
    all_ok &= check("train/val pools use disjoint seeds (no identical targets)", not overlap)

    # 6. Dataset pool reuse: __getitem__ returns the SAME sample every call (no leakage via re-randomization)
    ds = IRSGraphDataset(cfg, n_samples=3, base_seed=999)
    f1, t1, s1, c1 = ds[0]
    f2, t2, s2, c2 = ds[0]
    all_ok &= check("dataset pool returns identical sample on repeated access",
                     torch.equal(t1, t2) and torch.equal(f1, f2))

    print(f"\n{'ALL TESTS PASSED' if all_ok else 'SOME TESTS FAILED'}")
    return all_ok


if __name__ == "__main__":
    ok = main()
    exit(0 if ok else 1)
