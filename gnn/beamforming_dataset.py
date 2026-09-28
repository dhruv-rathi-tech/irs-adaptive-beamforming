"""
Phase 3 -- Dataset for End-to-End Deep Beamforming.

Generates paired tuples:
  (C_hat_LS, noise_dbm, G_true, H_true)
using the exact simulation physics from simulation/channel.py and simulation/multi_phase_estimation.py.
Only C_hat_LS and noise_dbm are exposed to the model at inference.
G_true and H_true are used strictly during offline training to compute true sum-rate loss.
"""
from __future__ import annotations

import os
import torch
from torch.utils.data import Dataset
import numpy as np

from config.system_config import SystemConfig
from simulation.channel import generate_bs_irs_channel, generate_irs_user_channels
from simulation.multi_phase_estimation import multi_phase_pilot_estimate_C


class BeamformingDataset(Dataset):
    def __init__(self, cfg: SystemConfig, n_samples: int,
                 noise_dbm_range: tuple[float, float] = (-100.0, -70.0),
                 base_seed: int = 0, cache_file: str | None = None):
        self.cfg = cfg
        self.n_samples = n_samples
        self.noise_dbm_range = noise_dbm_range
        self.base_seed = base_seed

        if cache_file and os.path.exists(cache_file):
            print(f"Loading cached dataset from {cache_file} ...")
            data = torch.load(cache_file)
            self.c_hat_ls = data["c_hat_ls"]
            self.noise_dbm = data["noise_dbm"]
            self.G_true = data["G_true"]
            self.H_true = data["H_true"]
            return

        print(f"Generating {n_samples} beamforming realizations (seeds {base_seed} to {base_seed + n_samples - 1}) ...")
        c_hat_list = []
        noise_list = []
        G_list = []
        H_list = []

        for idx in range(n_samples):
            rng = np.random.default_rng(base_seed + idx)
            noise_val = float(rng.uniform(*noise_dbm_range))
            G = generate_bs_irs_channel(cfg, rng)
            H, _ = generate_irs_user_channels(cfg, rng)

            est = multi_phase_pilot_estimate_C(G, H, cfg, rng, noise_power_dbm=noise_val)
            c_hat_np = np.stack(est["C_hat"], axis=0)  # (K, M, N)

            c_hat_list.append(torch.from_numpy(c_hat_np).to(torch.complex64))
            noise_list.append(torch.tensor(noise_val, dtype=torch.float32))
            G_list.append(torch.from_numpy(G).to(torch.complex64))
            H_list.append(torch.from_numpy(H).to(torch.complex64))

        self.c_hat_ls = torch.stack(c_hat_list, dim=0)   # (S, K, M, N)
        self.noise_dbm = torch.stack(noise_list, dim=0)  # (S,)
        self.G_true = torch.stack(G_list, dim=0)         # (S, M, N)
        self.H_true = torch.stack(H_list, dim=0)         # (S, M, K)

        if cache_file:
            os.makedirs(os.path.dirname(cache_file), exist_ok=True)
            torch.save({
                "c_hat_ls": self.c_hat_ls,
                "noise_dbm": self.noise_dbm,
                "G_true": self.G_true,
                "H_true": self.H_true,
            }, cache_file)
            print(f"Saved dataset cache to {cache_file}")

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int):
        return (
            self.c_hat_ls[idx],
            self.noise_dbm[idx],
            self.G_true[idx],
            self.H_true[idx],
        )
