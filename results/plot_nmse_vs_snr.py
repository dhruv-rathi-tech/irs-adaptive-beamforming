"""
Plots Graph 1: Average Cascaded-Channel NMSE vs Received SNR, with a
shaded +-1 std confidence band. Reads nmse_vs_snr.npz (same folder).

Lives in results/. Run: python plot_nmse_vs_snr.py   (after generate_nmse_vs_snr.py)
"""
import os
import numpy as np
import matplotlib.pyplot as plt

here = os.path.dirname(__file__)
d = np.load(os.path.join(here, "nmse_vs_snr.npz"))
snr = d["snr_db"]
mean = d["nmse_db_mean"]
std = d["nmse_db_std"]

order = np.argsort(snr)
snr, mean, std = snr[order], mean[order], std[order]

plt.figure(figsize=(7, 5))
plt.plot(snr, mean, "o-", color="#1f77b4", linewidth=2, markersize=6, label="Mean NMSE")
plt.fill_between(snr, mean - std, mean + std, color="#1f77b4", alpha=0.2, label="±1 std")
plt.xlabel("Received SNR (dB)")
plt.ylabel("Cascaded-Channel NMSE (dB)")
plt.title("Average Cascaded-Channel NMSE vs Received SNR\n(50 realizations per point)")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
out_path = os.path.join(here, "fig1_nmse_vs_snr.png")
plt.savefig(out_path, dpi=200)
print(f"Saved {out_path}")
plt.show()