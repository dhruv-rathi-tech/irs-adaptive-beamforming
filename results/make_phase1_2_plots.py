"""
Verification-grade plots for Phase 1/2 presentation.
NOT performance claims -- these are single-realization sanity checks proving
the simulator behaves physically before Phase 4's proper Monte Carlo sweeps.
"""
import numpy as np
import matplotlib.pyplot as plt

# --- Plot 1: Path loss vs distance (Phase 1 channel model sanity check) ---
d = np.load('results/pathloss_curve.npz')
fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(d['d'], 10 * np.log10(d['pl_bi']), label='BS-IRS link (alpha=2.2, near-LOS)', linewidth=2)
ax.plot(d['d'], 10 * np.log10(d['pl_iu']), label='IRS-User link (alpha=3.5, NLoS-rich)', linewidth=2)
ax.set_xlabel('Distance (m)')
ax.set_ylabel('Path loss (dB)')
ax.set_title('Phase 1: Path-Loss Model Sanity Check\n(monotonic decay confirms correct log-distance model)')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('results/fig1_pathloss_sanity.png', dpi=150)
plt.close()

# --- Plot 2: C_hat recovery error vs noise floor (Phase 2 estimator sanity check) ---
d2 = np.load('results/recovery_vs_noise.npz')
fig, ax = plt.subplots(figsize=(7, 5))
ax.semilogy(d2['noise_floors'], d2['errs'], marker='o', linewidth=2, markersize=8, color='tab:red')
ax.axhline(1.0, color='gray', linestyle='--', alpha=0.5, label='100% error (unusable)')
ax.set_xlabel('Noise floor (dBm) — higher = worse noise')
ax.set_ylabel('C_hat mean relative recovery error (log scale)')
ax.set_title('Phase 2: Cascaded-Channel Estimator Degradation\n'
              '(single realization -- illustrative; Phase 4 averages many trials)', fontsize=11)
ax.legend()
ax.grid(True, alpha=0.3, which='both')
plt.tight_layout()
plt.savefig('results/fig2_recovery_vs_noise.png', dpi=150)
plt.close()

print("Saved results/fig1_pathloss_sanity.png")
print("Saved results/fig2_recovery_vs_noise.png")
