# Reproducible Latency Benchmark Report

**Date / Timestamp:** 2026-10-15 18:27:36  
**Platform:** Windows-11-10.0.26200-SP0  
**Processor:** Intel64 Family 6 Model 126 Stepping 5, GenuineIntel  
**PyTorch Version:** 2.14.0+cpu (CPU, 4 threads)  

---

## Latency Summary Table

| Pipeline Stage | Implementation | Sample Size | Mean | Std Dev | Median | 95th Percentile |
|---|---|:---:|:---:|:---:|:---:|:---:|
| **Pure ML Forward Inference** | `DeepBeamformingNet` (CPU) | 200 | **2.37 ms** | 0.77 ms | 2.25 ms | 3.62 ms |
| **Active RZF Matrix Inversion** | PyTorch Complex Batch Inv | 200 | **0.45 ms** | 0.18 ms | 0.38 ms | 0.77 ms |
| **Multi-Phase LS Estimation** | Multi-phase orthogonal pilots | 50 | **44.78 ms** | 5.32 ms | 43.22 ms | 52.64 ms |
| **Classical AO Optimization** | MATLAB CVX / SeDuMi SDP | 500 | **10.49 s** | 4.24 s | 9.36 s | 17.93 s |

---

## Acceleration Ratios

1. **Algorithmic Optimization Acceleration:**
   $$\text{Speedup} = \frac{10.49\text{ s}}{2.37\text{ ms}} = \mathbf{4430.1\times}$$
   *DeepBeamformingNet accelerates active-passive beamforming calculation by over 3,800-fold compared to classical alternating optimization.*

2. **Complete End-to-End Latency Acceleration:**
   - Deep Learning Pipeline: $44.78\text{ ms (LS)} + 2.37\text{ ms (ML)} = \mathbf{47.15\text{ ms}}$
   - Classical Pipeline: $44.78\text{ ms (LS)} + 10492.81\text{ ms (AO)} = \mathbf{10537.60\text{ ms}}$
   $$\text{End-to-End Speedup} = \frac{10537.60\text{ ms}}{47.15\text{ ms}} = \mathbf{223.5\times}$$

---

*Note: All measurements are algorithmic runtimes measured on identical CPU hardware. No claims of hardware circuit power or energy savings are made without physical power meter instrumentation.*
