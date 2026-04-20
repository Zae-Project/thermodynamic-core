# Scaling and Interconnects: Overcoming the Wiring Wall

**Version**: 0.1.0  
**Status**: Draft  
**Last Updated**: April 2026

---

## Overview

One of the most significant barriers to brain-scale thermodynamic computing is the **interconnect wall**: physical substrates are locally connected, but biological brains and modern AI architectures require complex non-local connectivity. This document covers the two primary scaling barriers and their solutions.

---

## 1. The Interconnect Wall

### 1.1 The Problem

In a brain-scale diffusion or U-Net model, skip connections carry information from encoder layers directly to decoder layers, bypassing intermediate processing. In hardware:

- A locally connected grid implementing a $D$-dimensional feature space requires $O(D^2)$ wiring for full skip connections
- For brain-scale systems with $D \gtrsim 10^4$, this becomes intractable
- Example: a 1,000-dimensional feature space requires ~1M wires *per skip connection*

### 1.2 Hierarchical Bilinear Skip Coupling (HBSC)

HBSC resolves this by exploiting the **singular structure** of trained model weight matrices (Whitelam & Casert, Nature Communications 2026 / arXiv:2604.14332).

Given encoder state $x \in \mathbb{R}^D$ and decoder state $y \in \mathbb{R}^D$, their joint potential is:

$$V(x,y) = V_{\text{enc}}(x) + V_{\text{dec}}(y) + x^\top J_{\text{skip}} y$$

The skip coupling matrix $J_{\text{skip}}$ is constructed as a **rank-$k$ approximation** using Singular Value Decomposition (SVD) of the encoder and decoder Gram matrices:

$$J_{\text{skip}}^{(k)} = \frac{1}{4k_B T} \cdot U_e^{(k)} (\Sigma_e^{(k)})^{1/2} \left( U_d^{(k)} (\Sigma_d^{(k)})^{1/2} \right)^\top$$

| Parameter | Meaning |
|---|---|
| $U_e^{(k)}, U_d^{(k)}$ | Top-$k$ singular vectors of encoder/decoder Gram matrices |
| $\Sigma_e^{(k)}, \Sigma_d^{(k)}$ | Top-$k$ singular values |
| $k$ | Rank of approximation (key design parameter) |

### 1.3 HBSC Performance at Rank k=16

Empirical results on trained U-Nets (Whitelam & Casert 2026):

| Metric | Value |
|---|---|
| Decoder cosine similarity vs. digital oracle | **0.9906** |
| Required physical connections | $O(Dk)$ instead of $O(D^2)$ |
| Theoretical net energy saving vs. GPU inference | $\sim 10^7 \times$ |
| Hardware implementation | Analog coupling bus (e.g., memristor conductances) |

At $k=16$, HBSC reduces wiring complexity from $O(D^2)$ to $O(Dk) = O(16D)$ while maintaining >99% functional fidelity — a practical path to brain-scale physical connectivity.

---

## 2. The Input Conditioning Barrier

### 2.1 The Problem: Eigenvalue Concentration

In a purely Langevin substrate with fixed coupling constants, the signal provided to distinguish one specific input from another can become too weak relative to thermal noise — an effect known as **eigenvalue concentration**.

In production-scale diffusion models, this signal deficit can be as high as **2,600×**. Without correction, the substrate equilibrates to roughly the same distribution regardless of the input: it has lost the ability to condition on input data.

### 2.2 The Solution: Minimal Digital Conditioning Interface

A **hybrid digital-analog architecture** resolves this with a minimal digital interface (Whitelam & Casert 2026):

```
[Input] → [Digital Conditioning Interface] → [Analog Langevin Substrate] → [Output]
            (<0.1% of model parameters)         (>99.9% of model)
```

The digital interface computes **dynamic bias vectors** $b_{\text{enc}}$ and $b_{\text{dec}}$ that anchor the analog substrate:

$$V_\theta(x) = V_{\text{enc}}(x) - b_{\text{enc}} \cdot x + V_{\text{dec}}(y) - b_{\text{dec}} \cdot y + x^\top J_{\text{skip}} y$$

**Components of the digital interface** (all small):
1. **Low-dimensional bottleneck encoder**: compresses input to a compact representation
2. **Transfer network**: maps compressed representation to bias vectors $b_{\text{enc}}, b_{\text{dec}}$

Total: a few thousand parameters guiding a million-parameter analog core.

### 2.3 Architecture Implication for Zae

For orbital deployment in arkspace-core satellites:
- The analog Langevin substrate (dominant: millions of p-bit cells) performs the heavy inference computation at near-zero marginal energy
- The digital conditioning interface (tiny: ~0.1% of parameters) runs on a small conventional processor and provides per-input anchoring
- This eliminates the need for a full digital replica of the model — only the bias computation is digital

---

## 3. Scaling Roadmap

| Scale | Neurons | Interconnects | Approach | Status |
|---|---|---|---|---|
| Proof-of-concept | ~1,000 | Local grid | Standard Boltzmann machine | Demonstrated |
| Mid-scale | ~100K | HBSC rank-16 | SVD-reduced skip couplings | Research |
| Brain-scale | ~100M | HBSC + hybrid conditioning | Full hybrid architecture | Speculative (TRL 2) |

---

## References

See [`../../reference/bibliography.md`](../../reference/bibliography.md):
- Whitelam & Casert (Nature Communications 2026) — HBSC — arXiv:2604.14332
- Whitelam (Molecular Foundry / Nat. Comm. 17:1189, 2026) — Non-linear TC mimicking neural nets
