# Hybrid Digital-Analog Conditioning Interface

**Version**: 0.1.0  
**Status**: Draft  
**Last Updated**: April 2026

---

## Overview

Production-scale thermodynamic inference requires a small **digital conditioning interface** to anchor the analog Langevin substrate to specific input data. Without it, eigenvalue concentration causes the analog substrate to lose input discriminability. This document specifies the architecture and interface contract for the hybrid digital-analog approach.

---

## 1. Problem Statement

### 1.1 Eigenvalue Concentration

In a large-scale analog Langevin substrate with fixed coupling constants $J_{ij}$, the coupling signal between input and output can be overwhelmed by thermal noise. The ratio of "signal eigenvalue" to "noise eigenvalue" diminishes as model size grows.

For production-scale diffusion models, this signal deficit reaches **up to 2,600×** — meaning the substrate's equilibrium distribution is nearly independent of the input. The hardware cannot condition on data.

### 1.2 Why This Happens

The Langevin dynamics pull the system toward the minimum of $V_\theta(x)$. If the bias contribution from the input (through fixed couplings $J_{\text{in}} \cdot x_{\text{input}}$) is weak compared to the internal coupling energy $\sum_{ij} J_{ij} x_i x_j$, the input signal is "drowned out."

---

## 2. The Hybrid Architecture

### 2.1 Architecture Overview

```
                     ┌─────────────────────────────────────────────────┐
                     │              HYBRID SYSTEM                       │
                     │                                                   │
[Input Data]         │  ┌──────────────────────┐                        │
   ↓                 │  │  DIGITAL INTERFACE   │                        │
   ├────────────────►│  │  (<0.1% of params)   │                        │
   │                 │  │                       │                        │
   │                 │  │  ┌─────────────────┐  │                        │
   │                 │  │  │ Bottleneck Enc  │  │  b_enc, b_dec          │
   │                 │  │  │ (low-dim rep.)  │  ├───────────────────────►│
   │                 │  │  └────────┬────────┘  │                        │
   │                 │  │           ↓            │                        │
   │                 │  │  ┌─────────────────┐  │      ┌──────────────┐ │
   │                 │  │  │ Transfer Network│  │      │   ANALOG     │ │
   │                 │  │  │ → b_enc, b_dec  │  │      │  SUBSTRATE   │ │
   │                 │  │  └─────────────────┘  │      │ (>99.9% of  │ │
   │                 │  └──────────────────────┘  │      │  params)     │ │
   │                 │                             │      │              │ │
   └────────────────────────────────────────────►│      │ Langevin     │ │
                     │                             │      │ + HBSC       │ │
                     │                             │      └──────┬───────┘ │
                     │                             │             ↓         │
                     │                             │          [Output]     │
                     └─────────────────────────────────────────────────────┘
```

### 2.2 Energy Landscape with Conditioning

The conditioned energy function is:

$$V_\theta(x, y \mid \text{input}) = V_{\text{enc}}(x) - b_{\text{enc}}(\text{input})^\top x + V_{\text{dec}}(y) - b_{\text{dec}}(\text{input})^\top y + x^\top J_{\text{skip}} y$$

The dynamic bias vectors $b_{\text{enc}}$ and $b_{\text{dec}}$ are computed by the digital interface from the input and injected into the analog substrate at inference time.

---

## 3. Component Specifications

### 3.1 Bottleneck Encoder

| Parameter | Value |
|---|---|
| Input dimension | Full input size $D$ |
| Bottleneck dimension | $d \ll D$ (e.g., 32–128) |
| Architecture | 1–2 linear layers + activation |
| Parameter count | $\sim D \cdot d$ (small) |
| Hardware | Runs on host digital processor |

### 3.2 Transfer Network

| Parameter | Value |
|---|---|
| Input | Bottleneck representation (dim $d$) |
| Output | Bias vectors $b_{\text{enc}} \in \mathbb{R}^D$, $b_{\text{dec}} \in \mathbb{R}^D$ |
| Architecture | 1–2 linear layers |
| Parameter count | $\sim 2 \cdot d \cdot D$ (small) |
| Hardware | Runs on host digital processor |

**Total digital interface parameters**: $\sim 3 \cdot d \cdot D$. For $D = 10^6$ and $d = 128$: ~384K parameters vs. the analog core's ~$10^9$ — approximately 0.04% of total.

### 3.3 Bias Injection Interface

The bias vectors must be injected into the analog substrate at the start of each inference call:

| Interface Property | Specification |
|---|---|
| Update frequency | Once per input sample |
| Bias precision | Float32 (digital) → DAC to analog |
| Injection latency | Must be << relaxation time $\tau$ |
| Communication bus | Dedicated high-bandwidth digital↔analog bus |

---

## 4. Training the Conditioning Interface

The digital conditioning interface is trained jointly with the analog coupling parameters $\theta$:

1. Fix analog parameters $\theta$ (or train jointly)
2. For each training sample $(x_{\text{input}}, x_{\text{target}})$:
   a. Forward pass through bottleneck encoder → transfer network → $b_{\text{enc}}, b_{\text{dec}}$
   b. Inject biases into Langevin simulator
   c. Run Langevin dynamics → sample $\hat{x}$
   d. Compute loss $\mathcal{L}(\hat{x}, x_{\text{target}})$
   e. Backprop through digital interface (and, using Onsager-Machlup, through analog too)

The small size of the digital interface means this backpropagation is cheap.

---

## 5. Interface with Other Pillars

- **neutral-consciousness-engine**: The SNN generative model's inference pipeline would route through this hybrid interface. See [`../integration/with-consciousness-engine.md`](../integration/with-consciousness-engine.md).
- **arkspace-core**: Digital interface runs on satellite node's conventional processor (ARM/RISC-V). Analog substrate is the TC payload. See [`../integration/with-arkspace.md`](../integration/with-arkspace.md).

---

## References

See [`../../reference/bibliography.md`](../../reference/bibliography.md):
- Whitelam & Casert (Nature Communications 2026 / arXiv:2604.14332) — HBSC + conditioning
