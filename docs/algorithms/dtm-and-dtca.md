# Denoising Thermodynamic Models and Architecture (DTM / DTCA)

**Version**: 0.1.0  
**Status**: Draft  
**Last Updated**: April 2026

---

## Overview

Standard monolithic Energy-Based Models (EBMs) suffer from a fundamental **mixing-expressivity tradeoff**: as the model grows more expressive, its energy landscape becomes rougher, leading to slow relaxation and trapping in local minima. Denoising Thermodynamic Models (DTMs) resolve this by repurposing hardware EBMs as individual denoising steps rather than as a single monolithic model.

---

## 1. The Mixing-Expressivity Tradeoff

### 1.1 The Problem

A monolithic EBM defines a data distribution via a scalar energy function $E_\theta(x)$. For the distribution to be complex and expressive, the energy landscape must have many sharp wells — but this means Langevin dynamics takes a very long time to escape local minima and mix across the whole distribution.

This creates a hard tradeoff:
- **Low expressivity**: fast mixing, poor approximation
- **High expressivity**: accurate, but exponentially slow mixing

### 1.2 Why This Matters for TC Hardware

In physical TC hardware, the mixing time is a direct physical relaxation time. A slow-mixing model requires the hardware to equilibrate for a longer time before producing a useful sample — increasing latency and energy use.

---

## 2. Denoising Thermodynamic Models (DTMs)

### 2.1 Key Insight

DTMs borrow from **diffusion models**: instead of one complex EBM, use a *sequence* of many simple EBMs, each performing a small denoising step. Each hardware EBM:
1. Takes a partially-noisy input
2. Relaxes quickly to its shallow, simple energy landscape
3. Outputs a slightly-less-noisy state
4. Passes to the next step

Composition of many simple steps = complex generative model with fast mixing at each step.

### 2.2 Mathematical Formulation

DTM inference is a sequential Markov chain:

$$x_{t-1} = \text{TCStep}(x_t, t, \theta_t)$$

where each $\text{TCStep}$ is one Langevin equilibration on energy $V_{\theta_t}(x)$, parameterized for denoising step $t$.

The overall model defines a probability path:
$$p_0(\text{data}) \approx \prod_{t=1}^{T} p_{\theta_t}(x_{t-1} | x_t)$$

This is analogous to DDPM / score-matching diffusion, but each denoising step is executed by physical relaxation rather than a neural network forward pass.

---

## 3. Denoising Thermodynamic Computer Architecture (DTCA)

### 3.1 System Architecture

The DTCA integrates DTMs into a physical computing system:

```
[Noisy Input x_T]
      ↓
┌─────────────────────────────────────┐
│  DTCA: Array of Sampling Cores       │
│                                      │
│  ┌──────┐ ┌──────┐ ┌──────┐        │
│  │ TC   │ │ TC   │ │ TC   │  ...   │
│  │Step T│→│Step  │→│Step  │        │
│  │      │ │T-1   │ │T-2   │        │
│  └──────┘ └──────┘ └──────┘        │
│  (sparse locally connected           │
│   Boltzmann machines)                │
└─────────────────────────────────────┘
      ↓
[Clean Output x_0]
```

### 3.2 Key Properties

| Property | Value |
|---|---|
| Core type | Sparse, locally connected Boltzmann machines |
| Energy efficiency vs. GPU | $\sim 10^4 \times$ improvement (per Extropic AI / DTCA design) |
| Mechanism | Physical substrate equilibration replaces matrix multiplication |
| Avoided cost | Expensive matrix multiplications of digital score-matching |

### 3.3 Comparison with Standard Diffusion

| Aspect | Standard Diffusion (GPU) | DTM/DTCA (TC Hardware) |
|---|---|---|
| Each denoising step | Forward pass through neural net | Physical Langevin equilibration |
| Compute primitive | Float32 matrix multiply | Thermal noise + potential gradient |
| Energy per step | ~$10^{-12}$–$10^{-15}$ J | ~$k_B T \approx 10^{-21}$ J |
| Requires GPU VRAM | Yes (large activation tensors) | No |
| Parallel steps | Limited by memory bandwidth | Limited by physical relaxation time |

---

## 4. Relevance to Zae

### 4.1 Generative Model in neutral-consciousness-engine

The neutral-consciousness-engine includes a **Dream Engine** generative model. A TC-accelerated DTM could serve as the physical substrate for this generative model:
- Replace GPU-based diffusion with DTCA hardware
- Dramatically reduce power consumption for generative/predictive-coding inference
- Enable real-time generative modeling on orbital satellite hardware

### 4.2 Phase 2 Target

Implement a DTM simulation in JAX:
1. Simple Boltzmann machine denoising step
2. Sequential DTM inference chain
3. Benchmark against GPU softmax diffusion on a small dataset

---

## References

See [`../../reference/bibliography.md`](../../reference/bibliography.md):
- Jelinčič et al. (arXiv:2510.23972) — TSU hardware / DTCA design
- Extropic AI: "Thermodynamic Computing: From Zero to One" (2025)
