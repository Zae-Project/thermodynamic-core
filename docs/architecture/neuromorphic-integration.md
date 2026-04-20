# Neuromorphic Integration: Stochastic Spiking Neural Networks

**Version**: 0.1.0  
**Status**: Draft  
**Last Updated**: April 2026

---

## Overview

Neuromorphic engineering seeks to emulate the brain's massive parallelism and event-driven signaling. Integrating thermodynamic p-bits into neuromorphic architectures produces **Stochastic Spiking Neural Networks (SSNNs)** — systems that replicate the probabilistic computation found in biological neural circuits, while operating at thermodynamic efficiency.

---

## 1. The Neuristor: A CMOS-Compatible Stochastic Neuron

### 1.1 Background

A **neuristor** is a physical device that functions as an artificial neuron by exhibiting **leaky integrate-and-fire (LIF)** behavior. The challenge historically was implementing this in silicon without exotic materials.

Recent work (Nature Electronics 2025) demonstrated neuristors using conventional MOSFET technology via the **Single-Transistor Latch (STL)** mechanism.

### 1.2 The STL Mechanism

The STL exploits two operating regimes of the same device:

| Mode | Mechanism | Behavior | Network Layer |
|---|---|---|---|
| **Stochastic** | Impact ionization in floating body induces random spiking | Probabilistic spike generation | Input layers |
| **Deterministic** | Controlled charge accumulation triggers threshold crossing | Reliable LIF firing | Hidden/output layers |

This **dual-mode reconfigurability** simplifies circuit design significantly — the same neuristor can serve different roles across the network architecture.

### 1.3 Noise Resilience

Experimental validation on MNIST classification:
- Architecture: STL neuristor-based SSNN, single layer
- Accuracy: **92%** with **30% Gaussian noise** applied to inputs
- Hardware type: CMOS STL

This demonstrates a key advantage of stochastic architectures over deterministic ones: inherent noise resilience through probabilistic averaging rather than noise suppression.

---

## 2. Spike-Timing-Dependent Plasticity (STDP) in Thermodynamic Networks

### 2.1 STDP Basics

STDP is a biologically-grounded learning rule where synaptic weights change based on the relative timing of pre- and post-synaptic spikes:
- Pre-synaptic spike *before* post-synaptic spike → potentiation (weight increases)
- Pre-synaptic spike *after* post-synaptic spike → depression (weight decreases)

In the thermodynamic context, STDP can be interpreted as **noisy gradient descent** where query-key correlations are embedded in synaptic weights via Langevin dynamics.

### 2.2 The Spiking STDP Transformer (S²TDPT)

A major architectural advance is the **Spiking STDP Transformer** (arXiv:2511.14691), which replaces conventional dot-product attention with spike-timing interactions:

| Feature | Standard Transformer | Spiking STDP Transformer |
|---|---|---|
| Attention mechanism | Softmax(QKᵀ/√d) | Spike-timing correlation via STDP |
| Operations | Floating-point multiply-accumulate | Addition-only |
| Memory | Stores large intermediate attention matrices | In-memory computing — no attention matrix |
| Hardware target | GPU/TPU | Neuromorphic / TC hardware |
| Deployment | Von Neumann | Non-von Neumann |

The elimination of large intermediate attention matrices allows in-memory computing and direct deployment on neuromorphic hardware — critical for orbital applications where memory bandwidth is constrained.

---

## 3. Architecture of an SSNN

### 3.1 Network Layers

```
Input → [Stochastic Layer] → [Hidden Layers] → [Output Layer]
          (p-bit / STL        (Deterministic     (Stochastic
           stochastic mode)    LIF mode)           or det.)
```

### 3.2 Synaptic Implementation

For a network with $N$ neurons, synaptic weights $J_{ij}$ are implemented as:
- **Memristor crossbar arrays** — analog conductance maps to $J_{ij}$
- **Digital weight registers** — higher precision, higher power
- **Programmable bilinear couplings** — required for long-range skip connections (see [`scaling-interconnects.md`](scaling-interconnects.md))

### 3.3 Information Encoding

Stochastic SNNs use **rate coding** (spike frequency encodes value) and/or **temporal coding** (precise spike timing encodes information). The thermodynamic framework naturally accommodates both:
- Langevin relaxation time maps to rate-coded population dynamics
- Onsager-Machlup path statistics capture timing-coded trajectories

---

## 4. Brain-Scale SSNN: Scaling Targets

Targets from neutral-consciousness-engine and arkspace-core specifications:

| Parameter | Target | Current Demonstrated |
|---|---|---|
| Neuron count per node | 100M | ~1M (Intel Loihi 2, terrestrial) |
| Synapse count per node | ~10¹⁰ | ~10⁸ |
| Operating power | 50–200W | <10W (neuromorphic, terrestrial) |
| Radiation hardening | Required (LEO) | Not demonstrated |

Thermodynamic computing's energy efficiency is a prerequisite for reaching the 100M-neuron target within the satellite power budget.

---

## References

See [`../../reference/bibliography.md`](../../reference/bibliography.md):
- Nature Electronics (2025) — STL Neuristor / CMOS-compatible stochastic spiking neuron
- arXiv:2511.14691 — Spiking STDP Transformer (S²TDPT)
- Gao et al. (Micromachines 2025) — VCMA-MTJ neuromorphic architecture
