# Integration: thermodynamic-core ↔ neutral-consciousness-engine

**Version**: 0.1.0  
**Status**: Draft  
**Last Updated**: 2026-04-21

---

## Overview

This document defines the interface between **thermodynamic-core** (Substrate Layer) and **neutral-consciousness-engine** (Engine Layer). Thermodynamic hardware is the proposed physical substrate on which the SNN engine ultimately runs; this integration specifies how SNN primitives map to TC primitives.

---

## Interface Position

```
┌──────────────────────────────────────────────────────┐
│              neutral-consciousness-engine             │
│                                                       │
│  ┌─────────────────┐   ┌──────────────────────────┐  │
│  │  SNN Engine     │   │  Neural Firewall          │  │
│  │  (LIF neurons,  │   │  (TEE + AES-256)          │  │
│  │   STDP, ROS 2)  │   │                           │  │
│  └────────┬────────┘   └───────────────────────────┘  │
│           │ MODEL WEIGHTS + ACTIVATION TRAJECTORIES    │
└───────────┼──────────────────────────────────────────┘
            │
            ▼  [Onsager-Machlup Training / Hardware Programming]
┌───────────────────────────────────────────────────────┐
│              thermodynamic-core                        │
│                                                       │
│  ┌─────────────────────────────────────────────────┐  │
│  │  TC Substrate (p-bits / Langevin substrate)     │  │
│  │  + Hybrid Digital Conditioning Interface        │  │
│  │  + HBSC Skip Couplings                          │  │
│  └─────────────────────────────────────────────────┘  │
└───────────────────────────────────────────────────────┘
```

---

## 1. SNN → TC Primitive Mapping

| SNN Primitive (neutral-consciousness-engine) | TC Primitive (thermodynamic-core) |
|---|---|
| LIF neuron membrane potential | Continuous Langevin variable $x_i$ |
| Firing threshold | Energy barrier in $V_\theta(x_i)$ |
| Stochastic spiking | p-bit fluctuation (thermal noise) |
| Synaptic weight $w_{ij}$ | Coupling $J_{ij}$ in energy $V_\theta$ |
| Bias current | Bias $b_i$ in energy $V_\theta$ |
| STDP weight update | Onsager-Machlup gradient update $\Delta J_{ij}$ |
| Predictive coding prediction error | Residual term in conditioned energy |
| Generative model (Dream Engine) | DTM inference via DTCA hardware |

---

## 2. Training Interface

### 2.1 From SNN to TC Parameters

1. Train SNN in simulation using neutral-consciousness-engine (Nengo / PyTorch)
2. Record activation trajectories $\{x^{(0)}, \ldots, x^{(K)}\}$ for training inputs
3. Run Onsager-Machlup gradient descent to find TC coupling parameters $\theta = \{J_{ij}, b_i\}$
4. Validate: run Langevin simulator with learned $\theta$, check activation fidelity
5. Program $\theta$ onto physical TC hardware (TSU)

### 2.2 Teacher-Student Fidelity Metrics

| Metric | Acceptable Threshold | Notes |
|---|---|---|
| Activation cosine similarity | > 0.99 | Validated at rank-16 HBSC (Whitelam & Casert 2026) |
| Spike timing correlation | > 0.90 | Required for STDP-based learning |
| Generative model FID/KL | Match GPU baseline within 5% | For Dream Engine validation |

---

## 3. Runtime Interface

### 3.1 Inference Flow

```
[Biological Neural Input from brain-emulation]
         ↓
[Digital Conditioning Interface]
  - Encode input via bottleneck encoder
  - Compute bias vectors b_enc, b_dec
         ↓
[TC Substrate (TSU)]
  - Inject b_enc, b_dec
  - Run Langevin relaxation
  - Output: equilibrium sample = SNN activation pattern
         ↓
[neural-consciousness-engine Runtime]
  - Interpret TC output as SNN activations
  - Apply Neural Firewall checks
  - Route to appropriate SNN layer
```

### 3.2 Latency Budget

The neutral-consciousness-engine uses Libet's temporal buffer (~350 ms conscious awareness lag) to allow <50 ms satellite RTT. TC hardware relaxation time must fit within this budget:

| Component | Latency Budget |
|---|---|
| Digital conditioning compute | < 5 ms |
| TC substrate relaxation (one step) | < 20 ms |
| DTM inference chain (T steps) | < 50 ms total |
| OISL satellite round-trip | < 50 ms |
| Buffer available | ~200 ms |

---

## 4. Security Interface

The Neural Firewall in neutral-consciousness-engine must extend to the TC substrate:

| Security Requirement | Implementation |
|---|---|
| Parameter integrity | Cryptographic signing of $\theta$ at programming time |
| Inference channel encryption | AES-256 on digital conditioning bus |
| TC hardware attestation | TEE-style secure enclave for parameter storage |
| Brainjacking protection | Anomaly detection on TC output distribution |

---

## 5. Technology Gaps

| Gap | Description |
|---|---|
| TC hardware availability | TSU hardware not yet available; simulation-only for now |
| SNN-to-TC training pipeline | Onsager-Machlup training at SNN scale not yet demonstrated |
| Real time relaxation speed | Physical TSU relaxation latency not yet characterized |
| Neural Firewall on TC hardware | Security integration with stochastic hardware is novel |

---

## References

- See `neutral-consciousness-engine/` repository for SNN architecture details
- See [`../algorithms/langevin-training.md`](../algorithms/langevin-training.md) for Onsager-Machlup training
- See [`../protocols/hybrid-digital-analog.md`](../protocols/hybrid-digital-analog.md) for conditioning interface
- See [`../../reference/bibliography.md`](../../reference/bibliography.md) for full citations
