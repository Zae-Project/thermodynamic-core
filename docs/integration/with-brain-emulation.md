# Integration: thermodynamic-core ↔ brain-emulation

**Version**: 0.1.0  
**Status**: Draft  
**Last Updated**: April 2026

---

## Overview

This document describes the interface between **thermodynamic-core** (Substrate Layer) and **brain-emulation** (Interface Layer). The relationship is primarily scientific and architectural: brain-emulation models biological neural dynamics with Brian2 (deterministic simulations), while thermodynamic-core defines the stochastic physical substrate. Understanding where these paradigms diverge and where they converge is essential for the Zae project's coherence.

---

## 1. Paradigm Comparison

| Aspect | brain-emulation (Brian2) | thermodynamic-core (TC) |
|---|---|---|
| Neuron model | Biologically realistic (HH, LIF, etc.) | Langevin degrees of freedom |
| Dynamics | Deterministic ODEs (optionally noisy) | Inherently stochastic Langevin SDE |
| Synapse model | Conductance-based, STDP | Energy coupling $J_{ij}$ |
| Learning | STDP, homeostatic plasticity | Onsager-Machlup gradient descent |
| Purpose | Biological fidelity, atlas-based templates | Energy efficiency, physical substrate |
| Hardware target | CPU/GPU simulation | TSU (thermodynamic sampling unit) |
| Current status | v1.0 feature-complete | TRL 2 research |

### 1.1 Stochastic vs. Deterministic Neurons

Brian2 LIF neurons are deterministic with optional added Gaussian noise. TC p-bit neurons are *inherently* stochastic — their fluctuations are the computation, not a nuisance. The STL neuristor (see [`../architecture/neuromorphic-integration.md`](../architecture/neuromorphic-integration.md)) bridges this gap: it operates in stochastic mode at the input layer and deterministic mode at hidden/output layers, matching Brian2's architecture in spirit while running on TC hardware.

---

## 2. Brain Atlas → TC Energy Landscape

### 2.1 Atlas-Based Templates in brain-emulation

brain-emulation provides 5 brain region templates based on:
- Allen Brain Atlas (Motor Cortex, Prefrontal, Somatosensory)
- BrainGlobe (Visual Cortex)
- Julich-Brain / EBRAINS siibra (Thalamocortical)

Each template defines:
- Network topology (80% excitatory / 20% inhibitory in Realistic mode)
- Neuron types (30+ types with distinct firing patterns)
- Connectivity statistics (clustered, laminar)

### 2.2 Mapping to TC Energy Function

The brain-emulation network topology can be directly mapped to a TC energy landscape:

| Brian2 Object | TC Equivalent |
|---|---|
| Excitatory synapse weight $w_{ij} > 0$ | Negative coupling $J_{ij} < 0$ (lowers energy for aligned states) |
| Inhibitory synapse weight $w_{ij} < 0$ | Positive coupling $J_{ij} > 0$ (raises energy for aligned states) |
| Neuron bias current $I_{\text{bias}}$ | Bias term $b_i$ in $V_\theta$ |
| 80/20 exc/inh ratio | Ratio of negative to positive $J_{ij}$ couplings |
| Clustered connectivity | Local vs. long-range couplings (HBSC for long-range) |

### 2.3 Translation Pipeline (Future Work)

A brain-emulation → TC parameter translation tool (Phase 2 target):

```python
# Conceptual — Phase 2
from brain_emulation import BrainTemplate
from thermodynamic_core import CouplingMatrix

template = BrainTemplate.load("allen_motor_cortex")
J, b = CouplingMatrix.from_brian2_network(template.network)
# → J[i,j] = -w[i,j] (excitatory) or +|w[i,j]| (inhibitory)
# → b[i] = I_bias[i]
```

---

## 3. Visualization Interface

brain-emulation provides Three.js 3D spiking activity visualization. TC hardware outputs are equilibrium samples (populations of p-bit states), not discrete spike trains. Translation:

| TC Output | brain-emulation Visualization |
|---|---|
| $x_i \in [-1, +1]$ (Ising variable) | Map to spike rate: $r_i = (x_i + 1)/2 \times r_{\max}$ |
| Boltzmann sample (binary) | Direct spike event per positive $x_i = +1$ |
| Langevin trajectory (continuous) | Thresholded spike train |

The existing WebSocket real-time streaming interface in brain-emulation can carry TC output samples formatted as synthetic spike trains.

---

## 4. Shared Validation Benchmarks

Both projects should use common benchmarks for cross-validation:

| Benchmark | brain-emulation | thermodynamic-core |
|---|---|---|
| MNIST digit recognition | Brian2 SNN classifier | TC Langevin / DTCA classifier |
| Allen Motor Cortex activity | Biologically realistic simulation | TC energy landscape sampling |
| Split-brain protocol | Hemispheric independence test | TC dual-substrate simulation |

Shared benchmarks allow direct comparison of biological-fidelity (Brian2) vs. thermodynamic-efficiency (TC) implementations.

---

## 5. Technology Gaps

| Gap | Description |
|---|---|
| Atlas → TC mapping | No tool exists yet to convert Brian2 network to TC coupling matrix |
| Spike train ↔ TC output | Conversion protocol not yet specified beyond conceptual |
| Temporal resolution | TC Langevin timestep vs. Brian2 dt must be reconciled |
| Biological fidelity in TC | TC Boltzmann machines are simpler than conductance-based neuron models |

---

## References

- See `brain-emulation/` repository for Brian2 network details and atlas templates
- See [`../architecture/neuromorphic-integration.md`](../architecture/neuromorphic-integration.md) for STL neuristor
- See [`../../reference/bibliography.md`](../../reference/bibliography.md) for full citations
