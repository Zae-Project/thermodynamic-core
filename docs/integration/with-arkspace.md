# Integration: thermodynamic-core ↔ arkspace-core

**Version**: 0.1.0  
**Status**: Draft  
**Last Updated**: 2026-04-21

---

## Overview

This document defines the integration between **thermodynamic-core** (Substrate Layer) and **arkspace-core** (Infrastructure Layer). Thermodynamic computing is the proposed physical payload technology for the Exocortex Constellation satellites. It is the hardware paradigm that makes a 100M-neuron satellite node at 50–200W conceivable.

---

## 1. Why TC for Orbital Deployment

### 1.1 The Power Budget Problem

From `arkspace-core` specifications:
- Target: 100M neurons per satellite node
- Power budget: 50–200W per payload
- Current neuromorphic reference: Intel Loihi 2 at ~1M neurons / ~1W (terrestrial, unverified for space)

Scaling from Loihi 2 to 100M neurons at the same power density would require ~100W, at the boundary of the stated budget, before accounting for radiation-hardening overhead, thermal management in LEO, and the ~4× power scaling gap that space environments introduce.

Thermodynamic computing's $10^3$–$10^7 \times$ energy efficiency over digital accelerators is the principal reason TC is relevant to this orbital application. It could keep the 100M-neuron payload within a viable power envelope.

### 1.2 TC vs. Loihi-Class in LEO

| Parameter | Loihi-class Neuromorphic | TC (TSU-based) | Notes |
|---|---|---|---|
| Energy per synaptic op | ~10 pJ | ~$k_B T \sim 4 \times 10^{-21}$ J (theoretical) | 9 orders of magnitude difference |
| Radiation hardening | Not demonstrated | Not demonstrated | Both at TRL 2 for space |
| CMOS compatibility | Yes (14 nm Intel) | Yes (all-transistor approach) | TC has near-term fab path |
| Non-volatility | Some SRAM-based | Volatile (subthreshold CMOS) | MTJ p-bits are non-volatile |

---

## 2. Satellite Payload Architecture (Proposed)

### 2.1 TC Payload Block

```
┌─────────────────────────────────────────────────────────────────┐
│                     SATELLITE NODE (arkspace-core)               │
│                                                                   │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │                    TC PAYLOAD                                │ │
│  │                                                              │ │
│  │  ┌────────────────────┐    ┌──────────────────────────────┐ │ │
│  │  │  TSU Array          │    │  Digital Conditioning        │ │ │
│  │  │  (p-bit sampling   │    │  Processor (ARM/RISC-V)      │ │ │
│  │  │   cells)           │◄───│  - Bottleneck encoder        │ │ │
│  │  │                    │    │  - Transfer network          │ │ │
│  │  │  Langevin dynamics │    │  - Bias vector injection     │ │ │
│  │  │  + HBSC couplings  │    └──────────────────────────────┘ │ │
│  │  └─────────┬──────────┘                                       │ │
│  │            │                                                   │ │
│  │  ┌─────────▼──────────┐                                       │ │
│  │  │  Parameter Memory  │  (J_ij, b_i, programmed at launch     │ │
│  │  │  (non-volatile)    │   or via uplink update)               │ │
│  │  └────────────────────┘                                       │ │
│  └─────────────────────────────────────────────────────────────┘ │
│                                                                   │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │  OISL Router / Ground Link (existing arkspace-core spec)     │ │
│  └─────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 Parameter Update via Uplink

TC coupling parameters $\theta = \{J_{ij}, b_i\}$ can be updated via satellite uplink:
- Ground station computes updated $\theta$ using Onsager-Machlup training on latest data
- Encrypted parameter upload via Ka-band / optical uplink (existing arkspace spec)
- On-board non-volatile parameter memory (flash / embedded MRAM) stores current $\theta$

---

## 3. Radiation Tolerance Requirements

LEO orbital environment imposes radiation stress that is a critical technology gap:

| Threat | Description | Impact on TC Hardware |
|---|---|---|
| **Single-Event Upsets (SEU)** | Ionizing particle flips a bit | Flips p-bit state, potentially tolerable (substrate is stochastic anyway) |
| **Total Ionizing Dose (TID)** | Accumulated radiation shifts transistor thresholds | Alters subthreshold CMOS behavior → calibration drift |
| **Displacement Damage** | Lattice defects from neutrons | Degrades carrier mobility → changes $\mu$ in Langevin |
| **Thermal cycling** | -40°C to +80°C per orbit | Changes $k_B T$ → changes fluctuation amplitude → $\Delta$ shift in MTJ |

### 3.1 TC-Specific Advantage for SEU

In deterministic logic a single bit flip is a hard error. Thermodynamic hardware is probabilistic by construction. A random state perturbation from cosmic radiation is another thermal fluctuation and the system re-equilibrates. TC hardware therefore may have **intrinsic SEU tolerance** that deterministic neuromorphic hardware lacks.

> **Speculative**. Formal SEU tolerance analysis for TC hardware in LEO is a required research task.

### 3.2 TID and Thermal Mitigation

- **TID shielding**: Spot shielding of critical TC cores with tungsten or polyethylene
- **Thermal stabilization**: Active thermal control to hold TC payload at stable $T$ (reduced $k_B T$ variation)
- **Recalibration via uplink**: Periodic $\theta$ recalibration compensating for TID-induced drift

---

## 4. Technology Gaps (TC × Space)

| Gap | Severity | Path to Resolution |
|---|---|---|
| No radiation-hardened TC chips | Critical | Long-term fab R&D; near-term: shielding + error-tolerant architecture |
| Thermal cycling effect on $k_B T$ | High | Active thermal control subsystem |
| Subthreshold CMOS in space (TID) | High | Test campaigns (need dedicated space qualification) |
| Power scaling from 1 chip to 100M neurons | High | Multi-chip TSU array + low-power interconnect |
| Uplink bandwidth for $\theta$ updates | Medium | Compressed $\theta$ transmission; delta updates only |

---

## 5. Phase 3 Integration Targets

Phase 3 of thermodynamic-core:
1. Formal TRL assessment for TC hardware in LEO environment
2. Power budget model: watts per neuron for TSU-based payload at 100M scale
3. SEU tolerance analysis for subthreshold CMOS p-bits
4. Interface specification for parameter uplink and recalibration protocol
5. Thermal envelope specification compatible with arkspace-core satellite bus

---

## References

- See `arkspace-core/` repository for satellite specifications
- See [`../architecture/hardware-primitives.md`](../architecture/hardware-primitives.md) for TC hardware details
- See [`../../reference/bibliography.md`](../../reference/bibliography.md) for full citations
