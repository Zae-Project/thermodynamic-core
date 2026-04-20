# Hardware Primitives: Probabilistic Bits (p-bits)

**Version**: 0.1.0  
**Status**: Draft  
**Last Updated**: April 2026

---

## Overview

The fundamental building block of a thermodynamic neuromorphic system is the **probabilistic bit** (p-bit). Unlike a deterministic bit fixed at 0 or 1, a p-bit fluctuates between states with a tunable probability distribution. This stochastic behavior allows direct hardware implementation of Bernoulli and categorical sampling — essential for probabilistic graphical models and brain-inspired computation.

The p-bit's sigmoid voltage-to-probability response:

$$P(\text{up}) = \sigma(\beta V) = \frac{1}{1 + e^{-\beta V}}$$

maps cleanly onto the activation functions used in neural networks and the spin statistics of Ising/Boltzmann machines.

---

## 1. Spintronic p-bits: Magnetic Tunnel Junctions (MTJ)

### 1.1 Structure

An MTJ consists of:
1. A **reference ferromagnetic layer** (fixed magnetization)
2. A thin **insulating barrier** (MgO, ~1 nm)
3. A **free layer** (fluctuating magnetization)

In **low-barrier nanomagnets (LBNM)**, the thermal stability factor $\Delta < 25$, causing spontaneous magnetization fluctuations driven by ambient thermal energy.

### 1.2 Control Mechanisms

| Mechanism | Principle | Key Advantage | Limitation |
|---|---|---|---|
| **Spin-Transfer Torque (STT)** | Spin-polarized current torques the free layer | Established, controllable | High current density → Joule heating |
| **Spin-Orbit Torque (SOT)** | Heavy-metal underlayer generates spin-Hall effect | Faster switching, write/read decoupled | Requires additional layer |
| **Voltage-Controlled Magnetic Anisotropy (VCMA)** | Electric field modulates the anisotropy barrier | Minimal Ohmic losses, ultra-low power | Early-stage R&D |

VCMA-assisted switching in SHE-MTJ (Spin Hall Effect) devices minimizes Joule heating by modulating the anisotropy barrier with voltage rather than high current, providing a scalable foundation for high-density p-bit arrays (Gao et al., Micromachines 2025).

### 1.3 Experimental Validation

- **CIFAR-10 classification**: VCMA-MTJ-based SqueezeNet achieved 72.49% accuracy with $1.25 \times 10^6$ parameters (Gao et al., 2025)
- Demonstrated sigmoidal $P(\text{up})$ vs. applied voltage characteristic
- Integration with standard CMOS back-end-of-line processes demonstrated at research scale

### 1.4 Technology Gaps (MTJ)

- Heusler alloy free layers require specialized deposition equipment not in standard CMOS fabs
- MgO barrier uniformity at nm scale is yield-limiting
- Temperature sensitivity: $\Delta$ varies with $T$, complicating calibration in LEO thermal cycling
- No radiation-hardened MTJ p-bit arrays demonstrated

---

## 2. All-Transistor p-bits: Subthreshold CMOS

### 2.1 Principle

When a MOSFET operates in the **subthreshold regime** (gate voltage $V_G < V_{th}$), drain current is dominated by diffusion rather than drift. This makes it exquisitely sensitive to thermal fluctuations, creating a natural source of intense thermal noise for random number generation (RNG).

$$I_{DS} \approx I_0 \exp\!\left(\frac{V_{GS} - V_{th}}{n V_T}\right), \quad V_T = k_B T / q$$

### 2.2 The All-Transistor Approach

Extropic AI's **Thermodynamic Sampling Unit (TSU)** uses subthreshold transistor networks as stochastic primitives, avoiding the manufacturing challenges of MTJ integration. This approach:
- Uses present-day CMOS foundry processes (no exotic materials)
- Achieves fast, small, energy-efficient RNG cells
- Avoids communication overhead from interfacing disparate material systems
- Scales with standard CMOS roadmap

### 2.3 Performance Claims

Based on Jelinčič et al. (arXiv:2510.23972):
- $\sim 10,000 \times$ energy improvement over GPU inference for diffusion-like models
- All-transistor RNG achieves performance parity with GPUs at a fraction of the energy
- Extropic's XTR-0 development platform demonstrates communication between TSUs and conventional processors

> **Note**: These claims are from a 2025 preprint by the hardware developers (Extropic AI) and have not yet received independent third-party experimental replication at production scale.

---

## 3. Josephson Junction p-bits (Cryogenic)

### 3.1 Principle

Josephson junctions exploit macroscopic quantum tunneling of Cooper pairs across a thin insulating barrier between two superconductors. They exhibit fast, low-noise stochastic switching.

| Property | Value |
|---|---|
| Operating temperature | ~4 K (liquid helium), or ~15 mK (dilution refrigerator) |
| Switching speed | Sub-nanosecond |
| Energy per operation | Extremely low (quantum limit regime) |

### 3.2 Relevance to Zae

Cryogenic operation rules out direct use in LEO orbital payloads with current technology. Josephson junction TC is most relevant for terrestrial large-scale infrastructure. Included here for completeness of the technology landscape.

---

## 4. Substrate Comparison

| Substrate | Physical Principle | Key Advantage | Limitation | Scaling Path |
|---|---|---|---|---|
| **STT/SOT MTJ** | Magnon-driven switching | Non-volatility, speed | Integration complexity | Embedded MRAM process |
| **VCMA-MTJ** | Electric-field anisotropy modulation | Ultra-low power | Early-stage R&D | Spintronic-CMOS co-integration |
| **Subthreshold CMOS** | Thermal diffusion | Standard fabrication | Volatility | High-density ASIC |
| **Josephson Junction** | Macroscopic quantum tunneling | Speed, fidelity | Cryogenic only | Special-purpose terrestrial |

**Recommended near-term path for Zae**: All-transistor subthreshold CMOS. Standard fab compatibility provides a more immediate and scalable route to high-density integration within existing foundries. MTJ research should be monitored for eventual co-integration.

---

## References

See [`../../reference/bibliography.md`](../../reference/bibliography.md):
- Gao et al. (Micromachines 2025) — VCMA-MTJ + SqueezeNet CIFAR-10
- Jelinčič, Verdon, McCourt et al. (arXiv:2510.23972) — TSU probabilistic hardware
- Extropic AI: "TSU 101" and "Inside X0 and XTR-0" (2025)
