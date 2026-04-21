<div align="center">

# thermodynamic-core

### *Computing with noise, not against it.*

<img src="https://img.shields.io/badge/status-research-camel?style=for-the-badge&labelColor=1a1a1a&color=c4a574" alt="Status: Research"/>
<img src="https://img.shields.io/badge/pillar-substrate-grey?style=for-the-badge&labelColor=1a1a1a&color=8b7355" alt="Pillar: Substrate"/>
<img src="https://img.shields.io/badge/phase-docs--only-brown?style=for-the-badge&labelColor=1a1a1a&color=6b5344" alt="Phase: Docs Only"/>
<img src="https://img.shields.io/badge/license-MIT-brown?style=for-the-badge&labelColor=1a1a1a&color=6b5344" alt="License: MIT"/>

</div>

<br>

---

## Overview

This repository contains the research specifications, theoretical foundations, and architectural documentation for **thermodynamic computing** as the physical substrate layer of the Zae Project.

**Project Status**: conceptual research phase (TRL 2–3). No hardware exists in this repository. The repo documents theoretical architecture, identifies technology gaps, and maps the path from current research to brain scale emulation.

> **Note**: thermodynamic computing at brain scale is a research frontier. Claims about energy efficiency and scaling represent theoretical limits and experimental projections, not validated production performance. Consult primary sources in [`reference/bibliography.md`](reference/bibliography.md).

---

## The Problem: The Energy Gap

Modern AI accelerators operate far above the thermodynamic lower bound of computation. Erasing one bit of information dissipates a minimum of $k_B T \ln 2 \approx 2.8 \times 10^{-21}$ J at room temperature (Landauer's Principle). Current digital CMOS logic dissipates energy at levels many orders of magnitude higher:

| Metric | Digital CMOS (Approx.) | Thermodynamic Ideal | Delta |
|---|---|---|---|
| Energy per operation | $10^{-12}$ to $10^{-15}$ J | $\sim 10^{-21}$ J | $6$–$9$ orders of magnitude |
| Signal-to-Noise Ratio | >1000:1 | ~1:1 | Inverted operating regime |
| Computation mechanism | Switching / clocking | Physical relaxation | n/a |

For the Zae Project specifically, the [arkspace-core](https://github.com/Zae-Project/arkspace-core) design targets 100M neurons per satellite node at 50–200W. This target is not reachable with von Neumann digital architectures. Thermodynamic computing performs computation by the natural relaxation of a physical stochastic substrate, which is the physical paradigm that makes the target conceivable.

---

## The Substrate: What Thermodynamic Computing Is

Thermodynamic computing treats **thermal noise as a computational resource** rather than a source of error to be suppressed. A thermodynamic computer consists of physical degrees of freedom (voltages, magnetic moments, mechanical oscillators) whose dynamics are governed by the overdamped Langevin equation:

$$\dot{x}_i = -\mu \,\partial_i V_{\theta}(x) + \sqrt{2\mu k_B T}\, \eta_i(t)$$

where $V_\theta(x)$ is a programmable energy landscape (the "program"), $k_B T$ is thermal energy, and $\eta_i(t)$ is Gaussian white noise (the "fuel"). In thermal equilibrium the state distribution follows the Boltzmann-Gibbs law, $\rho(x) \propto e^{-\beta V_\theta(x)}$.

Inference, sampling, and optimization are core to spiking neural network operation, and they are **formally identical** to the natural dynamics of physical systems relaxing toward equilibrium. The substrate computes by physics, not by transistor switching.

---

## Where Thermodynamic-Core Sits in the Zae Project

The Zae Project has four interconnected pillars:

| Pillar | Repository | Layer | Role |
|---|---|---|---|
| 🛰️ **Infrastructure** | [arkspace-core](https://github.com/Zae-Project/arkspace-core) | Space | *Where* computation lives (LEO satellite constellation) |
| 🧠 **Interface** | [brain-emulation](https://github.com/Zae-Project/brain-emulation) | BCI | *How* bio↔synthetic connects (corpus callosum) |
| ⚡ **Engine** | [neutral-consciousness-engine](https://github.com/Zae-Project/neutral-consciousness-engine) | Software | *What* runs (SNN, neural firewall) |
| 🌡️ **Substrate** | **thermodynamic-core** (this repo) | Physics | *How it computes* (stochastic physics, p-bits, Langevin) |

The Substrate layer sits **beneath** the Engine and is **embedded in** the Infrastructure's payload design. It defines the physical computing paradigm on which the Engine ultimately runs.

---

## Key Technologies

| Technology | Physical Principle | Advantage | Status |
|---|---|---|---|
| **STT/SOT MTJ p-bits** | Magnon-driven switching | Non-volatility, speed | Research |
| **VCMA-MTJ p-bits** | Electric-field anisotropy modulation | Ultra-low switching energy | Early R&D |
| **Subthreshold CMOS** | Thermal diffusion in subthreshold transistors | Standard fabrication process | Demonstrated |
| **STL Neuristor** | Single-transistor latch, CMOS-compatible | Stochastic + deterministic dual mode | Research |
| **Josephson Junctions** | Macroscopic quantum tunneling | High speed, high fidelity | Cryogenic only |
| **THRML (JAX)** | Software-to-hardware compiler | Bridges ML graphs to TC hardware | Actively developed |

---

## Technology Gaps

The following gaps exist between current demonstrated technology and what is required for brain scale deployment on orbital hardware:

| Gap | Description | Path Forward |
|---|---|---|
| **Radiation hardening** | No radiation-hardened stochastic/TC chips demonstrated for LEO | Shielding + error-correcting architectures; long-term fab R&D |
| **Integration complexity** | MTJ p-bits require exotic materials (Heusler alloys, MgO barriers) not standard in CMOS fabs | All-transistor subthreshold CMOS as near-term substitute |
| **Interconnect wall** | $O(D^2)$ wiring intractable for $D \gtrsim 10^4$ feature dimensions | Hierarchical Bilinear Skip Coupling (rank-$k$ SVD approximation) |
| **Input conditioning** | Signal deficit of up to 2600× from fixed coupling constants in Langevin substrates | Minimal digital conditioning interface (<0.1% of model parameters) |
| **Training at scale** | Early TC designs required genetic algorithms; gradient-descent training only recently demonstrated | Whitelam (PNAS 2026) Onsager-Machlup functional approach |
| **Thermal management in orbit** | $k_B T$ computation requires careful thermal control; LEO has extreme hot/cold cycling | Thermal isolation + operating temperature engineering |

---

## Repository Structure

```
thermodynamic-core/
├── README.md                               # This file
├── LICENSE                                 # MIT
├── .gitignore
├── docs/
│   ├── architecture/
│   │   ├── theoretical-foundations.md      # Stochastic thermodynamics, Langevin, EDDP
│   │   ├── hardware-primitives.md          # p-bits: MTJ variants, subthreshold CMOS
│   │   ├── neuromorphic-integration.md     # SSNN, STL neuristors, Spiking STDP Transformer
│   │   ├── scaling-interconnects.md        # HBSC, eigenvalue concentration, digital conditioning
│   │   └── software-stack.md              # THRML, compiler/ISA, graph lowering
│   ├── algorithms/
│   │   ├── dtm-and-dtca.md                # Denoising Thermodynamic Models + Architecture
│   │   ├── thermodynamic-linear-algebra.md # K-FAC, natural gradient, matrix inversion
│   │   └── langevin-training.md           # Whitelam framework, Onsager-Machlup
│   ├── protocols/
│   │   └── hybrid-digital-analog.md       # Digital conditioning interface spec
│   └── integration/
│       ├── with-consciousness-engine.md    # TC ↔ neutral-consciousness-engine
│       ├── with-arkspace.md               # TC ↔ arkspace-core (orbital payload)
│       └── with-brain-emulation.md        # TC ↔ brain-emulation (stochastic vs. deterministic)
├── reference/
│   └── bibliography.md                    # All primary TC sources
└── research/
    └── technology-readiness.md            # TRL matrix for TC components
```

---

## Roadmap

| Phase | Scope | Status |
|---|---|---|
| **Phase 1**. Docs | Architecture specs, algorithm documentation, bibliography, cross-pillar integration contracts | In progress |
| **Phase 2**. Simulation | THRML (JAX) prototype simulations: p-bit dynamics, Boltzmann machines, Langevin integrators | Planned |
| **Phase 3**. Integration | TC hardware specs for arkspace-core orbital payloads, radiation-hardened substrate design | Future |

---

## Related Repositories

- [arkspace-core](https://github.com/Zae-Project/arkspace-core), the orbital infrastructure that this substrate must eventually run on.
- [neutral-consciousness-engine](https://github.com/Zae-Project/neutral-consciousness-engine), the SNN engine that this substrate underlies.
- [brain-emulation](https://github.com/Zae-Project/brain-emulation), the BCI interface layer.
- [zae-docs](https://github.com/Zae-Project/zae-docs), unified architecture documentation and bibliography.

---

## Contributing

All documentation changes should reference primary sources in [`reference/bibliography.md`](reference/bibliography.md). Clearly distinguish between:
- **Proven**: Demonstrated in peer-reviewed literature or deployed hardware
- **Speculative**: Theoretical projections or early-stage R&D claims

See [zae-docs](https://github.com/Zae-Project/zae-docs) for organization-wide contribution guidelines.

## License

MIT. See [LICENSE](LICENSE).
