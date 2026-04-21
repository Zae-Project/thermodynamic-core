# Technology Readiness Assessment: Thermodynamic Computing

**Version**: 0.1.0  
**Last Updated**: 2026-04-21  
**Framework**: NASA Technology Readiness Levels (TRL 1–9)

---

## TRL Scale Reference

| TRL | Definition |
|---|---|
| 1 | Basic principles observed |
| 2 | Technology concept formulated |
| 3 | Experimental proof of concept |
| 4 | Technology validated in lab |
| 5 | Technology validated in relevant environment |
| 6 | Technology demonstrated in relevant environment |
| 7 | System prototype demonstrated in operational environment |
| 8 | System complete and qualified |
| 9 | Actual system proven in operational environment |

---

## Component TRL Matrix

### Foundational Physics

| Component | TRL | Evidence | Gap to TRL+1 |
|---|---|---|---|
| Langevin dynamics as computation (theory) | **4** | Mathematically established; validated in simulations | Physical hardware demonstration at useful scale |
| Boltzmann-Gibbs sampling via TC | **4** | Demonstrated in lab prototypes (Whitelam PNAS 2026) | Scaling to production-class model size |
| Landauer bound as target (theory) | **5** | Well-established physics; no controversy | Engineering to approach the bound |
| Onsager-Machlup training (gradient descent) | **3–4** | PNAS 2026 proof-of-concept on MNIST (physical oscillator) | Demonstration on larger, practical SNN |

### Hardware Primitives

| Component | TRL | Evidence | Gap to TRL+1 |
|---|---|---|---|
| STT-MTJ p-bit | **5** | Multiple lab demonstrations; integrated with CMOS back-end | Density scaling; yield improvement |
| SOT-MTJ p-bit | **4** | Lab demonstrations; faster switching than STT | Full integration with CMOS; yield at scale |
| VCMA-MTJ p-bit | **3–4** | Research demonstrations (Gao et al. 2025) | Manufacturing robustness; standard fab integration |
| Subthreshold CMOS RNG | **5–6** | XTR-0 development platform (Extropic AI); standard fab | Production chip qualification (Z1 TSU) |
| STL Neuristor (CMOS) | **3–4** | Nature Electronics / ACS Nano 2025 proof-of-concept | Multi-neuron array; scaling to network size |
| Josephson Junction p-bit | **4** | Lab demonstrations (cryogenic) | Irrelevant for LEO without cryocooler |

### Algorithms & Software

| Component | TRL | Evidence | Gap to TRL+1 |
|---|---|---|---|
| Denoising Thermodynamic Model (DTM) | **3** | Theoretical framework plus small scale simulations | DTCA hardware prototype |
| DTCA (hardware architecture) | **2–3** | Described in arXiv:2510.23972; XTR-0 prototype | Z1 production chip |
| THRML library (JAX) | **4–5** | Available software; portfolio selection demo | Full production scale model support |
| HBSC skip couplings | **3** | Theoretical + simulation (arXiv:2604.14332) | Physical analog bus implementation |
| Hybrid digital conditioning | **3** | Theoretical + simulation | Hardware integration with TSU |
| Thermodynamic K-FAC | **2–3** | Theoretical (arXiv:2502.08603) | Hardware demonstration |
| Thermodynamic linear algebra | **3–4** | Proven mathematically; lab demonstrations | Scaling to practical matrix sizes |

### System Integration (Zae-specific)

| Component | TRL | Evidence | Gap to TRL+1 |
|---|---|---|---|
| TC as SNN substrate (theory) | **2** | Formal mathematical mapping established here | Simulation validation |
| SNN→TC parameter transfer (Onsager-Machlup) | **2** | Training pipeline designed (this repo) | Implementation + MNIST validation |
| TC payload for LEO satellite | **1–2** | Concept only; no space-relevant TC hardware | Radiation testing of TC components |
| Radiation-hardened TC hardware | **1** | No demonstrated radiation-hardened TC chips | Basic principles apply; no hardware |
| TC × neural firewall security | **1** | Conceptual only | Security architecture design |

---

## Summary Assessment

| Pillar | Overall TRL | Bottleneck |
|---|---|---|
| **TC Theory** | 4 | Lab-to-production scaling |
| **TC Hardware (terrestrial)** | 4–5 | Production chip availability (Z1 TSU) |
| **TC Software (THRML)** | 4–5 | Production scale model support |
| **TC for Brain Scale (terrestrial)** | 2–3 | HBSC, conditioning, SNN integration |
| **TC for Orbital (LEO)** | 1–2 | Radiation hardening. Critical gap |

---

## Risk Register

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Z1 TSU delayed or underperforms | Medium | High | Continue GPU simulation in Phase 2; hardware agnostic algorithm development |
| VCMA-MTJ fails to reach standard fab | Medium | Medium | Fallback to subthreshold CMOS; MTJ as long-term target |
| Radiation hardening intractable for LEO | Low–Medium | High | Shielding + periodic recalibration via uplink; TID tolerance study |
| Langevin relaxation too slow for <50ms latency | Unknown | High | Needs experimental characterization; hardware specific measurement |
| HBSC fidelity degrades at brain scale | Low | High | Rank-k can be increased; validation at scale needed |

---

## Next Steps (Phase 2)

1. Implement Langevin integrator in JAX; validate TRL-4 claim numerically
2. Run Onsager-Machlup training on MNIST; advance SNN→TC transfer to TRL 3
3. Benchmark THRML on standard probabilistic models
4. Initiate contact with Extropic AI / LBNL for hardware access or collaboration
5. Commission a radiation environment analysis for subthreshold CMOS in LEO (TRL gap assessment)
