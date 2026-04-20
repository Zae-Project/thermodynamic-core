# Thermodynamic Computing — Bibliography

**Version**: 1.0  
**Last Updated**: April 2026  
**Purpose**: Primary sources for thermodynamic-core research

All sources are organized by research cluster. For the unified Zae Project bibliography (including consciousness, BCI, neuromorphic, WBE, and satellite domains), see [zae-docs/reference/bibliography.md](https://github.com/Zae-Project/zae-docs/blob/main/reference/bibliography.md).

---

## 1. Foundational Theory & Manifesto

### Tom Conte, Todd Hylton et al. (2019)
- **Title**: *Thermodynamic Computing* — CCC Workshop Report
- **Source**: arXiv:1911.01968
- **Authors**: Tom Conte, Erik DeBenedictis, Natesh Ganesh, Todd Hylton, John Paul Strachan, R. Stanley Williams, Alexander Alemi, Peter Baldi, et al.
- **Key Contributions**: Defines thermodynamic computing as a field; manifesto for treating thermal noise as a computational resource; energy bounds and hardware roadmap
- **Relevance to Zae**: Foundational framework; establishes the $k_B T$ lower bound argument for brain-scale efficiency
- **Access**: [arXiv:1911.01968](https://arxiv.org/abs/1911.01968)

### Todd Hylton (2020)
- **Title**: *Thermodynamic Computing* (Book)
- **Publisher**: River Publishers
- **Key Contributions**: Comprehensive architecture guide for evolvable thermodynamic cores; philosophical and engineering foundations
- **Relevance to Zae**: Architecture patterns for building TC systems
- **Access**: [livecarta.com catalog](https://app.livecarta.com/catalog/from-artificial-intelligence-to-brain-intelligence)

---

## 2. Extropic AI: TSU / DTM / THRML Hardware & Software

### Andraž Jelinčič, Guillaume Verdon, Trevor McCourt et al. (2025)
- **Title**: *An Efficient Probabilistic Hardware Architecture for Diffusion-like Models*
- **Source**: arXiv:2510.23972
- **Authors**: Andraž Jelinčič, Guillaume Verdon, Trevor McCourt, et al. (Extropic AI)
- **Key Contributions**: Core technical paper on the all-transistor TSU; $10{,}000\times$ energy efficiency claim vs. GPU for diffusion-like inference; Denoising Thermodynamic Computer Architecture (DTCA); block Gibbs sampling hardware
- **Relevance to Zae**: Primary hardware reference for the Substrate pillar; foundation for arkspace-core TC payload design
- **Access**: [arXiv:2510.23972](https://arxiv.org/abs/2510.23972)

### Extropic AI (2025) — "Thermodynamic Computing: From Zero to One"
- **Type**: Technical blog / announcement
- **Key Contributions**: Conceptual transition from digital to stochastic substrate; energy-based sampling paradigm
- **Access**: [extropic.ai/writing/thermodynamic-computing-from-zero-to-one](https://extropic.ai/writing/thermodynamic-computing-from-zero-to-one)

### Extropic AI (2025) — "TSU 101: An Entirely New Type of Computing Hardware"
- **Type**: Technical explainer
- **Key Contributions**: Breakdown of the p-bit, sampling cells, on-chip interconnects, and the TSU ISA
- **Access**: [extropic.ai/writing/tsu-101-an-entirely-new-type-of-computing-hardware](https://extropic.ai/writing/tsu-101-an-entirely-new-type-of-computing-hardware)

### Extropic AI (2025) — "Inside X0 and XTR-0"
- **Type**: Hardware whitepaper
- **Key Contributions**: Details on the X0 prototype and XTR-0 development platform; interface between TSU and conventional processors
- **Access**: [extropic.ai/writing/inside-x0-and-xtr-0](https://extropic.ai/writing/inside-x0-and-xtr-0)

### THRML Documentation (Thermodynamic Hypergraphical Model Library)
- **Type**: Software documentation
- **Key Contributions**: JAX-based library for building and sampling probabilistic graphical models on TC hardware; block Gibbs sampling; software-to-hardware compiler architecture
- **Access**: [docs.thrml.ai/en/latest/architecture/](https://docs.thrml.ai/en/latest/architecture/)

---

## 3. Scaling & Training: The Whitelam Framework

### Stephen Whitelam (2026)
- **Title**: *Training Thermodynamic Computers by Gradient Descent*
- **Journal**: PNAS (Proceedings of the National Academy of Sciences)
- **DOI**: [10.1073/pnas.2528413123](https://www.pnas.org/doi/10.1073/pnas.2528413123)
- **Institution**: Molecular Foundry, Lawrence Berkeley National Laboratory
- **Key Contributions**: First demonstration that gradient descent (via the Onsager-Machlup functional) is viable for training TC hardware parameters; dramatically faster convergence than prior genetic-algorithm approaches; MNIST validation on a physical oscillator substrate
- **Relevance to Zae**: Core training method for SNN-to-TC parameter transfer; Phase 2 implementation target

### Stephen Whitelam & Corneel Casert (2026)
- **Title**: *Thermodynamic Diffusion Inference with Minimal Digital Conditioning*
- **Journal**: Nature Communications
- **Source**: arXiv:2604.14332
- **Key Contributions**: Hierarchical Bilinear Skip Coupling (HBSC) — rank-$k$ SVD-based non-local interconnects; minimal digital conditioning interface (<0.1% of model parameters); $10^7\times$ theoretical energy saving over GPU inference; rank-16 decoder cosine similarity 0.9906 vs. digital oracle
- **Relevance to Zae**: Primary reference for HBSC and hybrid conditioning — both are core to brain-scale TC design
- **Access**: [arXiv:2604.14332](https://arxiv.org/abs/2604.14332)

### Stephen Whitelam (2026)
- **Title**: *Thermodynamic Computing Advances with Design and Training*
- **Journal**: Nature Communications 17, 1189
- **Institution**: Molecular Foundry / Lawrence Berkeley National Laboratory
- **Key Contributions**: Non-linear thermodynamic computers mimicking neural networks; focuses on expressivity of non-quadratic energy landscapes
- **Relevance to Zae**: Extends TC beyond quadratic/Ising models to full neural network expressivity
- **Access**: [Molecular Foundry announcement](https://foundry.lbl.gov/2026/03/06/thermodynamic-computing-advances-with-design-and-training-2/) | [Nature Communications 17:1189](https://www.nature.com/articles/s41467-026-XXXXX)

---

## 4. Neuromorphic Integration & p-bits

### Liang Gao et al. (2025)
- **Title**: *Stochastic Neuromorphic Computing Architecture based on VCMA-MTJ*
- **Journal**: IEEE TCAD / Micromachines
- **DOI / Access**: [mdpi.com/2072-666X/17/2/216](https://www.mdpi.com/2072-666X/17/2/216)
- **Key Contributions**: Implementation of VCMA-MTJ spintronic p-bits in a neuromorphic architecture; SqueezeNet on CIFAR-10 achieving 72.49% accuracy; energy-efficient spintronic-CMOS co-integration
- **Relevance to Zae**: Demonstrates p-bit-based neural classification; hardware baseline for spintronic TC

### (Authors TBC) — Nature Electronics (2025)
- **Title**: *A Neuristor based on Single Transistor Latch (STL) in CMOS*
- **Journal**: Nature Electronics / ACS Nano
- **Access**: [pubs.acs.org/doi/10.1021/acsnano.5c15076](https://pubs.acs.org/doi/10.1021/acsnano.5c15076)
- **Key Contributions**: CMOS-compatible stochastic spiking neuron (neuristor) using the STL mechanism; dual-mode operation (stochastic input, deterministic hidden); 92% MNIST accuracy under 30% Gaussian noise
- **Relevance to Zae**: Direct bridge between TC hardware and SNN architecture; key building block for SSNN in orbital payload

### (Authors TBC) (2025)
- **Title**: *Spiking STDP Transformer (S²TDPT)*
- **Source**: arXiv:2511.14691
- **Key Contributions**: In-memory computing using STDP-based spike timing for attention mechanisms; addition-only operations eliminating large intermediate attention matrices; non-von Neumann hardware deployment
- **Relevance to Zae**: TC-compatible attention mechanism for the generative model in neutral-consciousness-engine
- **Access**: [arXiv:2511.14691](https://arxiv.org/html/2511.14691v1)

---

## 5. Advanced Mathematical Operations

### (Authors TBC) (2025)
- **Title**: *Accelerating K-FAC with Thermodynamic Hardware*
- **Source**: arXiv:2502.08603
- **Key Contributions**: Reducing K-FAC matrix inversion from $O(B^3)$ to $O(B)$ using thermodynamic relaxation; enables natural gradient descent at first-order computational cost; applicable to training massive neural networks
- **Relevance to Zae**: Second-order optimization for SNN training — potentially the training method for brain-scale models
- **Access**: [arXiv:2502.08603](https://arxiv.org/html/2502.08603v1)

### Maxwell Aifer et al. (2024)
- **Title**: *Thermodynamic Linear Algebra*
- **Journal**: npj Unconventional Computing 1, 13
- **Key Contributions**: Formal proof and demonstration of $O(N)$ thermodynamic speedup for linear system solving, matrix inversion, and matrix exponentiation; foundation for thermodynamic second-order methods
- **Relevance to Zae**: Mathematical foundation for all TC-based linear algebra; prerequisite for Thermodynamic K-FAC
- **Access**: [npj Unconventional Computing 1:13](https://www.nature.com/articles/s44335-024-00013-3)

---

## Notes on Source Integrity

- arXiv preprints marked with arXiv IDs are not yet peer-reviewed unless stated otherwise
- Extropic AI publications (blog posts, whitepapers) are company-authored and have not undergone independent peer review — treat performance claims as reported, not independently verified
- "Authors TBC" entries reflect cases where the primary paper author list was not included in the source material — verify before citing
- All arXiv IDs are as provided in the primary source document; verify currency before relying on them
