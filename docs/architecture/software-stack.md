# Software Stack: From ML Graph to TSU Execution

**Version**: 0.1.0  
**Status**: Draft  
**Last Updated**: April 2026

---

## Overview

Deploying brain-scale neural networks on thermodynamic hardware requires a complete software-to-silicon toolchain: compilers that translate high-level neural network graphs into thermodynamic primitives, runtimes that schedule sampling operations, and simulation libraries that allow development before hardware exists.

---

## 1. THRML: Thermodynamic Hypergraphical Model Library

### 1.1 What It Is

**THRML** (Thermodynamic Hypergraphical Model Library) is a JAX-based Python library for building and sampling probabilistic graphical models on thermodynamic hardware. It is the primary software interface to Extropic AI's TSU hardware.

- **Language**: Python (JAX backend)
- **Documentation**: [docs.thrml.ai](https://docs.thrml.ai/en/latest/architecture/)
- **Focus**: Block Gibbs sampling, factor graph models, hardware-aware compilation

### 1.2 Architecture

THRML's design philosophy bridges abstract probabilistic model graphs and physical hardware instruction sets:

```
[PyTorch / JAX Model Graph]
         ↓  graph lowering
[THRML Factor Graph Representation]
         ↓  block partitioning + bipartite coloring
[Hardware-aware Memory Mapping]
         ↓  ISA emission
[TSU Instruction Stream]
         ↓  execution
[Thermodynamic Sampling Unit (TSU)]
```

### 1.3 Key Techniques

| Technique | Purpose |
|---|---|
| **Factor-based interaction** | Decomposes energy function into local terms |
| **Global state representation** | Maximizes array-level parallelism in JAX |
| **Block Gibbs sampling** | Updates disjoint variable subsets in parallel |
| **Bipartite coloring** | Identifies independent variable sets for parallel updates |
| **Hardware-aware memory mapping** | Keeps parameters near sampling cells to minimize communication energy |

### 1.4 Portfolio Selection Validation

THRML SSNN validated on financial portfolio selection using an Ising model:
- Error: **4.31%** on GPU-simulated TSU
- Demonstrates applicability beyond image recognition tasks

---

## 2. Compiler Architecture

### 2.1 Graph Lowering

Converting standard neural network models (PyTorch, JAX) to energy-based representations requires:

1. **Layer decomposition**: Linear layers → quadratic energy terms ($x^\top J y + b^\top x$)
2. **Activation functions**: Approximate non-linearities with piecewise energy landscapes
3. **Skip connections**: Convert to bilinear coupling terms $x^\top J_{\text{skip}} y$ (see HBSC)
4. **Normalization layers**: Absorb into bias vectors or coupling rescaling

### 2.2 Block Partitioning

Gibbs sampling requires updating variable subsets. Effective partitioning:
- Uses **bipartite graph coloring** to identify independent blocks
- Ensures no two variables in the same update block are coupled
- Maximizes parallelism: all variables in a block update simultaneously

### 2.3 Hardware-Aware Memory Mapping

Minimizing energy from communication (not just computation):
- Coupling weights $J_{ij}$ stored adjacent to the sampling cells they connect
- Bias vectors $b_i$ co-located with their neurons
- MLIR/LLVM-based infrastructure for performance-centric ISA emission

---

## 3. The Thermodynamic Sampling Unit (TSU)

### 3.1 Architecture

Based on Extropic AI's XTR-0 development platform and forthcoming Z1 TSU:

```
┌────────────────────────────────────────────┐
│               TSU Chip                      │
│                                             │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐    │
│  │Sampling │  │Sampling │  │Sampling │    │
│  │ Core 0  │  │ Core 1  │  │ Core N  │    │
│  │(p-bits) │  │(p-bits) │  │(p-bits) │    │
│  └────┬────┘  └────┬────┘  └────┬────┘    │
│       └────────────┼────────────┘          │
│                    │  On-chip interconnect  │
│              ┌─────▼──────┐                │
│              │  Controller │                │
│              │  + Memory   │                │
│              └─────┬───────┘                │
└────────────────────┼───────────────────────┘
                     │
              [Host CPU / GPU]
              (digital conditioning
               interface)
```

### 3.2 XTR-0 → Z1 Roadmap

| Platform | Description | Sampling Cells | Status |
|---|---|---|---|
| X0 | First internal prototype | Small | Internal (Extropic) |
| XTR-0 | Development platform for hybrid deterministic-thermodynamic algorithms | Research-scale | Available to partners |
| Z1 | Production TSU | Hundreds of thousands per chip | Forthcoming |

> **Note**: Extropic AI claims and roadmap details are from company publications (2025). Independent verification of Z1 specifications is not yet available.

---

## 4. Simulation Path for Zae (Phase 2)

Before Z1 hardware is available, development proceeds via GPU simulation:

```python
# Example THRML model structure (illustrative, not actual API)
import thrml

model = thrml.BoltzmannMachine(
    n_visible=784,   # MNIST input
    n_hidden=512,
    beta=1.0
)

# Block Gibbs sampling
samples = model.sample(n_steps=100, block_size=64)
```

Phase 2 of thermodynamic-core will add:
- Langevin integrator simulations in JAX/NumPy
- p-bit dynamics notebooks
- Boltzmann machine training on standard benchmarks
- HBSC rank-k approximation validation

---

## References

See [`../../reference/bibliography.md`](../../reference/bibliography.md):
- THRML Documentation — [docs.thrml.ai](https://docs.thrml.ai/en/latest/architecture/)
- Jelinčič et al. (arXiv:2510.23972) — TSU probabilistic hardware architecture
- Extropic AI: "Inside X0 and XTR-0" (2025)
