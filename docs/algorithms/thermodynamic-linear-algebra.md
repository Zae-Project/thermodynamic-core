# Thermodynamic Linear Algebra and Second-Order Optimization

**Version**: 0.1.0  
**Status**: Draft  
**Last Updated**: April 2026

---

## Overview

Beyond generative modeling, thermodynamic hardware can accelerate the **linear algebra primitives** that bottleneck deep learning: solving linear systems, inverting matrices, and performing matrix exponentiation. The Langevin substrate can solve these problems with a **linear speedup** $O(N)$ relative to matrix dimensions — versus $O(N^3)$ for classical algorithms.

This capability is especially relevant for **second-order optimization** methods like Natural Gradient Descent (NGD), which require computing or approximating the inverse of the Fisher information matrix.

---

## 1. Thermodynamic Linear Algebra

### 1.1 Core Result

A thermodynamic system with a quadratic energy function:

$$V(x) = \frac{1}{2} x^\top A x - b^\top x$$

achieves its equilibrium minimum at $x^* = A^{-1} b$ — which is exactly the solution to the linear system $Ax = b$.

The system *solves the linear system by relaxing to equilibrium*. This gives:

| Operation | Classical Complexity | Thermodynamic | Speedup |
|---|---|---|---|
| Linear system solve $Ax = b$ | $O(N^3)$ | $O(N)$ (relaxation time) | $N^2$ |
| Matrix inversion $A^{-1}$ | $O(N^3)$ | $O(N)$ | $N^2$ |
| Matrix exponentiation $e^A$ | $O(N^3)$ | $O(N)$ | $N^2$ |

> **Note**: The $O(N)$ thermodynamic complexity refers to the *scaling with system size* in the physical relaxation time. Real-world performance depends on mixing time, precision requirements, and hardware noise floors. See Aifer et al. (2024) for rigorous derivations.

### 1.2 Foundation Reference

Aifer et al. (2024), *Thermodynamic Linear Algebra*, npj Unconventional Computing 1:13. This paper formally establishes the mapping between thermodynamic relaxation and linear algebra operations.

---

## 2. Kronecker-Factored Approximate Curvature (K-FAC)

### 2.1 Natural Gradient Descent Background

Standard SGD updates parameters in the direction of the Euclidean gradient. **Natural Gradient Descent (NGD)** updates in the direction of the steepest ascent in the space of probability distributions, preconditioned by the inverse Fisher information matrix $F^{-1}$:

$$\theta \leftarrow \theta - \alpha\, F(\theta)^{-1}\, \nabla_\theta \mathcal{L}$$

NGD converges faster and is more invariant to parameter reparameterization — but computing $F^{-1}$ for large networks is prohibitively expensive ($O(P^3)$ for $P$ parameters).

### 2.2 K-FAC as an Approximation

**K-FAC** (Kronecker-Factored Approximate Curvature) approximates $F$ as a block-diagonal matrix where each block is a Kronecker product of two smaller matrices:

$$F_\ell \approx A_\ell \otimes G_\ell$$

This reduces the inversion cost from $O(P^3)$ to $O(P \cdot \text{block\_size}^3)$ — but the block inversions are still expensive for large blocks.

### 2.3 Thermodynamic K-FAC

Running K-FAC on a thermodynamic computer reduces the matrix inversion bottleneck to $O(N)$ physical relaxation (arXiv:2502.08603):

```
Standard K-FAC:    O(P · block_size³)  matrix inversion per step
Thermodynamic K-FAC: O(P · block_size)  thermodynamic relaxation per step
```

This makes the inversion cost equivalent to that of a first-order method (SGD), allowing natural gradient descent to run at SGD-level computational cost.

### 2.4 Implications

| Method | Per-step cost | Convergence | Hardware |
|---|---|---|---|
| SGD | $O(P)$ | Slow | GPU |
| K-FAC (GPU) | $O(P \cdot B^3)$ | Fast | GPU (expensive) |
| Thermodynamic K-FAC | $O(P \cdot B)$ | Fast | TSU + small GPU |

For training massive neural networks — including the brain-scale SNN in neutral-consciousness-engine — Thermodynamic K-FAC offers a path to second-order optimization efficiency with first-order computational cost.

---

## 3. Additional Linear Algebra Operations

### 3.1 Matrix Exponentiation

For a system evolving under $\dot{x} = -Ax + \text{noise}$, the state covariance evolves via $e^{-At}$. A thermodynamic system with energy $V(x) = \frac{1}{2} x^\top A x$ naturally computes matrix exponentials as part of its transient dynamics.

### 3.2 Eigenvalue Estimation

The principal eigenvectors of a symmetric matrix $A$ are the modes of the quadratic energy landscape $V(x) = \frac{1}{2} x^\top A x$. Thermodynamic relaxation with appropriate initialization can estimate dominant eigenvectors via power-iteration-like dynamics.

---

## 4. Phase 2 Targets

Implement in JAX:
- Quadratic energy Langevin integrator solving $Ax = b$
- Benchmark against `jnp.linalg.solve` on varying matrix sizes $N$
- Thermodynamic K-FAC approximation on a small neural network

---

## References

See [`../../reference/bibliography.md`](../../reference/bibliography.md):
- Aifer et al. (2024) — *Thermodynamic Linear Algebra* — npj Unconventional Computing 1:13
- arXiv:2502.08603 — *Accelerating K-FAC with Thermodynamic Hardware*
