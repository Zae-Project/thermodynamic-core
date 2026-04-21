# Theoretical Foundations of Thermodynamic Computing

**Version**: 0.1.0  
**Status**: Draft  
**Last Updated**: 2026-04-21

---

## Overview

Thermodynamic computing is grounded in non-equilibrium statistical mechanics and the theory of stochastic differential equations. A thermodynamic computer exploits thermal fluctuations as a computational resource instead of suppressing them. Several machine learning primitives (sampling, relaxation, optimization) are formally identical to the natural dynamics of physical systems in contact with a thermal bath.

---

## 1. The Langevin Primitive

### 1.1 Equation of Motion

A thermodynamic computer consists of $N$ classical degrees of freedom $x = \{x_i\}$ (voltages, magnetic moments, oscillator positions). Their time evolution follows the **overdamped Langevin equation**:

$$\dot{x}_i = -\mu \,\partial_i V_{\theta}(x) + \sqrt{2\mu k_B T}\, \eta_i(t)$$

| Symbol | Meaning |
|---|---|
| $\mu$ | Mobility parameter (hardware-dependent) |
| $V_\theta(x)$ | Programmable potential energy landscape (encodes the problem) |
| $\theta$ | Hardware parameters: couplings $J_{ij}$, biases $b_i$ |
| $k_B$ | Boltzmann constant ($1.38 \times 10^{-23}$ J/K) |
| $T$ | Absolute temperature |
| $\eta_i(t)$ | Gaussian white noise: $\langle \eta_i(t) \eta_j(t') \rangle = \delta_{ij}\delta(t-t')$ |

### 1.2 Equilibrium Distribution

At thermal equilibrium, the stationary distribution of the system is the **Boltzmann-Gibbs distribution**:

$$\rho(x) \propto e^{-V_\theta(x) / k_B T} = e^{-\beta V_\theta(x)}$$

where $\beta = (k_B T)^{-1}$. Programming $V_\theta$ is equivalent to programming the target probability distribution. The hardware samples from it automatically by relaxing toward equilibrium.

### 1.3 Identity with Machine Learning Primitives

The Langevin equation is formally identical to key ML operations:

| ML Operation | Thermodynamic Equivalent |
|---|---|
| MCMC sampling | Langevin dynamics with programmable $V_\theta$ |
| Diffusion model inference | Reverse score-matching SDE = Langevin with $V_\theta = -\log p_{\text{data}}$ |
| Boltzmann machine inference | Equilibrium distribution of Ising-like energy |
| Natural gradient descent | Relaxation of Fisher-geometry system (see K-FAC doc) |

---

## 2. Energy-Delay-Deficiency Product (EDDP)

### 2.1 Definition

The **Energy-Delay-Deficiency Product** extends the classical energy-delay product to probabilistic systems:

$$EDDP = W_{\text{diss}} \cdot \tau \cdot (1 - Q)$$

| Symbol | Meaning |
|---|---|
| $W_{\text{diss}}$ | Total dissipated work (Joules) |
| $\tau$ | Total time for computation and sampling (seconds) |
| $Q$ | Accuracy or fidelity of the result (0–1) |

### 2.2 Landauer's Bound

By Landauer's Principle, the minimum energy to erase one bit of information at temperature $T$ is:

$$E_{\text{Landauer}} = k_B T \ln 2 \approx 2.8 \times 10^{-21} \text{ J at } 300\text{ K}$$

Digital CMOS operates $10^6$–$10^9 \times$ above this bound due to timing control overhead and voltage-level switching. Thermodynamic computing targets operation at or near $k_B T$.

### 2.3 Implications for Brain-Scale Emulation

The human brain consumes ~20 W for ~86 billion neurons. A brain scale synthetic substrate at $k_B T$ efficiency operating at $10^{-21}$ J/operation would, in principle, require negligible power compared to its digital equivalent. In practice, estimates for production-scale thermodynamic diffusion inference suggest energy savings of $\sim 10^7 \times$ over GPU execution (Jelinčič et al., arXiv:2510.23972).

---

## 3. The Compute Crisis Context

### 3.1 End of Dennard Scaling

As semiconductor fabrication approaches the 2 nm node, quantum tunneling and thermal fluctuations cause transistors to leak and fail. Moore's Law transistor density improvements continue, but power scaling has ended: clock frequencies are largely stalled, and the dominant cost has shifted to data movement (the von Neumann "memory wall").

### 3.2 AI Demand vs. CMOS Supply

AI compute demand has been doubling approximately every 100 days. This creates a fundamental mismatch with CMOS supply, which no longer scales as Moore's Law once guaranteed. Thermodynamic computing represents one of the primary candidate paradigms for breaking this bottleneck.

---

## 4. Non-Equilibrium Extensions

### 4.1 Open-System Thermodynamics

Real TC hardware operates out of equilibrium, driven by input signals and dissipating energy to perform useful computation. The relevant theoretical framework is **stochastic thermodynamics** (Seifert, 2012), which extends classical thermodynamics to individual trajectories of small systems.

Key results relevant to TC:
- **Jarzynski equality**: $\langle e^{-\beta W} \rangle = e^{-\beta \Delta F}$, relating non-equilibrium work to free energy differences.
- **Crooks fluctuation theorem**: ratio of forward/reverse path probabilities relates to entropy production.
- **Second-law inequality**: $\langle W_{\text{diss}} \rangle \geq \Delta F$, lower bound on computational cost.

### 4.2 Implications for Training

The Onsager-Machlup functional (see [`../algorithms/langevin-training.md`](../algorithms/langevin-training.md)) uses the path probability of Langevin trajectories to derive gradient descent update rules for hardware parameters $\theta$. This is the bridge between non-equilibrium thermodynamics theory and practical TC training.

---

## References

See [`../../reference/bibliography.md`](../../reference/bibliography.md) for full citations. Key foundational works:
- Conte, Hylton et al. (2019). CCC Workshop Report. arXiv:1911.01968
- Hylton (2020). *Thermodynamic Computing* (River Publishers).
- Whitelam (PNAS 2026). Training TC by Gradient Descent.
- Aifer et al. (2024). Thermodynamic Linear Algebra. *npj Unconventional Computing* 1:13.
