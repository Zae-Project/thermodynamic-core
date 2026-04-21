# Training Thermodynamic Computers by Gradient Descent

**Version**: 0.1.0  
**Status**: Draft  
**Last Updated**: 2026-04-21

---

## Overview

A major historical obstacle in thermodynamic computing was the absence of efficient training methods. Early designs relied on genetic algorithms or exhaustive search to identify coupling parameters. These approaches do not scale to large systems.

Whitelam (PNAS 2026) demonstrated that **gradient descent (GD) is viable for training TC hardware**. Training is expressed as maximizing the probability of desired Langevin trajectories.

---

## 1. The Teacher-Student Framework

### 1.1 Setup

Training proceeds by defining an **idealized target trajectory** (the "teacher"). This is a sequence of states $\{x^{(0)}, x^{(1)}, \ldots, x^{(K)}\}$ that reproduces the activations of a target digital neural network when given the same input. The thermodynamic computer (the "student") is trained to maximize the probability that its Langevin dynamics generate that trajectory.

### 1.2 Why This Works

The probability of a specific state change $x \to x + \Delta x$ in a Langevin system is proportional to the probability of drawing the exact noise values $\eta$ that would produce it. This gives us a differentiable loss function with respect to hardware parameters $\theta$.

---

## 2. The Onsager-Machlup Functional

### 2.1 Path Probability

The probability of a specific discrete Langevin step is given by the **discrete Onsager-Machlup action**:

$$-\ln P_\theta^{\text{step}}(\Delta x) = \sum_{i=1}^N \frac{(\Delta x_i + \mu\, \partial_i V_\theta(x)\, \Delta t)^2}{4\mu k_B T\, \Delta t}$$

This is the negative log-probability of observing the step $\Delta x$ given the current state $x$ and parameters $\theta$.

### 2.2 Gradient with Respect to Parameters

Taking the derivative with respect to hardware coupling $J_{ij}$ (where $\partial_i V_\theta = \sum_j J_{ij} x_j + b_i$):

$$\Delta J_{ij} = \alpha \sum_{k=0}^{K-1} \left( \frac{\Delta x_i^{(k)} + \mu\, \partial_i V_\theta(x^{(k)})\, \Delta t}{2 k_B T}\, x_j^{(k)} + \frac{\Delta x_j^{(k)} + \mu\, \partial_j V_\theta(x^{(k)})\, \Delta t}{2 k_B T}\, x_i^{(k)} \right)$$

where $\alpha$ is a learning rate and the sum runs over $K$ time steps.

Similarly for biases $b_i$:

$$\Delta b_i = \alpha \sum_{k=0}^{K-1} \frac{\Delta x_i^{(k)} + \mu\, \partial_i V_\theta(x^{(k)})\, \Delta t}{2 k_B T}$$

### 2.3 Practical Training Loop

```
1. Run target digital network on input → record activations as target trajectory {x*}
2. Initialize hardware parameters θ (random or from prior)
3. For each training batch:
   a. Run Langevin dynamics on TC simulator → trajectory {x}
   b. Compute Onsager-Machlup loss: L = -log P_θ(trajectory matches {x*})
   c. Backpropagate through L → compute ∂L/∂θ
   d. Update θ ← θ - α ∇_θ L
4. Once converged: program learned θ into physical TC hardware
5. Physical hardware then computes by natural thermal relaxation (no backprop needed)
```

---

## 3. Validation Results

From Whitelam (PNAS 2026):

| Task | Architecture | Result |
|---|---|---|
| MNIST digit recognition | Trained Langevin TC (physical oscillator substrate) | High fidelity |
| Convergence vs. genetic algorithm | Gradient descent (Onsager-Machlup) | Orders of magnitude fewer trajectories needed |

The gradient-based approach requires orders of magnitude fewer sampled trajectories to converge on tasks like MNIST than genetic search.

---

## 4. Implications for Zae

### 4.1 Training Pipeline

For neutral-consciousness-engine SNN models:
1. Train SNN on GPU/CPU in simulation using conventional backprop
2. Extract activation trajectories for each layer
3. Use Onsager-Machlup GD to find TC parameters $\theta$ that reproduce those trajectories
4. Program $\theta$ into physical TSU/p-bit arrays
5. Deploy: physical hardware runs inference via thermal relaxation

### 4.2 Phase 2 Target

Phase 2 of thermodynamic-core should implement:
- Langevin trajectory simulator (JAX/NumPy)
- Onsager-Machlup loss function
- Gradient descent training loop for small Boltzmann machines
- Validation on MNIST as benchmark

---

## References

See [`../../reference/bibliography.md`](../../reference/bibliography.md):
- Whitelam (PNAS 2026). *Training Thermodynamic Computers by Gradient Descent*.
- Whitelam (Molecular Foundry / Nat. Comm. 17:1189, 2026). Non-linear TC.
