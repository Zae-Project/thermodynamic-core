# sims/ . Phase 2 Simulations

**Status**: Phase 2 foundation. NumPy first, JAX deferred.

This directory holds executable simulations of the primitives described in `docs/`. The goal is to validate the theoretical claims in the architecture and algorithm docs against reference implementations, before any hardware target exists.

## Layout

```
sims/
├── README.md              # This file
├── requirements.txt       # Pinned dependencies
├── conftest.py            # pytest config (deterministic seeds)
├── langevin/              # Overdamped Langevin integrator + potentials
├── pbit/                  # Sigmoid p-bit sampler
├── linalg/                # Thermodynamic solver for Ax = b
└── boltzmann/             # Restricted Boltzmann machine, block Gibbs
```

Each submodule has its own `README.md` with the physics, the test, and the figures produced.

## Why NumPy, not JAX

The near-term targets (Langevin integrator, p-bit sigmoid, quadratic-energy solver, small RBM) are sampling primitives. None of them require automatic differentiation. NumPy gives zero install friction on Windows and identical numerics on any machine the tests run on.

JAX becomes necessary when Phase 2 picks up Onsager-Machlup gradient-descent training (Whitelam, PNAS 2026). At that point the training loop migrates to JAX; the primitives here stay as NumPy reference implementations, and the two must agree on the forward pass. Treat this directory as the ground truth the JAX port has to reproduce, not as an intermediate step that gets deleted.

## Running

```bash
python -m pip install -r sims/requirements.txt
python -m pytest sims/ -v
```

Individual demos produce PNG figures in `sims/<module>/figures/`. Figures are committed so a reviewer does not need to re-run the code to see the result.

## Relation to THRML

[THRML](https://docs.thrml.ai/) is Extropic's JAX library and the eventual compile target for hardware. The NumPy code here is not a THRML replacement, it is a reference that a THRML or hardware implementation must match within tolerance. See [docs/architecture/software-stack.md](../docs/architecture/software-stack.md).
