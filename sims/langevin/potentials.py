"""Potential energy landscapes V_theta(x) and their gradients.

Each potential exposes `energy(x)` and `grad(x)`. `x` is a 1-D array (state
vector); `grad(x)` returns dV/dx as an array of the same shape.
"""
import numpy as np


class Harmonic:
    """V(x) = 0.5 * k * ||x - x0||^2.

    Equilibrium marginal under overdamped Langevin at temperature T is
    Gaussian with mean x0 and variance k_B T / k per component.
    """

    def __init__(self, k=1.0, x0=None):
        self.k = float(k)
        self.x0 = None if x0 is None else np.asarray(x0, dtype=float)

    def energy(self, x):
        dx = x if self.x0 is None else x - self.x0
        return 0.5 * self.k * np.sum(dx * dx)

    def grad(self, x):
        dx = x if self.x0 is None else x - self.x0
        return self.k * dx


class Quadratic:
    """V(x) = 0.5 * x^T A x - b^T x, with A symmetric positive definite.

    Minimum at A^{-1} b. Used by the thermodynamic linear algebra solver.
    """

    def __init__(self, A, b):
        A = np.asarray(A, dtype=float)
        if not np.allclose(A, A.T, atol=1e-10):
            raise ValueError("A must be symmetric for a well-defined potential")
        self.A = A
        self.b = np.asarray(b, dtype=float)

    def energy(self, x):
        return 0.5 * x @ self.A @ x - self.b @ x

    def grad(self, x):
        return self.A @ x - self.b


class DoubleWell:
    """V(x) = a * (x^2 - 1)^2 in one dimension.

    Two symmetric minima at x = +/- 1 separated by a barrier of height `a`.
    Used to exercise stochastic barrier crossing.
    """

    def __init__(self, a=1.0):
        self.a = float(a)

    def energy(self, x):
        x2 = np.sum(x * x)
        return self.a * (x2 - 1.0) ** 2

    def grad(self, x):
        x2 = np.sum(x * x)
        return 4.0 * self.a * (x2 - 1.0) * x
