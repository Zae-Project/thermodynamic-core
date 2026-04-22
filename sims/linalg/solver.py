"""Thermodynamic linear-system solver.

Solves A x = b (A symmetric positive definite) by simulating an overdamped
Langevin system in the quadratic potential

    V(x) = 0.5 * x^T A x - b^T x

whose unique minimum is x* = A^{-1} b. At low temperature kT the sampled
trajectory concentrates around x*; averaging the trajectory estimates x* to
statistical precision.

This is the NumPy reference for the "linear solve in O(N) relaxation time"
claim in docs/algorithms/thermodynamic-linear-algebra.md (Aifer et al. 2024).
The O(N) claim refers to physical relaxation-time scaling, not to wall-clock
speed on a CPU, which is dominated by the matrix-vector product in grad V.
"""
import numpy as np

from sims.langevin import Quadratic, simulate


def solve_thermodynamic(A, b, n_steps=20_000, dt=None, mu=1.0, kT=0.05,
                        burn_in_frac=0.3, record_every=1, rng=None):
    """Estimate A^{-1} b by trajectory averaging under Langevin dynamics.

    Parameters
    ----------
    A : (N, N) symmetric positive definite
    b : (N,)
    n_steps : total integration steps
    dt : step size; if None, a conservative value 0.2 / lambda_max(A) is used
    mu : mobility
    kT : thermal energy; lower kT tightens concentration around x*, at the
         cost of slower mixing across the well
    burn_in_frac : fraction of n_steps to discard before averaging
    record_every : stride for averaging
    rng : numpy Generator

    Returns
    -------
    x_est : (N,) time-averaged trajectory estimate of A^{-1} b
    info : dict with "dt", "burn_in", "n_samples"
    """
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float)
    n = b.size
    if dt is None:
        lam_max = np.linalg.eigvalsh(A)[-1]
        dt = 0.2 / (mu * lam_max)
    burn_in = int(burn_in_frac * n_steps)

    pot = Quadratic(A, b)
    x0 = np.zeros(n)
    traj = simulate(pot, x0, n_steps=n_steps, dt=dt, mu=mu, kT=kT,
                    rng=rng, burn_in=burn_in, record_every=record_every)
    x_est = traj.mean(axis=0)
    return x_est, {"dt": dt, "burn_in": burn_in, "n_samples": traj.shape[0]}
