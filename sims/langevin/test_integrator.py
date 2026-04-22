"""Tests for the overdamped Langevin integrator.

Check that the sampled trajectory of a harmonic oscillator reproduces the
Boltzmann equilibrium variance k_B T / k within tolerance.
"""
import numpy as np
import pytest

from sims.langevin import simulate, Harmonic, Quadratic


def test_harmonic_equilibrium_variance(rng):
    """Harmonic oscillator at kT=1, k=1 should sample N(0, 1) at equilibrium.

    Relaxation time tau = 1 / (mu k) = 1. Record every 100 steps = every tau
    to approximate IID samples. At N=3000 nearly-independent samples the
    standard error on the variance is ~0.026; 5% tolerance is comfortable.
    """
    pot = Harmonic(k=1.0)
    x0 = np.array([0.0])
    dt = 1e-2
    traj = simulate(pot, x0, n_steps=300_000, dt=dt, mu=1.0, kT=1.0,
                    rng=rng, burn_in=10_000, record_every=100)
    samples = traj[:, 0]
    emp_var = samples.var()
    assert abs(emp_var - 1.0) < 0.05, (
        "empirical variance {:.4f} not within 5% of 1.0".format(emp_var)
    )
    assert abs(samples.mean()) < 0.05, (
        "empirical mean {:.4f} not within 0.05 of 0".format(samples.mean())
    )


def test_harmonic_temperature_scaling(rng):
    """Variance should scale linearly with kT for fixed k."""
    pot = Harmonic(k=2.0)
    x0 = np.array([0.0])
    variances = {}
    for kT in (0.5, 1.0, 2.0):
        traj = simulate(pot, x0, n_steps=80_000, dt=1e-2, mu=1.0, kT=kT,
                        rng=rng, burn_in=8_000, record_every=10)
        variances[kT] = traj[:, 0].var()
    for kT, var in variances.items():
        expected = kT / 2.0
        assert abs(var - expected) / expected < 0.08, (
            "kT={:.2f} var={:.4f} expected={:.4f}".format(kT, var, expected)
        )


def test_quadratic_equilibrium_mean_solves_linear_system(rng):
    """For V = 0.5 x^T A x - b^T x, equilibrium mean equals A^{-1} b."""
    n = 4
    M = rng.standard_normal((n, n))
    A = M @ M.T + n * np.eye(n)  # SPD
    b = rng.standard_normal(n)
    x_exact = np.linalg.solve(A, b)

    pot = Quadratic(A, b)
    traj = simulate(pot, np.zeros(n), n_steps=60_000, dt=5e-3, mu=1.0,
                    kT=0.05, rng=rng, burn_in=20_000, record_every=10)
    x_est = traj.mean(axis=0)
    rel_err = np.linalg.norm(x_est - x_exact) / np.linalg.norm(x_exact)
    assert rel_err < 0.1, (
        "relative error {:.4f} in Langevin-estimated A^-1 b".format(rel_err)
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
