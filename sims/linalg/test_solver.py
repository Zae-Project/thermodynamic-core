"""Tests for the thermodynamic linear-system solver."""
import numpy as np
import pytest

from sims.linalg import solve_thermodynamic


def _random_spd(n, rng):
    M = rng.standard_normal((n, n))
    return M @ M.T + n * np.eye(n)


def test_solves_small_spd(rng):
    n = 8
    A = _random_spd(n, rng)
    b = rng.standard_normal(n)
    x_ref = np.linalg.solve(A, b)
    x_est, info = solve_thermodynamic(A, b, n_steps=40_000, kT=0.02, rng=rng)
    rel_err = np.linalg.norm(x_est - x_ref) / np.linalg.norm(x_ref)
    assert rel_err < 0.05, (
        "rel err {:.4f} for n={} (info={})".format(rel_err, n, info)
    )


def test_solves_medium_spd(rng):
    n = 32
    A = _random_spd(n, rng)
    b = rng.standard_normal(n)
    x_ref = np.linalg.solve(A, b)
    x_est, _ = solve_thermodynamic(A, b, n_steps=80_000, kT=0.01, rng=rng)
    rel_err = np.linalg.norm(x_est - x_ref) / np.linalg.norm(x_ref)
    assert rel_err < 0.05, "rel err {:.4f} for n={}".format(rel_err, n)


def test_diagonal_A_matches_closed_form(rng):
    """For diagonal A the problem factorizes; this is a cheap sanity check."""
    diag = rng.uniform(1.0, 4.0, size=6)
    A = np.diag(diag)
    b = rng.standard_normal(6)
    x_ref = b / diag
    x_est, _ = solve_thermodynamic(A, b, n_steps=30_000, kT=0.02, rng=rng)
    assert np.allclose(x_est, x_ref, rtol=0.05, atol=0.02)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
