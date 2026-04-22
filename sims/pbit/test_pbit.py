"""Tests for the p-bit sigmoid sampler."""
import numpy as np
import pytest

from sims.pbit import PBit, sample_pbit, sigmoid


def test_sigmoid_matches_analytic():
    x = np.linspace(-20, 20, 41)
    out = sigmoid(x)
    ref = 1.0 / (1.0 + np.exp(-x))
    assert np.allclose(out, ref, atol=1e-12)


def test_sigmoid_numerical_stability():
    x = np.array([-1000.0, 1000.0])
    out = sigmoid(x)
    assert np.all(np.isfinite(out))
    assert out[0] < 1e-300 or out[0] == 0.0
    assert abs(out[1] - 1.0) < 1e-10


def test_pbit_empirical_probability(rng):
    """For a voltage sweep, empirical P(up) must match sigmoid(beta V)."""
    beta = 1.5
    voltages = np.array([-3.0, -1.0, -0.3, 0.0, 0.3, 1.0, 3.0])
    n_samples = 20_000
    for V in voltages:
        p_theory = 1.0 / (1.0 + np.exp(-beta * V))
        samples = sample_pbit(np.full(n_samples, V), beta=beta,
                              encoding="binary", rng=rng)
        p_emp = samples.mean()
        se = np.sqrt(p_theory * (1 - p_theory) / n_samples)
        assert abs(p_emp - p_theory) < max(0.02, 4 * se), (
            "V={:.2f} p_emp={:.4f} p_theory={:.4f}".format(V, p_emp, p_theory)
        )


def test_pbit_spin_encoding_has_expected_mean(rng):
    beta = 1.0
    V = np.full(15_000, 1.0)
    s = sample_pbit(V, beta=beta, encoding="spin", rng=rng)
    p_up = 1.0 / (1.0 + np.exp(-1.0))
    expected_mean = 2 * p_up - 1
    assert abs(s.mean() - expected_mean) < 0.02


def test_pbit_class_reproducible():
    pb = PBit(size=4, beta=1.0, seed=42)
    V = np.array([-1.0, -0.2, 0.2, 1.0])
    s1 = pb.sample(V)
    pb2 = PBit(size=4, beta=1.0, seed=42)
    s2 = pb2.sample(V)
    assert np.array_equal(s1, s2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
