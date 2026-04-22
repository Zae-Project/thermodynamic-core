"""Tests for the RBM block-Gibbs sampler."""
import numpy as np
import pytest

from sims.boltzmann import RBM


def _empirical_joint(vs, hs, n_visible, n_hidden):
    """Return a dict {(v_tuple, h_tuple): frequency} from chain samples."""
    counts = {}
    for v, h in zip(vs, hs):
        key = (tuple(int(x) for x in v), tuple(int(x) for x in h))
        counts[key] = counts.get(key, 0) + 1
    total = len(vs)
    return {k: v / total for k, v in counts.items()}


def _tv_distance(p_emp, states_probs):
    probs = {state: prob for state, prob in states_probs}
    keys = set(probs) | set(p_emp)
    return 0.5 * sum(abs(probs.get(k, 0.0) - p_emp.get(k, 0.0)) for k in keys)


def test_two_spin_rbm_matches_analytic():
    """1 visible + 1 hidden: enumerable exactly. Empirical TV < 0.03."""
    W = np.array([[1.2]])
    b = np.array([-0.3])
    c = np.array([0.5])
    rbm = RBM(1, 1, W=W, b=b, c=c, seed=20260422)

    vs, hs = rbm.gibbs_chain(n_samples=20_000, burn_in=500, thin=1)
    p_emp = _empirical_joint(vs, hs, 1, 1)
    states, probs = rbm.exact_joint()
    states_probs = list(zip(states, probs))
    tv = _tv_distance(p_emp, states_probs)
    assert tv < 0.03, "TV distance {:.4f} too large".format(tv)


def test_three_visible_two_hidden_matches_analytic(rng):
    """Larger enumerable RBM (2^5 = 32 states). Empirical TV < 0.05."""
    W = rng.standard_normal((3, 2)) * 0.7
    b = rng.standard_normal(3) * 0.3
    c = rng.standard_normal(2) * 0.3
    rbm = RBM(3, 2, W=W, b=b, c=c, seed=11)

    vs, hs = rbm.gibbs_chain(n_samples=40_000, burn_in=2000, thin=1)
    p_emp = _empirical_joint(vs, hs, 3, 2)
    states, probs = rbm.exact_joint()
    tv = _tv_distance(p_emp, list(zip(states, probs)))
    assert tv < 0.05, "TV distance {:.4f} too large".format(tv)


def test_conditional_factorization(rng):
    """Independent conditional sampling: P(h | v) = prod_j sigmoid(c_j + W_j^T v)."""
    W = rng.standard_normal((4, 5)) * 0.5
    b = rng.standard_normal(4)
    c = rng.standard_normal(5)
    rbm = RBM(4, 5, W=W, b=b, c=c, seed=0)
    v = np.array([1, 0, 1, 0])
    p_theory = 1.0 / (1.0 + np.exp(-(c + v @ W)))

    hs = np.array([rbm.sample_h(v) for _ in range(20_000)])
    p_emp = hs.mean(axis=0)
    assert np.max(np.abs(p_emp - p_theory)) < 0.02


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
