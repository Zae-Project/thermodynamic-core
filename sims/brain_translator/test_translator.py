"""Tests for the atlas -> TC energy-landscape translator and Gibbs sampler.

Fast by design: tiny hand-built templates, one small exact-distribution check.
The full Allen Motor Cortex run lives in benchmark.py, not here.
"""
import os

import numpy as np
import pytest

from sims.brain_translator import (
    BrainNetwork,
    IsingSampler,
    exact_marginals,
    load_template,
    samples_to_rates,
)

HERE = os.path.dirname(__file__)
ALLEN = os.path.join(HERE, "data", "allen_motor_cortex.json")


def _two_neuron_template(syn_type, weight=0.9):
    return {
        "clusters": [
            {
                "id": "c",
                "neuronGroups": [
                    {"preset": "pyramidal", "count": 1},
                    {"preset": "sink", "count": 1},
                ],
                "internalConnectivity": [
                    {"from": "pyramidal", "to": "sink",
                     "probability": 1.0, "type": syn_type, "weight": weight},
                ],
            }
        ],
        "connections": [],
    }


def test_allen_counts_and_dale():
    net = BrainNetwork(load_template(ALLEN), seed=0)
    assert net.n == 438
    assert int(net.is_excitatory.sum()) == 350      # pyramidal: 120+140+90
    assert int((~net.is_excitatory).sum()) == 88     # basket+chandelier
    # equilibrium landscape must be symmetric with no self-coupling
    assert np.allclose(net.J, net.J.T)
    assert np.allclose(np.diag(net.J), 0.0)


def test_sign_convention_matches_contract():
    # literal mapping (no normalization): excitatory -> J = -w/2, inhibitory -> +w/2
    exc = BrainNetwork(_two_neuron_template("excitatory"), seed=1, normalize=None)
    inh = BrainNetwork(_two_neuron_template("inhibitory"), seed=1, normalize=None)
    assert exc.J[0, 1] == pytest.approx(-0.9 / 2)
    assert inh.J[0, 1] == pytest.approx(+0.9 / 2)
    # sign is preserved under the default "field" normalization
    exc_n = BrainNetwork(_two_neuron_template("excitatory"), seed=1)
    inh_n = BrainNetwork(_two_neuron_template("inhibitory"), seed=1)
    assert exc_n.J[0, 1] < 0
    assert inh_n.J[0, 1] > 0


def test_rate_mapping_bounds():
    allup = np.ones((10, 3), dtype=np.int8)
    alldown = -np.ones((10, 3), dtype=np.int8)
    assert np.allclose(samples_to_rates(allup, r_max=50.0), 50.0)
    assert np.allclose(samples_to_rates(alldown, r_max=50.0), 0.0)


def test_gibbs_matches_exact_marginals():
    rng = np.random.default_rng(7)
    n = 5
    A = rng.normal(scale=0.5, size=(n, n))
    J = 0.5 * (A + A.T)
    np.fill_diagonal(J, 0.0)
    b = rng.normal(scale=0.4, size=n)
    beta = 0.8

    exact = exact_marginals(J, b, beta=beta)
    sampler = IsingSampler(J, b, beta=beta, seed=3)
    samples = sampler.chain(n_samples=8000, burn_in=500, thin=1)
    emp = ((samples == 1).mean(axis=0))
    assert np.max(np.abs(emp - exact)) < 0.03


def test_inhibition_constrains_excitatory_activity():
    # small E/I column: 8 excitatory (pyramidal) + 4 inhibitory (basket)
    template = {
        "clusters": [
            {
                "id": "col",
                "neuronGroups": [
                    {"preset": "pyramidal", "count": 8},
                    {"preset": "basket", "count": 4},
                ],
                "internalConnectivity": [
                    {"from": "pyramidal", "to": "pyramidal",
                     "probability": 0.5, "type": "excitatory", "weight": 0.6},
                    {"from": "pyramidal", "to": "basket",
                     "probability": 0.6, "type": "excitatory", "weight": 0.6},
                    {"from": "basket", "to": "pyramidal",
                     "probability": 0.8, "type": "inhibitory", "weight": 0.9},
                ],
            }
        ],
        "connections": [],
    }
    net = BrainNetwork(template, seed=2)
    exc = net.is_excitatory
    b = net.bias(b0=0.2)            # mild drive toward firing

    full = IsingSampler(net.J, b, beta=1.0, seed=5)
    rate_full = samples_to_rates(full.chain(600, burn_in=300, thin=1))

    J_no_inh = net.J.copy()
    inh = ~exc
    J_no_inh[np.ix_(inh, np.arange(net.n))] = 0.0
    J_no_inh[np.ix_(np.arange(net.n), inh)] = 0.0
    no_inh = IsingSampler(J_no_inh, b, beta=1.0, seed=5)
    rate_no_inh = samples_to_rates(no_inh.chain(600, burn_in=300, thin=1))

    # removing inhibition must raise mean excitatory firing
    assert rate_no_inh[exc].mean() > rate_full[exc].mean()
