"""Restricted Boltzmann machine with block Gibbs sampling.

A Restricted Boltzmann machine (RBM) is a bipartite Markov random field
over visible units v in {0, 1}^n_v and hidden units h in {0, 1}^n_h with
energy

    E(v, h) = - b^T v - c^T h - v^T W h

and Boltzmann distribution p(v, h) ~ exp(-E(v, h)). The bipartite structure
makes the conditional distributions factorized,

    P(h_j = 1 | v) = sigmoid(c_j + sum_i W_{ij} v_i)
    P(v_i = 1 | h) = sigmoid(b_i + sum_j W_{ij} h_j)

so a full visible or hidden vector can be sampled in a single vectorized p-bit
call. That is "block Gibbs" on a bipartite coloring. It maps directly onto
THRML's block-Gibbs sampler on TSU hardware (see
docs/architecture/software-stack.md section 1.3).

This NumPy implementation does not train the RBM. It only samples from a
given parameterization so we can verify that the sampler reproduces the
analytic Boltzmann distribution on small cases where the partition function
is tractable.
"""
import itertools

import numpy as np

from sims.pbit import sample_pbit, sigmoid


class RBM:
    """Restricted Boltzmann machine with block Gibbs sampling."""

    def __init__(self, n_visible, n_hidden, W=None, b=None, c=None, seed=None):
        self.n_visible = int(n_visible)
        self.n_hidden = int(n_hidden)
        self.W = (np.zeros((n_visible, n_hidden)) if W is None
                  else np.asarray(W, dtype=float))
        self.b = np.zeros(n_visible) if b is None else np.asarray(b, dtype=float)
        self.c = np.zeros(n_hidden) if c is None else np.asarray(c, dtype=float)
        self.rng = np.random.default_rng(seed)

    def energy(self, v, h):
        v = np.asarray(v, dtype=float)
        h = np.asarray(h, dtype=float)
        return -(v @ self.b) - (h @ self.c) - v @ self.W @ h

    def p_h_given_v(self, v):
        return sigmoid(self.c + v @ self.W)

    def p_v_given_h(self, h):
        return sigmoid(self.b + h @ self.W.T)

    def sample_h(self, v):
        return sample_pbit(self.c + v @ self.W, beta=1.0,
                           encoding="binary", rng=self.rng).astype(np.int8)

    def sample_v(self, h):
        return sample_pbit(self.b + h @ self.W.T, beta=1.0,
                           encoding="binary", rng=self.rng).astype(np.int8)

    def gibbs_chain(self, n_samples, burn_in=200, thin=1, v0=None):
        """Run a block-Gibbs chain and return (visible_samples, hidden_samples).

        Each returned array has shape (n_samples, n_visible) or
        (n_samples, n_hidden).
        """
        if v0 is None:
            v = self.rng.integers(0, 2, size=self.n_visible).astype(np.int8)
        else:
            v = np.asarray(v0, dtype=np.int8).copy()

        for _ in range(burn_in):
            h = self.sample_h(v)
            v = self.sample_v(h)

        vs = np.empty((n_samples, self.n_visible), dtype=np.int8)
        hs = np.empty((n_samples, self.n_hidden), dtype=np.int8)
        for i in range(n_samples):
            for _ in range(thin):
                h = self.sample_h(v)
                v = self.sample_v(h)
            vs[i] = v
            hs[i] = h
        return vs, hs

    def exact_joint(self):
        """Brute-force joint distribution p(v, h). Only tractable for tiny RBMs.

        Returns
        -------
        states : list of (v_tuple, h_tuple)
        probs : array of joint probabilities aligned with states
        """
        nv, nh = self.n_visible, self.n_hidden
        if nv + nh > 14:
            raise ValueError(
                "refusing brute-force enumeration for n_visible + n_hidden > 14"
            )
        states = []
        energies = []
        for v in itertools.product([0, 1], repeat=nv):
            for h in itertools.product([0, 1], repeat=nh):
                v_arr = np.asarray(v, dtype=float)
                h_arr = np.asarray(h, dtype=float)
                states.append((tuple(v), tuple(h)))
                energies.append(self.energy(v_arr, h_arr))
        energies = np.asarray(energies)
        logZ = _logsumexp(-energies)
        probs = np.exp(-energies - logZ)
        return states, probs


def _logsumexp(x):
    m = np.max(x)
    return m + np.log(np.sum(np.exp(x - m)))
