"""Probabilistic-bit (p-bit) sigmoid sampler.

A p-bit is the elementary stochastic primitive of thermodynamic hardware. Its
state s in {0, 1} (or {-1, +1}) fluctuates with a Bernoulli probability

    P(s = 1) = sigmoid(beta * V)

where V is an applied bias voltage and beta = 1 / (k_B T). See
docs/architecture/hardware-primitives.md for the physical motivation
(MTJ spintronic free layer, subthreshold-CMOS stochastic transistors).

This module provides a clean reference sampler that a hardware or THRML
implementation must reproduce in distribution.
"""
import numpy as np


def sigmoid(x):
    """Numerically stable sigmoid."""
    out = np.empty_like(x, dtype=float)
    pos = x >= 0
    neg = ~pos
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    ex = np.exp(x[neg])
    out[neg] = ex / (1.0 + ex)
    return out


def sample_pbit(V, beta=1.0, encoding="binary", rng=None):
    """Draw samples from a vector of independent p-bits.

    Parameters
    ----------
    V : array_like, bias voltages (any shape)
    beta : inverse temperature, scalar
    encoding : "binary" -> {0, 1}; "spin" -> {-1, +1}
    rng : numpy Generator

    Returns
    -------
    s : int8 array with the same shape as V
    """
    V = np.asarray(V, dtype=float)
    if rng is None:
        rng = np.random.default_rng()
    p = sigmoid(beta * V)
    up = rng.random(size=V.shape) < p
    if encoding == "binary":
        return up.astype(np.int8)
    if encoding == "spin":
        return (2 * up - 1).astype(np.int8)
    raise ValueError("encoding must be 'binary' or 'spin'")


class PBit:
    """Stateful p-bit array with persistent RNG, useful for long sample runs."""

    def __init__(self, size, beta=1.0, encoding="binary", seed=None):
        self.size = int(size)
        self.beta = float(beta)
        if encoding not in ("binary", "spin"):
            raise ValueError("encoding must be 'binary' or 'spin'")
        self.encoding = encoding
        self.rng = np.random.default_rng(seed)

    def sample(self, V):
        V = np.asarray(V, dtype=float)
        if V.shape[-1] != self.size:
            raise ValueError(
                "V last dim {} must match p-bit size {}".format(V.shape[-1], self.size)
            )
        return sample_pbit(V, beta=self.beta, encoding=self.encoding, rng=self.rng)

    def p_up(self, V):
        return sigmoid(self.beta * np.asarray(V, dtype=float))
