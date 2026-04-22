"""Overdamped Langevin integrator (Euler-Maruyama).

Integrates the stochastic differential equation

    dx/dt = -mu * grad V(x) + sqrt(2 * mu * k_B * T) * eta(t)

with Gaussian white noise `eta`, using the Euler-Maruyama scheme

    x_{n+1} = x_n - mu * grad V(x_n) * dt + sqrt(2 * mu * k_B * T * dt) * z

where z ~ N(0, I). Natural units: k_B = 1. Temperature `kT` is passed in
directly as the thermal energy scale.
"""
import numpy as np


def step(potential, x, dt, mu=1.0, kT=1.0, rng=None):
    """Advance one Euler-Maruyama step. Pure function, no state mutation."""
    if rng is None:
        rng = np.random.default_rng()
    noise = rng.standard_normal(size=x.shape)
    drift = -mu * potential.grad(x) * dt
    diffusion = np.sqrt(2.0 * mu * kT * dt) * noise
    return x + drift + diffusion


def simulate(potential, x0, n_steps, dt, mu=1.0, kT=1.0, rng=None,
             record_every=1, burn_in=0):
    """Run `n_steps` Langevin steps; return trajectory of recorded states.

    Parameters
    ----------
    potential : object with `grad(x)` method
    x0 : initial state, 1-D array
    n_steps : total Euler-Maruyama steps to run
    dt : integration step size
    mu : mobility coefficient
    kT : thermal energy k_B * T in natural units
    rng : numpy Generator; fresh default if None
    record_every : stride for saving states
    burn_in : number of initial steps to discard before recording

    Returns
    -------
    trajectory : array of shape (n_recorded, dim)
    """
    if rng is None:
        rng = np.random.default_rng()
    x = np.asarray(x0, dtype=float).copy()

    for _ in range(burn_in):
        x = step(potential, x, dt, mu=mu, kT=kT, rng=rng)

    n_record = (n_steps + record_every - 1) // record_every
    trajectory = np.empty((n_record, x.size), dtype=float)
    idx = 0
    for i in range(n_steps):
        x = step(potential, x, dt, mu=mu, kT=kT, rng=rng)
        if i % record_every == 0:
            trajectory[idx] = x
            idx += 1
    return trajectory[:idx]
