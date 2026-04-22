"""Generate the diagnostic figure for the Langevin integrator.

Runs a harmonic oscillator at three temperatures and overlays the empirical
marginal distribution with the analytic Boltzmann distribution
p(x) ~ exp(-k x^2 / (2 k_B T)).

Writes figures/harmonic_boltzmann.png.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from sims.langevin import Harmonic, simulate


def main():
    rng = np.random.default_rng(20260422)
    pot = Harmonic(k=1.0)
    x0 = np.zeros(1)

    fig, ax = plt.subplots(figsize=(6.0, 3.8))
    colors = ("#6b5344", "#8b7355", "#c4a574")
    kTs = (0.5, 1.0, 2.0)

    x_grid = np.linspace(-5, 5, 400)
    for kT, color in zip(kTs, colors):
        traj = simulate(pot, x0, n_steps=60_000, dt=1e-2, mu=1.0, kT=kT,
                        rng=rng, burn_in=10_000, record_every=5)
        samples = traj[:, 0]
        ax.hist(samples, bins=80, density=True, histtype="step",
                color=color, linewidth=1.4,
                label="empirical kT={:.1f}".format(kT))
        sigma2 = kT / pot.k
        analytic = np.exp(-x_grid ** 2 / (2 * sigma2)) / np.sqrt(2 * np.pi * sigma2)
        ax.plot(x_grid, analytic, color=color, linestyle=":", linewidth=1.2)

    ax.set_xlabel("x")
    ax.set_ylabel("p(x)")
    ax.set_title("Harmonic oscillator: Langevin samples vs. Boltzmann")
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()

    out_dir = os.path.join(os.path.dirname(__file__), "figures")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "harmonic_boltzmann.png")
    fig.savefig(out_path, dpi=120)
    print("wrote", out_path)


if __name__ == "__main__":
    main()
