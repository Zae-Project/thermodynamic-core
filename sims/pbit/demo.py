"""Generate the p-bit sigmoid sweep figure.

Plots the empirical P(up) vs. applied voltage for several beta values, with
the analytic sigmoid curves overlaid.

Writes figures/sigmoid_sweep.png.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from sims.pbit import sample_pbit, sigmoid


def main():
    rng = np.random.default_rng(20260422)
    voltages = np.linspace(-4.0, 4.0, 33)
    betas = (0.5, 1.0, 2.0)
    colors = ("#6b5344", "#8b7355", "#c4a574")
    n_samples = 5_000

    fig, ax = plt.subplots(figsize=(6.0, 3.8))
    v_smooth = np.linspace(-4.0, 4.0, 400)
    for beta, color in zip(betas, colors):
        p_emp = np.empty_like(voltages)
        for i, V in enumerate(voltages):
            s = sample_pbit(np.full(n_samples, V), beta=beta, rng=rng)
            p_emp[i] = s.mean()
        ax.plot(v_smooth, sigmoid(beta * v_smooth), color=color, linestyle=":",
                linewidth=1.2, label="sigmoid beta={:.1f}".format(beta))
        ax.plot(voltages, p_emp, "o", color=color, markersize=4,
                markerfacecolor="none")

    ax.set_xlabel("V (bias)")
    ax.set_ylabel("P(up)")
    ax.set_title("p-bit response: empirical samples vs. sigmoid(beta V)")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    fig.tight_layout()

    out_dir = os.path.join(os.path.dirname(__file__), "figures")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "sigmoid_sweep.png")
    fig.savefig(out_path, dpi=120)
    print("wrote", out_path)


if __name__ == "__main__":
    main()
