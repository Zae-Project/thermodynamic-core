"""Benchmark the thermodynamic solver against np.linalg.solve.

Produces figures/benchmark_error.png showing relative error vs. system size
at a fixed integration budget.

This is a correctness benchmark, not a wall-clock benchmark. On a CPU the
thermodynamic solver is strictly slower than LAPACK; the point is to show
that the Langevin trajectory converges to A^{-1} b, which is what a physical
TC substrate would produce in O(N) relaxation time.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from sims.linalg import solve_thermodynamic


def main():
    rng = np.random.default_rng(20260422)
    sizes = (4, 8, 16, 32, 64)
    n_steps_per_size = 60_000
    errors = []
    for n in sizes:
        M = rng.standard_normal((n, n))
        A = M @ M.T + n * np.eye(n)
        b = rng.standard_normal(n)
        x_ref = np.linalg.solve(A, b)
        x_est, _ = solve_thermodynamic(A, b, n_steps=n_steps_per_size,
                                       kT=0.01, rng=rng)
        err = np.linalg.norm(x_est - x_ref) / np.linalg.norm(x_ref)
        errors.append(err)
        print("n={:3d}  rel_err={:.4e}".format(n, err))

    fig, ax = plt.subplots(figsize=(6.0, 3.8))
    ax.semilogy(sizes, errors, "o-", color="#8b7355", linewidth=1.4)
    ax.set_xlabel("system size N")
    ax.set_ylabel("relative error ||x_est - A^-1 b|| / ||A^-1 b||")
    ax.set_title("Thermodynamic solver vs. np.linalg.solve (n_steps={})".format(
        n_steps_per_size))
    ax.grid(True, alpha=0.3, which="both")
    fig.tight_layout()

    out_dir = os.path.join(os.path.dirname(__file__), "figures")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "benchmark_error.png")
    fig.savefig(out_path, dpi=120)
    print("wrote", out_path)


if __name__ == "__main__":
    main()
