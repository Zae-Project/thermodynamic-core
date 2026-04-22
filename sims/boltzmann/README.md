# sims/boltzmann

Restricted Boltzmann machine with block Gibbs sampling.

## Model

$$E(v, h) = -b^\top v - c^\top h - v^\top W h, \qquad p(v, h) \propto e^{-E(v, h)}$$

with $v \in \{0, 1\}^{n_v}$, $h \in \{0, 1\}^{n_h}$. The bipartite structure makes the conditionals factorize:

$$P(h_j = 1 \mid v) = \sigma(c_j + \textstyle\sum_i W_{ij} v_i), \quad P(v_i = 1 \mid h) = \sigma(b_i + \textstyle\sum_j W_{ij} h_j)$$

Sampling alternates full-vector updates of $v$ and $h$ in parallel. See [docs/architecture/software-stack.md §1.3](../../docs/architecture/software-stack.md) on block-Gibbs sampling and bipartite coloring as used by THRML.

## Files

| File | Purpose |
|---|---|
| `rbm.py` | `RBM` class with `sample_v`, `sample_h`, `gibbs_chain`, `exact_joint` (brute-force for small models) |
| `test_rbm.py` | Empirical joint matches analytic Boltzmann for tiny enumerable cases; conditional factorization check |

Sampling delegates to `sims.pbit.sample_pbit` so the RBM shares its p-bit primitive with the rest of the repo.

## Tests

```bash
python -m pytest sims/boltzmann -v
```

- `test_two_spin_rbm_matches_analytic`: 1 visible + 1 hidden, 4 total states. Total-variation distance between empirical and analytic joint distribution is under 0.03 at $2 \times 10^4$ samples.
- `test_three_visible_two_hidden_matches_analytic`: 32 states, TV distance under 0.05 at $4 \times 10^4$ samples.
- `test_conditional_factorization`: verifies $P(h \mid v)$ matches the factorized sigmoid directly at fixed $v$.

## Limits of this module

- No training. Contrastive Divergence or Onsager-Machlup training is the obvious next step, deferred to a JAX-based follow-up because autograd is worth having.
- `exact_joint` is brute force. It refuses to enumerate past $n_v + n_h = 14$ (16k states). Larger models only support Gibbs sampling, and correctness must be evaluated by other means (moment matching, pseudo-likelihood).
- No GPU path. The whole module is $O(n_v n_h)$ per Gibbs step in NumPy, which is fine for small test cases and prohibitive past a few thousand units. JAX + JIT is the escape hatch when we need it.
