# brain_translator. Atlas template to TC energy landscape

The first runnable, falsifiable cross-pillar deliverable between
**brain-emulation** (Interface) and **thermodynamic-core** (Substrate). It
implements the translation pipeline sketched in the contracts
([`../../docs/integration/with-brain-emulation.md`](../../docs/integration/with-brain-emulation.md)
section 2.3, and the zae-docs
`integration/thermodynamic-brain-emulation-interface.md` section 3) and the
`Allen Motor Cortex activity` row of the shared-benchmark table (section 4),
on the thermodynamic-core side.

## What it does

1. Loads a brain-emulation atlas template (clusters of excitatory and
   inhibitory neuron groups with probabilistic, weighted, typed connectivity).
2. Builds a neuron-level Ising energy landscape `(J, b)` from it.
3. Samples the landscape with single-site Gibbs dynamics, the reference a
   p-bit / TSU substrate must reproduce in distribution.
4. Maps Ising samples back to firing rates, `r_i = (s_i + 1)/2 * r_max`.

## Energy and sign convention (matches the contract)

State `s_i in {-1, +1}`. Energy

```
E(s) = sum_{i<j} J_ij s_i s_j  -  sum_i b_i s_i
```

with `excitatory synapse (w>0) -> J_ij = -w` and `inhibitory -> J_ij = +w`, so
the Gibbs conditional is `P(s_i=+1) = sigmoid(2 beta (b_i - (J s)_i))`. An
active excitatory neighbour raises firing probability, an active inhibitory
neighbour lowers it. The unit test `test_gibbs_matches_exact_marginals`
verifies the sampler reproduces the exact Boltzmann marginals on a small case,
which pins this convention down.

## Two honest approximations

- **Directed -> symmetric.** Biological connectivity is directed; an
  equilibrium energy landscape needs symmetric coupling, so the translator
  symmetrizes `J = (M + M^T)/2`.
- **Coupling normalization.** The literal `w -> J` mapping produces a strongly
  coupled ferromagnet (mean row-coupling ~45) whose only operating points are
  near-random or saturated. The default `normalize="field"` divides `J` by its
  mean absolute row sum so the local field is O(1) and `beta` sets a usable
  operating point. This is balanced-network scaling
  (van Vreeswijk and Sompolinsky 1996) as a symmetric scalar normalization.
  Pass `normalize=None` for the raw mapping.

## Running the benchmark

```
python -m sims.brain_translator.benchmark
```

At the default balanced operating point (`beta=0.7`, `b0=-0.3`) the Allen M1
template (438 neurons, 3 layers, 350 excitatory / 88 inhibitory) gives:

- Biologically correct rate ordering: excitatory ~12 Hz < inhibitory ~21 Hz.
- Layer variation (L6 highest, L5B lowest).
- An inhibition-gain sweep where scaling inhibition up suppresses excitatory
  firing monotonically (~12 -> ~8 Hz).

## The honest finding

At **native** strength (gain 1.0) the inhibition effect is small: after
symmetric normalization the 88 inhibitory units carry a minority of the
coupling mass, so this equilibrium Ising model under-represents inhibition's
functional weight. That is a concrete, falsifiable limitation, the contract's
biological-fidelity gap (section 5) made measurable. It motivates the next
steps: asymmetric or non-equilibrium TC dynamics, or inhibition-aware
weighting, before the energy-landscape translation can claim biological
fidelity beyond rate ordering.

## What is NOT claimed

- No Brian2 spike-train match or absolute-Hz claim. A binary Ising unit is
  simpler than a conductance-based neuron.
- The Brian2 side of the shared benchmark is run only if `brian2` is importable
  (it is the contract's implementation home for the biological side). This
  module provides the thermodynamic-core reference; no Brian2 numbers are
  fabricated when it is absent.

## Files

- `translator.py`. `BrainNetwork` (template -> J, b), `IsingSampler`
  (single-site Gibbs), `samples_to_rates`, `exact_marginals`.
- `benchmark.py`. The Allen M1 run and inhibition-gain sweep.
- `test_translator.py`. Counts/Dale, sign convention, rate mapping, Gibbs vs
  exact marginals, inhibition-constrains-excitation on a small net.
- `data/allen_motor_cortex.json`. Bundled mirror of the brain-emulation
  template (provenance noted in the file).
