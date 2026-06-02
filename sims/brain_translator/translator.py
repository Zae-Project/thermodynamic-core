"""Brian2 atlas template -> thermodynamic-core energy landscape translator.

This is the Phase 2 cross-pillar deliverable described in
`docs/integration/with-brain-emulation.md` section 2.3 and the zae-docs
contract `integration/thermodynamic-brain-emulation-interface.md` section 3.
It converts a brain-emulation atlas template (clusters of excitatory and
inhibitory neuron groups with probabilistic, weighted connectivity) into a
thermodynamic-core Ising energy landscape (J, b) and samples it with
single-site Gibbs dynamics, the reference behaviour a p-bit substrate must
reproduce in distribution.

Sign and energy convention (matches the contract's spin convention)
-------------------------------------------------------------------
State s_i in {-1, +1}. Energy

    E(s) = sum_{i<j} J_ij s_i s_j  -  sum_i b_i s_i

so a positive bias b_i favours s_i = +1 (firing). Per the contract:

    excitatory synapse (w > 0)  ->  J_ij = -w   (lowers energy for aligned s_i, s_j)
    inhibitory synapse (w > 0)  ->  J_ij = +w   (raises energy for aligned s_i, s_j)

The single-site Gibbs conditional that follows from this energy is

    P(s_i = +1 | rest) = sigmoid( 2 beta ( b_i - sum_j J_ij s_j ) )

so an active excitatory neighbour (J_ij < 0) raises the firing probability and
an active inhibitory neighbour (J_ij > 0) lowers it, as required.

Two honest approximations, both already flagged as technology gaps in the
contract (section 5):

1. Biological connectivity is directed; an equilibrium energy landscape needs a
   symmetric coupling. We accumulate the directed signed couplings into M and
   symmetrize, J = (M + M^T) / 2.
2. A binary Ising unit is far simpler than a conductance-based neuron. This
   translator reproduces excitatory/inhibitory balance and co-activation
   structure, not detailed spike-train dynamics.

The Ising sample maps back to a firing rate per the contract visualization
rule, r_i = (s_i + 1) / 2 * r_max.
"""
import itertools
import json

import numpy as np

from sims.pbit import sigmoid


def load_template(path):
    """Load an atlas template JSON (brain-emulation `brain_region_maps` format)."""
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _preset_polarities(template):
    """Infer each neuron preset's Dale polarity from the connectivity rules.

    A preset that appears as the source of an "excitatory" rule is excitatory,
    one that sources "inhibitory" rules is inhibitory. Raises if a preset is
    used with both polarities (a Dale's-law violation in the template).
    """
    polarity = {}

    def record(rule_list):
        for rule in rule_list:
            src = rule["from"]
            pol = +1 if rule["type"] == "excitatory" else -1
            if src in polarity and polarity[src] != pol:
                raise ValueError(
                    "preset '{}' used as both excitatory and inhibitory "
                    "source (Dale violation)".format(src)
                )
            polarity[src] = pol

    for cluster in template["clusters"]:
        record(cluster.get("internalConnectivity", []))
    for conn in template.get("connections", []):
        record(conn.get("connectivity", []))
    return polarity


class BrainNetwork:
    """A neuron-level realization of an atlas template.

    Attributes
    ----------
    n : int                      total neuron count
    is_excitatory : bool array   length n, True for excitatory neurons
    layer : int array            length n, cluster index per neuron
    layer_names : list[str]      cluster id per index
    groups : dict                (cluster_id, preset) -> list of neuron indices
    """

    def __init__(self, template, seed=0, normalize="field"):
        """Build a neuron-level network from an atlas template.

        normalize : how to scale the coupling matrix.
            None      raw weight->J mapping (literal contract mapping). Produces
                      a strongly coupled ferromagnet whose mean field is O(K*w),
                      so the only operating points are near-random or saturated.
            "field"   divide J by the mean absolute row sum, so the typical
                      local field is O(1) and beta sets a usable operating
                      point. This is the balanced-network scaling
                      (van Vreeswijk and Sompolinsky 1996) expressed as a
                      symmetric scalar normalization. Default.
        """
        self.template = template
        self.rng = np.random.default_rng(seed)
        self.normalize = normalize
        self.polarity = _preset_polarities(template)
        self._enumerate_neurons()
        self._build_coupling()

    def _enumerate_neurons(self):
        self.groups = {}
        is_exc = []
        layer = []
        layer_names = []
        idx = 0
        for li, cluster in enumerate(self.template["clusters"]):
            layer_names.append(cluster["id"])
            for group in cluster["neuronGroups"]:
                preset = group["preset"]
                count = int(group["count"])
                pol = self.polarity.get(preset)
                if pol is None:
                    # preset never sources a synapse; default by name heuristic
                    pol = +1 if preset == "pyramidal" else -1
                ids = list(range(idx, idx + count))
                self.groups[(cluster["id"], preset)] = ids
                is_exc.extend([pol > 0] * count)
                layer.extend([li] * count)
                idx += count
        self.n = idx
        self.is_excitatory = np.asarray(is_exc, dtype=bool)
        self.layer = np.asarray(layer, dtype=int)
        self.layer_names = layer_names

    def _add_rule(self, M, src_ids, dst_ids, rule):
        """Accumulate one connectivity rule into the directed coupling M."""
        if not src_ids or not dst_ids:
            return
        p = float(rule["probability"])
        w = float(rule["weight"])
        # contract: excitatory -> J = -w, inhibitory -> J = +w
        coupling = -w if rule["type"] == "excitatory" else +w
        src = np.asarray(src_ids)
        dst = np.asarray(dst_ids)
        mask = self.rng.random((src.size, dst.size)) < p
        for a in range(src.size):
            row = dst[mask[a]]
            i = src[a]
            # no self-coupling even if a preset projects onto itself
            row = row[row != i]
            M[i, row] += coupling

    def _build_coupling(self):
        n = self.n
        M = np.zeros((n, n))
        for cluster in self.template["clusters"]:
            cid = cluster["id"]
            for rule in cluster.get("internalConnectivity", []):
                src = self.groups.get((cid, rule["from"]), [])
                dst = self.groups.get((cid, rule["to"]), [])
                self._add_rule(M, src, dst, rule)
        for conn in self.template.get("connections", []):
            fc, tc = conn["fromCluster"], conn["toCluster"]
            for rule in conn.get("connectivity", []):
                src = self.groups.get((fc, rule["from"]), [])
                dst = self.groups.get((tc, rule["to"]), [])
                self._add_rule(M, src, dst, rule)
        # symmetrize the directed couplings into an equilibrium energy landscape
        J = 0.5 * (M + M.T)
        np.fill_diagonal(J, 0.0)
        scale = 1.0
        if self.normalize == "field":
            row_abs = np.abs(J).sum(axis=1)
            scale = float(row_abs.mean()) or 1.0
            J = J / scale
        elif self.normalize not in (None, "none"):
            raise ValueError("unknown normalize mode: {}".format(self.normalize))
        self.coupling_scale = scale
        self.J = J

    def bias(self, b0=0.0, exc_bias=None, inh_bias=None):
        """Build a bias vector b.

        The atlas template carries no explicit I_bias, so bias is a tuning knob
        that stands in for resting excitability. `b0` is the global baseline;
        `exc_bias` / `inh_bias` optionally override per polarity.
        """
        b = np.full(self.n, float(b0))
        if exc_bias is not None:
            b[self.is_excitatory] = float(exc_bias)
        if inh_bias is not None:
            b[~self.is_excitatory] = float(inh_bias)
        return b


def coupling_from_template(template, seed=0):
    """Convenience: return (J, b0, network) for a template. b is zeros here;
    use `network.bias(...)` to set a baseline."""
    net = BrainNetwork(template, seed=seed)
    return net.J, np.zeros(net.n), net


class IsingSampler:
    """Single-site Gibbs sampler for a symmetric Ising energy landscape.

    Uses the spin conditional P(s_i = +1) = sigmoid(2 beta (b_i - (J s)_i)).
    Single-site (sequential) updates are required because the atlas graph is
    not bipartite, so the block-Gibbs sampler in `sims.boltzmann` does not
    apply here.
    """

    def __init__(self, J, b, beta=1.0, seed=0):
        self.J = np.asarray(J, dtype=float)
        self.b = np.asarray(b, dtype=float)
        self.beta = float(beta)
        self.n = self.b.size
        if self.J.shape != (self.n, self.n):
            raise ValueError("J shape {} inconsistent with b length {}".format(
                self.J.shape, self.n))
        self.rng = np.random.default_rng(seed)

    def _sweep(self, s, Js):
        b, J, beta, rng = self.b, self.J, self.beta, self.rng
        for i in range(self.n):
            v = 2.0 * beta * (b[i] - Js[i])
            p = 1.0 / (1.0 + np.exp(-v)) if v >= 0 else np.exp(v) / (1.0 + np.exp(v))
            new = 1 if rng.random() < p else -1
            if new != s[i]:
                Js += J[:, i] * (new - s[i])
                s[i] = new

    def chain(self, n_samples, burn_in=300, thin=2, s0=None):
        """Run a Gibbs chain. Returns spin samples, shape (n_samples, n) in {-1,+1}."""
        if s0 is None:
            s = self.rng.choice(np.array([-1, 1], dtype=float), size=self.n)
        else:
            s = np.asarray(s0, dtype=float).copy()
        Js = self.J @ s
        for _ in range(burn_in):
            self._sweep(s, Js)
        out = np.empty((n_samples, self.n), dtype=np.int8)
        for k in range(n_samples):
            for _ in range(thin):
                self._sweep(s, Js)
            out[k] = s.astype(np.int8)
        return out


def samples_to_rates(samples, r_max=50.0):
    """Map Ising samples in {-1,+1} to mean firing rates, r_i = (s_i+1)/2 * r_max."""
    s = np.asarray(samples, dtype=float)
    active = 0.5 * (s + 1.0)            # fraction of time at +1
    return active.mean(axis=0) * r_max


def exact_marginals(J, b, beta=1.0):
    """Brute-force P(s_i = +1) by enumeration. Only for small n (<= 20)."""
    J = np.asarray(J, dtype=float)
    b = np.asarray(b, dtype=float)
    n = b.size
    if n > 20:
        raise ValueError("refusing brute-force enumeration for n > 20")
    states = np.array(list(itertools.product([-1, 1], repeat=n)), dtype=float)
    # E(s) = sum_{i<j} J_ij s_i s_j - b.s ; with symmetric J this is
    # 0.5 * s J s (diag zero) - b.s
    quad = 0.5 * np.einsum("ki,ij,kj->k", states, J, states)
    lin = states @ b
    energy = quad - lin
    logw = -beta * energy
    logw -= logw.max()
    w = np.exp(logw)
    w /= w.sum()
    p_up = ((states == 1).astype(float) * w[:, None]).sum(axis=0)
    return p_up
