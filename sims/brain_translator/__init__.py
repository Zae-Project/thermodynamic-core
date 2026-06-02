"""Atlas template -> thermodynamic-core energy landscape translator + sampler.

Phase 2 cross-pillar deliverable (brain-emulation <-> thermodynamic-core).
See translator.py for the sign/energy convention and the contract references.
"""
from sims.brain_translator.translator import (
    BrainNetwork,
    IsingSampler,
    coupling_from_template,
    exact_marginals,
    load_template,
    samples_to_rates,
)

__all__ = [
    "BrainNetwork",
    "IsingSampler",
    "coupling_from_template",
    "exact_marginals",
    "load_template",
    "samples_to_rates",
]
