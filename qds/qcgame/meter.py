# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""How many times the cost model was evaluated, which is what a search spends.

Comparing a learned policy against a search needs one currency, and the only one
they share is an evaluation of the analytic cost model: greedy scores every free
site with it, beam scores the sites of every node it expands, and a network reads
none of it beyond what the environment itself asks for. Wall clock will not do,
since it says as much about the implementation as about the method.

The estimator's entry points delegate, so counting all of them counts one
evaluation more than once: ``gate_cost`` is ``gate_cost_detail`` without the
detail. Only the leaves are counted here, and a batch counts as the number of
layouts it evaluates rather than as one call, because that is the work.
"""

from collections import Counter
from contextlib import contextmanager

from qds.qcgame.environment import BaseCircuitEnv
from qds.qcgame.estimator import CostEstimator

# Each of these evaluates the physical model once, for one layout.
LEAVES = (
    "is_parallel_executable",
    "gate_cost_detail",
    "reconfiguration_cost_detail",
)
# These evaluate it for a stack of layouts, whose height is the first argument's.
BATCHES = (
    "gate_cost_batch",
    "reconfigure_cost_batch",
)


@contextmanager
def counting():
    """Count cost model evaluations while the block runs.

    The counter is yielded and keeps its tally afterwards, so a caller may read
    it per rollout by clearing it between rollouts.
    """
    counts = Counter()
    originals = {name: getattr(CostEstimator, name) for name in LEAVES + BATCHES}
    stepper = BaseCircuitEnv.step

    def wrap(name, original, weight):
        def counted(self, *args, **kwargs):
            counts[name] += weight(args)
            return original(self, *args, **kwargs)

        return counted

    try:
        for name in LEAVES:
            setattr(CostEstimator, name, wrap(name, originals[name], lambda args: 1))
        for name in BATCHES:
            setattr(CostEstimator, name, wrap(name, originals[name], _rows))
        # Advancing the environment is its own budget, and not one every method
        # spends alike: a policy that decides per atom steps once per decision,
        # while a search that plans ahead steps copies of the environment it then
        # throws away. If the environment were a simulation rather than an
        # analytic model, these would be the expensive part.
        BaseCircuitEnv.step = wrap("step", stepper, lambda args: 1)
        yield counts
    finally:
        for name, original in originals.items():
            setattr(CostEstimator, name, original)
        BaseCircuitEnv.step = stepper


def evaluations(counts):
    """Cost model evaluations only, which is what a search spends to choose."""
    return sum(value for name, value in counts.items() if name != "step")


def steps(counts):
    """Environment transitions, including the ones a search takes on a copy."""
    return counts["step"]


def _rows(args):
    """Layouts in a batched call, whose first argument holds one per row."""
    active_sites = args[0] if args else ()
    return len(active_sites)
