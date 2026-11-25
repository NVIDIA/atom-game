# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""How many episodes a PPO run plays, so a restarted heuristic can be given as many.

A run reports the cheapest schedule it ever produced, which makes it a search, and
a search is only fairly compared against another search on the same number of
tries. The number of tries is not the iteration count: an iteration collects
``n_components`` steps in each environment, while an episode takes as many steps as
the circuit has decisions, which is several times that. So

    episodes = iterations * environments * n_components / decisions per episode

and the per episode figure has to be measured by playing one.

    PYTHONPATH=. python report/episode_budget.py qaoa_14 groundstate_14 random_30 qnn_50 qft_100 hubbard_100
"""

import sys
from pathlib import Path

from hydra import compose, initialize_config_dir
from ml_collections import ConfigDict
from omegaconf import OmegaConf

from qds.qcgame.heuristic import GreedyPolicy
from qds.task.container import Container

CONF = Path(__file__).resolve().parent.parent / "conf"


def resolve(name, seed=0, overrides=()):
    with initialize_config_dir(config_dir=str(CONF), version_base="1.3"):
        config = compose(config_name=name, overrides=[f"seed={seed}", *overrides])
    resolved = ConfigDict(OmegaConf.to_container(config, resolve=True))
    problem, atoms = resolved.benchmark.id.rsplit("-", maxsplit=1)
    resolved.benchmark.problem = problem
    resolved.benchmark.n_atoms = int(atoms)
    return resolved


def play(resolved):
    """One greedy rollout: its length in decisions and its cost model calls."""
    env = Container(resolved.mode).get_env(resolved).unwrapped
    policy = GreedyPolicy(resolved)
    env.reset()
    decisions = 0
    while True:
        action = policy.action(env)
        decisions += 1
        _, _, over, truncated, _ = env.step(action)
        if over or truncated:
            break
    return decisions, policy.evaluations, env.log.cost


def report(name):
    resolved = resolve(name)
    decisions, evaluations, cost = play(resolved)
    training = resolved.training
    steps = resolved.n_components * training.rollout_multiplier
    episodes = training.num_iterations * training.num_envs * steps / decisions
    print(
        f"{name}: {resolved.benchmark.id} on "
        f"{resolved.board.row}x{resolved.board.col} "
        f"({resolved.board.row * resolved.board.col} sites), "
        f"{resolved.n_components} components, {decisions} decisions per episode, "
        # Candidates, not cost model calls: scoring one calls the model more than
        # once. bin/benchmark_search.py meters the model itself.
        f"{evaluations} candidates scored per rollout, greedy cost {cost:.1f}\n"
        f"  {training.num_iterations} iterations x {training.num_envs} envs x "
        f"{steps} steps = {episodes:.0f} episodes",
        flush=True,
    )
    return episodes


if __name__ == "__main__":
    for name in sys.argv[1:]:
        report(name)
