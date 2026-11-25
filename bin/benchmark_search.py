# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Cost against cost model evaluations, for the searches a learned policy is measured against.

A method is only as good as the budget it was given, and the budget these methods
share is not wall clock but evaluations of the analytic cost model: greedy scores
every free site once per atom, restarting it pays that again per restart, beam pays
it for every node it expands, and a learned policy pays only what the environment
asks for while it plays. So a claim of the form "we beat greedy" has to name the
number of evaluations on both sides, and this records it: for every policy a curve
of the best cost reached against the calls spent reaching it, in one file per
instance and seed.

The configuration is the one PPO reads, so an instance is named the same way here as
there and cannot drift between the two:

    PYTHONPATH=. python bin/benchmark_search.py --config-name qaoa_14 seed=0
    PYTHONPATH=. python bin/benchmark_search.py --config-name qft_100 seed=1 search.restarts=10

A pool of circuits draws a different one on every reset, so a curve over restarts
would not be a search on one instance. This is meant for ``mode: fixed``.
"""

import json
import time
from pathlib import Path

import hydra
from ml_collections import ConfigDict
from omegaconf import DictConfig, OmegaConf

from bin.benchmark_heuristic import build_policy
from qds.qcgame.heuristic import POLICIES
from qds.qcgame.meter import counting, evaluations, steps
from qds.task.container import Container


def rollout(env, policy, counts):
    """One episode: what it cost, how long it was, and what it spent finding out."""
    env.reset()
    counts.clear()
    started_at = time.perf_counter()
    decisions = 0
    while True:
        action = policy.action(env)
        _, _, terminated, truncated, _ = env.step(action)
        decisions += 1
        if terminated or truncated:
            break
    return {
        "decisions": decisions,
        "calls": evaluations(counts),
        # A search that plans ahead advances copies of the environment and throws
        # them away, so it takes many more transitions than it makes decisions.
        # Kept apart from the calls because the two would be separate costs if the
        # environment were a simulation.
        "steps": steps(counts),
        "cost": env.log.cost,
        "seconds": time.perf_counter() - started_at,
    }


def on_ladder(index):
    """Keep 1, 2, 5, 10, 20, 50 and so on, which draws a curve on a log axis."""
    step = 1
    while step <= index:
        if index in (step, 2 * step, 5 * step):
            return True
        step *= 10
    return False


def trace_restarts(env, config, restarts, counts):
    """Best cost so far against calls so far, as a noisy greedy is restarted.

    Restarts have strongly diminishing returns, so the budget at which a
    comparison is made decides the answer and has to be part of the record.
    """
    _, policy_class = POLICIES["sampled_greedy"]
    policy = build_policy(policy_class, config)
    points, costs, spent, stepped, best = [], [], 0, 0, None
    for attempt in range(1, restarts + 1):
        result = rollout(env, policy, counts)
        spent += result["calls"]
        stepped += result["steps"]
        best = result["cost"] if best is None else min(best, result["cost"])
        costs.append(result["cost"])
        if on_ladder(attempt) or attempt == restarts:
            points.append(
                {
                    "restarts": attempt,
                    "calls": spent,
                    "steps": stepped,
                    "cost": best,
                    "mean_cost": sum(costs) / len(costs),
                    "decisions": result["decisions"],
                }
            )
            label = f"sampled_greedy x{attempt}"
            print(
                f"  {label:<22s} {spent:12d} calls {stepped:9d} steps  "
                f"best {best:10.2f}",
                flush=True,
            )
    return {"points": points, "costs": costs, "epsilon": config.epsilon}


def trace_beam(env, config, sweep, counts):
    """One point per beam setting, so widening can be priced in calls."""
    _, policy_class = POLICIES["beam"]
    points = []
    for settings in sweep:
        policy = policy_class(
            config,
            width=settings["width"],
            branch=settings["branch"],
            turns=settings["turns"],
        )
        result = rollout(env, policy, counts)
        points.append(
            {
                "width": settings["width"],
                "branch": settings["branch"],
                "turns": settings["turns"],
                **result,
            }
        )
        label = (
            f"beam w{settings['width']} b{settings['branch']} t{settings['turns']}"
        )
        print(
            f"  {label:<22s} {result['calls']:12d} calls "
            f"{result['steps']:9d} steps  cost {result['cost']:10.2f}  "
            f"in {result['seconds']:7.1f}s",
            flush=True,
        )
    return {"points": points}


def trace_single(env, config, name, counts):
    """A deterministic policy is one rollout, so it is one point."""
    _, policy_class = POLICIES[name]
    result = rollout(env, build_policy(policy_class, config), counts)
    print(
        f"  {name:<22s} {result['calls']:12d} calls "
        f"{result['steps']:9d} steps  cost {result['cost']:10.2f}  "
        f"in {result['seconds']:7.1f}s",
        flush=True,
    )
    return {"points": [result]}


def learner_budget(config, played, demonstrated):
    """The same currency on the learned side, both while training and once deployed.

    An iteration collects ``n_components`` steps in each environment rather than a
    whole episode, so the episode count is not the iteration count, and the
    demonstrations a warm start is fitted to are search rollouts whose calls the
    learner has spent as surely as if it had searched itself.
    """
    training = config.training
    steps = config.n_components * training.rollout_multiplier
    episodes = training.num_iterations * training.num_envs * steps / played["decisions"]
    demonstrations = config.search.demonstrations or config.imitation.episodes
    return {
        # What one deployed rollout costs: the environment's own accounting, with
        # no candidate ever scored.
        "calls_per_episode": played["calls"],
        "calls_per_decision": played["calls"] / played["decisions"],
        # One transition per decision and none on a copy, which is the whole of
        # what a policy that decides per atom asks of the environment.
        "steps_per_episode": played["decisions"],
        "episodes": episodes,
        "training_calls": episodes * played["calls"],
        "training_steps": episodes * played["decisions"],
        "demonstrations": demonstrations,
        "demonstration_calls": demonstrations * demonstrated["calls"],
        "total_calls": episodes * played["calls"]
        + demonstrations * demonstrated["calls"],
    }


@hydra.main(version_base="1.3", config_path="../conf", config_name="train")
def main(config: DictConfig):
    resolved = ConfigDict(OmegaConf.to_container(config, resolve=True))
    if resolved.mode == "fixed":
        problem, n_atoms = resolved.benchmark.id.rsplit("-", maxsplit=1)
        resolved.benchmark.problem = problem
        resolved.benchmark.n_atoms = int(n_atoms)
    settings = resolved.search
    # The noise of a restart reads this, and reading it from the search block keeps
    # the epsilon of a traced curve next to the number of restarts it was traced at.
    resolved.epsilon = settings.epsilon

    env = Container(resolved.mode).get_env(resolved)
    shape = f"{resolved.board.row}x{resolved.board.col}"
    print(
        f"{resolved.benchmark.id} on {shape} "
        f"({resolved.board.row * resolved.board.col} sites), "
        f"{resolved.n_components} components, seed {resolved.seed}",
        flush=True,
    )

    with counting() as counts:
        traces = {}
        for name in settings.policies:
            if name == "beam":
                traces[name] = trace_beam(env, resolved, settings.sweep, counts)
            elif name == "sampled_greedy":
                traces[name] = trace_restarts(
                    env, resolved, settings.restarts, counts
                )
            else:
                traces[name] = trace_single(env, resolved, name, counts)
        # A learned policy never scores a candidate, so a policy that scores none
        # measures what its rollout costs; greedy prices the demonstrations.
        played = traces.get("no_reconfiguration") or {
            "points": [
                trace_single(env, resolved, "no_reconfiguration", counts)["points"][0]
            ]
        }
        demonstrated = traces.get("greedy") or {
            "points": [trace_single(env, resolved, "greedy", counts)["points"][0]]
        }
    env.close()

    record = {
        "benchmark": resolved.benchmark.id,
        "board": shape,
        "sites": resolved.board.row * resolved.board.col,
        "seed": resolved.seed,
        "components": resolved.n_components,
        "policies": traces,
        "learner": learner_budget(
            resolved, played["points"][0], demonstrated["points"][0]
        ),
    }
    path = Path(settings.out) / f"{resolved.benchmark.id}_{shape}_s{resolved.seed}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2))
    print(f"wrote {path}", flush=True)


if __name__ == "__main__":
    main()
