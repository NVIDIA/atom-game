# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import time

import hydra
from ml_collections import ConfigDict
from omegaconf import DictConfig, OmegaConf

from qds.qcgame.heuristic import POLICIES
from qds.task.container import Container

CURVE_POINTS = {1, 10, 100, 1_000, 10_000, 20_000}


def build_policy(policy_class, config):
    """Give the noise a seed of its own, so restarts can be split across processes.

    ``config.seed`` fixes the circuit and the starting layout, which every shard of
    a budget has to share: a shard that varied it would be measuring a different
    instance and pooling the results would be meaningless.
    """
    if policy_class.deterministic:
        return policy_class(config)
    return policy_class(config, seed=config.get("policy_seed", config.seed))


def run_heuristic(
    config: ConfigDict,
    heuristic: str,
    container: Container | None = None,
    repeats: int = 1,
):
    if heuristic not in POLICIES:
        raise ValueError(
            f"Unknown heuristic {heuristic!r}; choose from {tuple(POLICIES)}"
        )
    # In transferable mode a reference is only meaningful per circuit, so the
    # held out set is what the heuristic is measured on, one circuit per reset.
    env = (container or Container(config.mode)).get_env(config)
    method, policy_class = POLICIES[heuristic]
    policy = build_policy(policy_class, config)

    started_at = time.perf_counter()
    episodes = []
    curve = {}
    for attempt in range(repeats):
        env.reset()
        steps = 0
        while True:
            action = policy.action(env)
            _, _, terminated, truncated, _ = env.step(action)
            steps += 1
            if terminated or truncated:
                break
        episodes.append(
            {
                "steps": steps,
                "cost": env.log.cost,
                "reward": env.log.reward,
                "duration": env.log.duration,
                "touches": env.log.n_touches,
                "reconfig_count": env.log.reconfig_count,
            }
        )
        # Restarts have strongly diminishing returns, so the budget at which the
        # comparison is made changes the answer and has to be reported.
        if attempt + 1 in CURVE_POINTS or attempt + 1 == repeats:
            curve[attempt + 1] = min(episode["cost"] for episode in episodes)
        if repeats >= 1000 and (attempt + 1) % 1000 == 0:
            best_so_far = min(episode["cost"] for episode in episodes)
            print(
                f"# {attempt + 1}/{repeats} restarts, best {best_so_far:.2f}",
                flush=True,
            )
    runtime = time.perf_counter() - started_at

    best = min(episodes, key=lambda episode: episode["cost"])
    result = {
        "method": method,
        "benchmark": config.benchmark.id,
        "seed": config.seed,
        "repeats": repeats,
        "evaluations": policy.evaluations,
        "runtime_seconds": runtime,
        "mean_cost": sum(e["cost"] for e in episodes) / len(episodes),
        "best_curve": curve,
        # Every attempt, so that shards of one budget pool into an exact curve.
        "costs": [episode["cost"] for episode in episodes],
        **best,
        # A pool has no single starting layout: every reset draws another one.
        "atom_map": json.loads(config.atom_map) if config.get("atom_map") else None,
    }
    env.close()
    return result


@hydra.main(version_base="1.3", config_path="../conf", config_name="train")
def main(config: DictConfig):
    resolved = ConfigDict(OmegaConf.to_container(config, resolve=True))
    # Only a fixed benchmark names its size in its id; transferable runs draw
    # theirs from a range and carry the generator settings instead.
    if resolved.mode == "fixed":
        problem, n_atoms = resolved.benchmark.id.rsplit("-", maxsplit=1)
        resolved.benchmark.problem = problem
        resolved.benchmark.n_atoms = int(n_atoms)
    heuristic = resolved.get("heuristic", "greedy")
    repeats = resolved.get("repeats", 1)
    if heuristic == "all":
        container = Container(resolved.mode)
        result = [
            # Restarting a deterministic policy just repeats the same rollout.
            run_heuristic(
                resolved,
                name,
                container,
                1 if policy_class.deterministic else repeats,
            )
            for name, (_, policy_class) in POLICIES.items()
            if not getattr(policy_class, "planning", False)
        ]
    else:
        result = run_heuristic(resolved, heuristic, repeats=repeats)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
