# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The bar a restarted heuristic sets on the same number of episodes PPO plays.

A run reports the cheapest schedule it ever produced, so it is a search, and a
search is only fairly compared against another search on an equal number of tries.
``episode_budget`` measures how many episodes a run of each instance plays; this
plays a noisy greedy that many times and keeps the cheapest.

The budget does not fit in one process, so it is split into shards. The instance
has to stay fixed across them, which is why the noise takes a seed of its own and
``seed`` keeps setting the circuit and the starting layout.

    PYTHONPATH=. python report/search_bar.py qaoa_14 groundstate_14 --seeds 0 1 2 --workers 48
"""

import argparse
import json
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from report.episode_budget import play, resolve

REPO = Path(__file__).resolve().parent.parent
PYTHON = sys.executable
CURVE_POINTS = (1, 10, 100, 1_000, 10_000)


def cache_path(name, seed):
    return REPO / "outputs" / "circuits" / f"{name}_s{seed}.pkl"


def budget(name, seed):
    """Episodes a PPO run of this instance plays, which is the number of tries.

    Measuring it also builds the circuit, which leaves the shards a copy to read.
    """
    resolved = resolve(name, seed, [f"circuit_cache={cache_path(name, seed)}"])
    decisions, _, _ = play(resolved)
    training = resolved.training
    steps = resolved.n_components * training.rollout_multiplier
    return round(training.num_iterations * training.num_envs * steps / decisions)


def shard(name, seed, repeats, policy_seed, heuristic, epsilon):
    overrides = [
        f"--config-name={name}",
        f"seed={seed}",
        "system.device=cpu",
        "logger=csv",
        f"circuit_cache={cache_path(name, seed)}",
        f"output=/tmp/search_bar/{name}_s{seed}_p{policy_seed}",
        f"+heuristic={heuristic}",
        f"+repeats={repeats}",
        f"+epsilon={epsilon}",
        f"+policy_seed={policy_seed}",
    ]
    result = subprocess.run(
        [PYTHON, "bin/benchmark_heuristic.py", *overrides],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=True,
    )
    start = re.search(r"^\{$", result.stdout, re.MULTILINE)
    return json.loads(result.stdout[start.start() :])


def interleave(shards):
    """One attempt from each shard in turn.

    Best of k has to mean k independent tries, so the k cheapest cannot all come
    from the shard that happened to be luckiest.
    """
    pooled = []
    for index in range(max(len(costs) for costs in shards)):
        pooled += [costs[index] for costs in shards if index < len(costs)]
    return pooled


def measure(name, seed, workers, heuristic, epsilon, pool, repeats=0):
    # Measured even when overridden, because measuring it builds the circuit that
    # the shards then read instead of each building its own.
    measured = budget(name, seed)
    repeats = repeats or measured
    shards = max(1, min(workers, repeats))
    per_shard = [repeats // shards + (index < repeats % shards) for index in range(shards)]
    with ThreadPoolExecutor(max_workers=shards) as executor:
        results = list(
            executor.map(
                lambda item: shard(name, seed, item[1], item[0], heuristic, epsilon),
                enumerate(per_shard),
            )
        )
    costs = interleave([result["costs"] for result in results])
    curve = {
        k: min(costs[:k]) for k in (*CURVE_POINTS, len(costs)) if k <= len(costs)
    }
    record = {
        "instance": name,
        "seed": seed,
        "heuristic": heuristic,
        "epsilon": epsilon,
        "repeats": len(costs),
        "shards": shards,
        "cost": min(costs),
        "mean_cost": sum(costs) / len(costs),
        "curve": curve,
        "runtime_seconds": max(result["runtime_seconds"] for result in results),
        # Every try in the order they pool, so a curve at any other budget needs
        # no rerun.
        "costs": costs,
    }
    # Epsilon is part of the name: on a long circuit a noisier greedy damages every
    # restart, so which level is the stronger baseline is itself a measurement.
    path = pool / f"{name}_s{seed}_{heuristic}_e{epsilon:g}.json"
    path.write_text(json.dumps(record, indent=2))
    print(
        f"{name} seed {seed}: {heuristic} best of {len(costs)} is {min(costs):.2f} "
        f"(one try {curve[1]:.2f}), {record['runtime_seconds'] / 60:.0f} min in "
        f"{shards} shards -> {path}",
        flush=True,
    )
    return record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("instances", nargs="+")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--workers", type=int, default=48)
    parser.add_argument("--heuristic", default="sampled_greedy")
    parser.add_argument("--epsilon", type=float, default=0.02)
    parser.add_argument("--out", default="outputs/search_bar")
    # Zero measures the budget from the instance's own training settings, which is
    # the point; a small number is for checking the plumbing.
    parser.add_argument("--repeats", type=int, default=0)
    args = parser.parse_args()

    pool = REPO / args.out
    pool.mkdir(parents=True, exist_ok=True)
    for name in args.instances:
        for seed in args.seeds:
            measure(
                name,
                seed,
                args.workers,
                args.heuristic,
                args.epsilon,
                pool,
                args.repeats,
            )


if __name__ == "__main__":
    main()
