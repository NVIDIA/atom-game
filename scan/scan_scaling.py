# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compare the cost of one decision for the heuristic and for the policy.

Greedy asks the cost model about every empty site, so its per-decision work grows
with the board while a policy scores all sites in one batched forward. This is
the axis on which a learned scheduler can win outright, so it needs numbers.

    PYTHONPATH=. python scan/scan_scaling.py
"""

import json
import re
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PYTHON = sys.executable

SCALES = [
    # atoms, row, col, gates per layer
    (20, 5, 5, 4),
    (20, 6, 6, 4),
    (50, 10, 10, 8),
    (100, 10, 20, 12),
]


def greedy(atoms, row, col, gates, truncate, seed):
    command = [
        PYTHON,
        "bin/benchmark_heuristic.py",
        f"benchmark.id=qnn-lite-{atoms}",
        f"board.row={row}",
        f"board.col={col}",
        "benchmark.layers=40",
        f"benchmark.gates_per_layer={gates}",
        f"benchmark.truncate={truncate}",
        f"seed={seed}",
        "output=/tmp/scan_scaling",
        "+heuristic=greedy",
    ]
    result = subprocess.run(command, cwd=REPO, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip().splitlines()[-1])
    start = re.search(r"^\{$", result.stdout, re.MULTILINE)
    return json.loads(result.stdout[start.start() :])


def policy(atoms, row, col, gates, truncate, seed, batch):
    """Time one batched forward of the policy on the same instance."""
    import torch
    from ml_collections import ConfigDict
    from omegaconf import OmegaConf

    from qds.task.container import Container

    config = OmegaConf.load(REPO / "conf" / "train.yaml")
    config = ConfigDict(OmegaConf.to_container(config, resolve=True))
    config.benchmark.id = f"qnn-lite-{atoms}"
    config.benchmark.problem = "qnn-lite"
    config.benchmark.n_atoms = atoms
    config.benchmark.layers = 40
    config.benchmark.gates_per_layer = gates
    config.benchmark.truncate = truncate
    config.board.row, config.board.col = row, col
    config.seed = seed
    config.system.device = "cuda" if torch.cuda.is_available() else "cpu"

    container = Container("fixed")
    env = container.get_env(config)
    net = container.get_policy(config).to(config.system.device)
    observation, _ = env.reset()
    space = env.observation_space
    from gymnasium.spaces import flatten

    flat = torch.as_tensor(
        flatten(space, observation), dtype=torch.float32, device=config.system.device
    )
    batched = flat.unsqueeze(0).repeat(batch, 1)
    state = unflatten(space, batched)

    with torch.no_grad():
        for _ in range(3):
            net.get_action_and_value(state)
        if config.system.device == "cuda":
            torch.cuda.synchronize()
        started_at = time.perf_counter()
        for _ in range(20):
            net.get_action_and_value(state)
        if config.system.device == "cuda":
            torch.cuda.synchronize()
        elapsed = (time.perf_counter() - started_at) / 20
    env.close()
    return elapsed


def unflatten(space, flat_tensor):
    import gymnasium as gym
    import numpy as np

    obs = {}
    index = 0
    batch_size = flat_tensor.shape[0]
    for key, subspace in space.spaces.items():
        size = int(np.prod(subspace.shape))
        shape = subspace.shape
        if isinstance(subspace, gym.spaces.Discrete):
            size = subspace.n
            shape = (subspace.n,)
        obs[key] = flat_tensor[:, index : index + size].reshape((batch_size,) + shape)
        index += size
    return obs


def main():
    truncate, seed, batch = 40, 0, 80
    print(
        f"{'instance':<16}{'sites':>7}{'steps':>7}{'greedy/step':>13}"
        f"{'policy/step':>13}{'speedup':>9}"
    )
    for atoms, row, col, gates in SCALES:
        try:
            heuristic = greedy(atoms, row, col, gates, truncate, seed)
        except RuntimeError as error:
            print(f"qnn-lite-{atoms} {row}x{col} failed: {error}")
            continue
        per_decision = heuristic["runtime_seconds"] / heuristic["steps"]
        # A rollout batches num_envs decisions into one forward, so the cost per
        # decision is the forward divided by the batch.
        forward = policy(atoms, row, col, gates, truncate, seed, batch)
        amortised = forward / batch
        print(
            f"qnn-lite-{atoms} {row}x{col:<7}{row * col:>7}{heuristic['steps']:>7}"
            f"{1000 * per_decision:>11.3f}ms{1000 * amortised:>11.3f}ms"
            f"{per_decision / amortised:>8.1f}x"
        )


if __name__ == "__main__":
    main()
