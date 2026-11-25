# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Rank candidate small instances by how well they separate policies.

The 5x5 / 20-atom QNN-lite instance puts random, stay-in-place and greedy within
about 2% of each other, so no learning change can show up above the seed noise.
This runs the heuristics on candidate instances and reports the spread, which is
the headroom any ablation on that instance could possibly measure.

    PYTHONPATH=. python scan/scan_instances.py --seeds 0 1 2
"""

import argparse
import json
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PYTHON = sys.executable

# The sizes the project already runs, from experiments/benchmark.sh: every board has twice as
# many sites as atoms, which is also what the small instance was moved to.
CANDIDATES = {
    "qaoa-14 10x4": ["benchmark.id=qaoa-14", "board.row=10", "board.col=4"],
    "groundstate-14 10x4": [
        "benchmark.id=groundstate-14",
        "board.row=10",
        "board.col=4",
    ],
    "random-30 10x6": ["benchmark.id=random-30", "board.row=10", "board.col=6"],
    "qnn-50 20x5": ["benchmark.id=qnn-50", "board.row=20", "board.col=5"],
    "qft-100 20x10": ["benchmark.id=qft-100", "board.row=20", "board.col=10"],
    "hubbard-100 20x10": ["benchmark.id=hubbard-100", "board.row=20", "board.col=10"],
    "qnn-lite-100 20x10": [
        "benchmark.id=qnn-lite-100",
        "board.row=20",
        "board.col=10",
    ],
    "qnn-lite-20 5x8": [
        "benchmark.id=qnn-lite-20",
        "benchmark.layers=40",
        "benchmark.gates_per_layer=4",
        "board.row=5",
        "board.col=8",
    ],
}


def measure(overrides, seed, truncate, repeats, epsilon):
    command = [
        PYTHON,
        "bin/benchmark_heuristic.py",
        *overrides,
        f"seed={seed}",
        f"benchmark.truncate={truncate}",
        # One directory per job, since jobs run in parallel.
        f"output=/tmp/scan_instances_{abs(hash((tuple(overrides), seed)))}",
        "+heuristic=all",
        f"+repeats={repeats}",
        f"+epsilon={epsilon}",
    ]
    result = subprocess.run(command, cwd=REPO, capture_output=True, text=True)
    if result.returncode != 0:
        # Hydra's last line is only a hint to set a variable, so keep the tail.
        raise RuntimeError("\n".join(result.stderr.strip().splitlines()[-8:]))
    return {entry["method"]: entry for entry in parse_payload(result.stdout)}


def parse_payload(stdout):
    """Qiskit logs lines that start with a bracket, so anchor on the array itself."""
    start = re.search(r"^\[$", stdout, re.MULTILINE)
    if start is None:
        raise RuntimeError("no JSON array in heuristic output")
    return json.loads(stdout[start.start() :])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--truncate", type=int, default=0)
    parser.add_argument("--only", nargs="+", choices=list(CANDIDATES))
    parser.add_argument("--repeats", type=int, default=200)
    parser.add_argument("--epsilon", type=float, default=0.02)
    # The MQT dataset is opened read-write, so two processes cannot hold it at
    # once; only the generated families (qnn-lite, hubbard) tolerate workers > 1.
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()

    names = args.only or list(CANDIDATES)
    jobs = {}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for name in names:
            for seed in args.seeds:
                jobs[(name, seed)] = pool.submit(
                    measure,
                    CANDIDATES[name],
                    seed,
                    args.truncate,
                    args.repeats,
                    args.epsilon,
                )

    print(
        f"\n{'instance':<20}{'steps':>7}{'stay':>9}{'greedy':>9}{'search':>9}"
        f"{'below greedy':>14}{'total room':>12}{'seed spread':>13}"
    )
    for name in names:
        try:
            runs = [jobs[(name, seed)].result() for seed in args.seeds]
        except RuntimeError as error:
            print(f"{name:<20} failed: {error}")
            continue
        greedy = [r["greedy_one_step"]["cost"] for r in runs]
        search = [r["sampled_greedy"]["cost"] for r in runs]
        stay = [r["no_reconfiguration"]["cost"] for r in runs]
        mean = lambda values: sum(values) / len(values)
        # What the learner has to work with: how far below greedy a cheap search
        # already reaches, and how far below stay-in-place the best known is.
        below = 100.0 * (mean(greedy) - mean(search)) / mean(greedy)
        total = 100.0 * (mean(stay) - mean(search)) / mean(stay)
        spread = 100.0 * (max(greedy) - min(greedy)) / mean(greedy)
        print(
            f"{name:<20}{runs[0]['greedy_one_step']['steps']:>7}{mean(stay):>9.2f}"
            f"{mean(greedy):>9.2f}{mean(search):>9.2f}"
            f"{below:>13.2f}%{total:>11.2f}%{spread:>12.2f}%"
        )


if __name__ == "__main__":
    main()
