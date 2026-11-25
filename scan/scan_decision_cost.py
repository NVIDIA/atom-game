# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""What one decision costs each method, as the instance grows.

Greedy asks the cost model about every candidate site, so its work per decision
grows with the board. The network reads the estimator only through features the
environment has already computed, so its work per decision is one forward pass.
The lookahead teacher plays greedy on from each of its shortlisted sites for whole
turns, so its work per decision grows with the product of atoms and sites. Those
three scalings are the content of the scalability section, and this script is
where its table comes from.

Circuits are sparse Pauli time evolution of the family the transferable pools are
drawn from, at a fixed ratio of terms to qubits, on a grid of twice as many sites
as atoms, so that only the size of the instance changes between rows. A circuit is
cached the first time a size is asked for, because generating one grows sharply
with the qubit count and the same circuit should be used by every method.

    PYTHONPATH=. python scan/scan_decision_cost.py
    PYTHONPATH=. python scan/scan_decision_cost.py --sizes 100 200
    PYTHONPATH=. python scan/scan_decision_cost.py --sizes 1000 --teacher-decisions 1
"""

import argparse
import time
from pathlib import Path

import torch
from gymnasium.spaces import flatten
from hydra import compose, initialize_config_dir
from ml_collections import ConfigDict
from omegaconf import OmegaConf

from qds.qcgame.heuristic import GreedyPolicy, LookaheadPolicy
from qds.qcgame.meter import counting, evaluations
from qds.task.container import Container
from scan.scan_scaling import unflatten

CONF = Path(__file__).resolve().parent.parent / "conf"
CACHE = Path("outputs/scale_probe")
# The pools scale terms with qubits at a fixed ratio and leave half the board
# empty; held there so that a row differs from the one above it only in size.
TERMS_PER_QUBIT = 0.8
SITES_PER_ATOM = 2
# Wide enough that a row of the grid is not a single site at any size.
ROWS = {100: 20, 200: 20, 400: 20, 1000: 40}


def settings(atoms, circuit):
    row = ROWS.get(atoms, 20)
    with initialize_config_dir(config_dir=str(CONF), version_base="1.3"):
        config = compose(
            config_name="transfer_100",
            overrides=[
                "task=test",
                f"benchmark.min_atoms={atoms}",
                f"benchmark.max_atoms={atoms}",
                f"benchmark.n_terms={int(TERMS_PER_QUBIT * atoms)}",
                f"board.row={row}",
                f"board.col={SITES_PER_ATOM * atoms // row}",
                f"test_circuit_path={circuit}",
                "benchmark.test_count=1",
            ],
        )
    return ConfigDict(OmegaConf.to_container(config, resolve=True))


def clock(work, repeats):
    """Seconds per call, after one untimed call to leave nothing warming up."""
    work()
    start = time.perf_counter()
    for _ in range(repeats):
        work()
    return (time.perf_counter() - start) / repeats


def measure(atoms, teacher_decisions, repeats):
    circuit = CACHE / f"q{atoms}.json"
    fresh = not circuit.exists()
    CACHE.mkdir(parents=True, exist_ok=True)
    config = settings(atoms, circuit)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    config.system.device = device
    start = time.time()
    env = Container(config.mode).get_env(config)
    # Only meaningful the first time a size is asked for, when the container
    # generates the circuit instead of reading it back.
    built = time.time() - start if fresh else None
    observation, _ = env.reset()
    sites = config.board.row * config.board.col

    greedy = GreedyPolicy(config)
    with counting() as counts:
        # One decision is the whole ranking, which is what choosing already does.
        scalar = clock(lambda: greedy.scores(env), repeats)
    greedy_calls = evaluations(counts) / (repeats + 1)

    net = Container(config.mode).get_policy(config).to(device)
    space = env.observation_space
    flat = torch.as_tensor(
        flatten(space, observation), dtype=torch.float32, device=device
    ).unsqueeze(0)
    state = unflatten(space, flat)

    def forward():
        with torch.no_grad():
            net.get_action_and_value(state)
        if device == "cuda":
            torch.cuda.synchronize()

    with counting() as forward_counts:
        batched = clock(forward, 10 * repeats)
    policy_calls = evaluations(forward_counts) / (10 * repeats + 1)

    # The policy as the paper deploys it, the guided greedy: the estimator scores
    # every candidate, exactly as greedy does, to draw up the shortlist, and the
    # network chooses inside it. The mask over the shortlist is a top-k and is
    # left out of the clock along with the rest of the bookkeeping.
    def guided():
        greedy.scores(env)
        forward()

    with counting() as guided_counts:
        steered = clock(guided, repeats)
    guided_calls = evaluations(guided_counts) / (repeats + 1)

    if not teacher_decisions:
        teacher_calls = teacher_time = None
        return {
            "atoms": atoms,
            "sites": sites,
            "chunks": env.reader.n_components(),
            "built": built,
            "greedy_calls": greedy_calls,
            "greedy_time": scalar,
            "policy_calls": policy_calls,
            "policy_time": batched,
            "guided_calls": guided_calls,
            "guided_time": steered,
            "teacher_calls": teacher_calls,
            "teacher_time": teacher_time,
        }

    teacher = LookaheadPolicy(config)
    with counting() as teacher_counts:
        start = time.time()
        for _ in range(teacher_decisions):
            _, _, terminated, truncated, _ = env.step(teacher.action(env))
            if terminated or truncated:
                break
        taught = (time.time() - start) / teacher_decisions
    teacher_calls = evaluations(teacher_counts) / teacher_decisions

    return {
        "atoms": atoms,
        "sites": sites,
        "chunks": env.reader.n_components(),
        "built": built,
        "greedy_calls": greedy_calls,
        "greedy_time": scalar,
        "policy_calls": policy_calls,
        "policy_time": batched,
        "guided_calls": guided_calls,
        "guided_time": steered,
        "teacher_calls": teacher_calls,
        "teacher_time": taught,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", type=int, nargs="+", default=[100, 200, 400])
    # One decision is enough above a few hundred atoms, where the teacher takes
    # minutes over a single one; zero leaves the teacher out.
    parser.add_argument("--teacher-decisions", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()

    header = (
        f"{'atoms':>6}{'sites':>7}{'chunks':>8}"
        f"{'greedy calls':>14}{'greedy s':>11}"
        f"{'policy calls':>14}{'policy s':>11}"
        f"{'guided calls':>14}{'guided s':>11}"
        f"{'teacher calls':>15}{'teacher s':>12}"
    )
    print(header)
    for atoms in args.sizes:
        row = measure(atoms, args.teacher_decisions, args.repeats)
        print(
            f"{row['atoms']:>6}{row['sites']:>7}{row['chunks']:>8}"
            f"{row['greedy_calls']:>14.0f}{row['greedy_time']:>11.4f}"
            f"{row['policy_calls']:>14.0f}{row['policy_time']:>11.4f}"
            f"{row['guided_calls']:>14.0f}{row['guided_time']:>11.4f}"
            + (
                f"{row['teacher_calls']:>15.0f}{row['teacher_time']:>12.2f}"
                if row["teacher_calls"] is not None
                else f"{'-':>15}{'-':>12}"
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
