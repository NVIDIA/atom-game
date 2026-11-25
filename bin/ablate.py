# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run and collect small-instance ablations for the fixed-circuit PPO stack.

Each arm is a single-change delta from BASELINE so that results stay
interpretable. Arms run one-per-GPU in parallel.

    PYTHONPATH=. python bin/ablate.py run merge_move --seeds 0 1 2
    PYTHONPATH=. python bin/ablate.py collect merge_move --seeds 0 1 2

``--arms`` runs a subset while the group still decides what collect compares, so
new arms can be added to an existing comparison without rerunning it:

    PYTHONPATH=. python bin/ablate.py run geometry --arms geometry_add geometry_only
    PYTHONPATH=. python bin/ablate.py collect geometry
"""

import argparse
import csv
import itertools
import json
import math
import os
import re
import subprocess
import sys
import time
from pathlib import Path

from report import search_bar
from report.episode_budget import resolve

REPO = Path(__file__).resolve().parent.parent
PYTHON = sys.executable
TORCHRUN = str(Path(PYTHON).with_name("torchrun"))
PROJECT = "neutral-atom-final-v5"

# The paper's instances, one config each in conf/. The grid and the training
# budget live there rather than in override lists here, so that this harness cannot
# drift from what the paper reports.
INSTANCES = [
    "qaoa_14",
    "groundstate_14",
    "random_30",
    "qnn_50",
    "qft_100",
    "hubbard_100",
]

# What the harness needs on top of an instance's own config: metrics on disk for
# collect to read back, and no checkpoint, so that one arm never resumes another's.
SETTINGS = ["logger=csv", "load_checkpoint=false"]


def use_instance(name):
    """Point the harness, the run names and the artefact paths at one instance."""
    global INSTANCE, CONFIG, BOARD, SHAPE, BASELINE
    resolved = resolve(name)
    INSTANCE = name
    CONFIG = f"--config-name={name}"
    BOARD = f"{resolved.board.row}x{resolved.board.col}"
    # The board is part of the name so that a change of instance never reads back
    # results measured on another one; two benchmarks can share a shape.
    SHAPE = f"{resolved.benchmark.id}_{BOARD}"
    BASELINE = SETTINGS


use_instance("qaoa_14")

IMITATION_WEIGHTS = "init_model=outputs/imitation/{shape}_s{seed}.pt"
# Labels from a quieter expert: on this board a destination drawn at random is a
# long move, so epsilon 0.05 rolls in through states worth little.
SHARP_WEIGHTS = "init_model=outputs/imitation/{shape}_e02_s{seed}.pt"

# Warm start plus the exploration level and the imitation anchor that go with it:
# on its own the warm start decays to a policy that no longer proposes anything new.
SEARCH_KNOBS = [
    "training.target_entropy=0.06",
    "training.sil_coef=0.1",
    "training.sil_seed_episodes=8",
]
BC_INIT = [IMITATION_WEIGHTS] + SEARCH_KNOBS

# Weights whose critic was warm started too. Handed a trained actor and an
# untrained critic, PPO's explained variance starts at -1.9 and only turns useful
# around iteration 20, by which time the actor's entropy is spent.
WARM_WEIGHTS = "init_model=outputs/imitation/{shape}_v2_s{seed}.pt"

# Weights fitted to the expert's whole ranking of candidate costs. One label out
# of two hundred sites is too thin a target on the large boards, where the expert
# leaves the atom alone four times out of five and imitation of its choice alone
# collapses onto exactly the no-reconfiguration schedule.
RANKED_WEIGHTS = "init_model=outputs/imitation/{shape}_v3_s{seed}.pt"
SHARP_RANKED = "init_model=outputs/imitation/{shape}_v4_s{seed}.pt"

# Only the last atom of a reconfiguration earns a reward, so a turn is treated as
# one decision. Each iteration then covers four times as many steps, and the
# iteration count drops to match so that the experience and wall clock stay equal.
TURN = [
    "training.turn_actions=true",
    "training.rollout_multiplier=4",
    "training.num_iterations=500",
]

ARMS = {
    "control": [],
    "merge_move_off": ["policy.merge_move=false"],
    "vf_coef_low": ["training.vf_coef=0.1"],
    "grad_norm_high": ["training.max_grad_norm=5.0"],
    "no_clip_vloss": ["training.clip_vloss=false"],
    "orthogonal_init": ["policy.orthogonal_init=true"],
    "ent_high": ["training.ent_coef=0.05"],
    "lr_high": ["training.learning_rate=1e-3"],
    "minibatch_many": ["training.num_minibatches=10"],
    "pool_attention": ["policy.pool=attention"],
    "rollout_long": ["training.rollout_multiplier=4"],
    # Exploration: control collapses below entropy 0.1 around iteration 85 and
    # then stops improving, so these arms attack the collapse itself.
    "ent_adaptive_050": ["training.target_entropy=0.5"],
    "ent_adaptive_075": ["training.target_entropy=0.75"],
    # Sampled greedy wins at epsilon 0.02, which is about 0.1 nats per step, so
    # the useful exploration band is far below a fraction of the uniform policy.
    "ent_low_003": ["training.target_entropy=0.03"],
    "ent_low_006": ["training.target_entropy=0.06"],
    "ent_low_012": ["training.target_entropy=0.12"],
    "lr_floor": ["training.lr_floor=0.1"],
    # Representation: hand the cost model's own quantities to the embedding, and
    # let the lookahead tasks interact instead of being encoded independently.
    # Search: the metric is the best episode found, so keep a small amount of
    # noise and train the policy to reproduce its own best rollouts.
    "sil_010": ["training.sil_coef=0.1"],
    "sil_030": ["training.sil_coef=0.3"],
    "sil_ent_low": ["training.sil_coef=0.1", "training.target_entropy=0.06"],
    # Seed the imitation buffer with epsilon-greedy schedules, so the search
    # starts from the baseline's quality and can only replace it by beating it.
    "seed_only": ["training.sil_coef=0.1", "training.sil_seed_episodes=8"],
    "sil_seeded": [
        "training.sil_coef=0.1",
        "training.sil_seed_episodes=8",
        "training.target_entropy=0.06",
    ],
    # Imitation reaches the heuristic's quality in a single rollout, so starting
    # there leaves training with only the part it has never managed: improving.
    "bc_plain": [IMITATION_WEIGHTS],
    "bc_init": BC_INIT,
    "bc_plain_v2": [WARM_WEIGHTS],
    "bc_init_v2": [WARM_WEIGHTS] + SEARCH_KNOBS,
    # The warm start hands over an entropy of 0.16 and PPO spends it within thirty
    # iterations, faster than a controller on the linear error can answer.
    "bc_init_v3": [WARM_WEIGHTS] + SEARCH_KNOBS + ["training.ent_control=log"],
    # The same recipe on a board where imitation had to learn the expert's whole
    # cost ranking to represent it at all, and where geometry is therefore part
    # of the weights being loaded rather than an arm of its own.
    # The anchor episodes are drawn at a tenth of the small board's noise: with
    # 819 decisions in an episode, epsilon 0.02 perturbs forty of them and lands
    # above plain greedy, which is the level the buffer is meant to hold.
    "bc_big": [RANKED_WEIGHTS, "policy.geometry=add"]
    + SEARCH_KNOBS
    + ["training.ent_control=log", "training.sil_seed_epsilon=0.002"],
    # The same weights annealed to a confident policy by the last epoch. Fitting
    # the ranking leaves an entropy of 4.5 nats over two hundred sites, so the
    # episodes PPO samples from it carry no more information than random ones.
    "bc_big_sharp": [SHARP_RANKED, "policy.geometry=add"]
    + SEARCH_KNOBS
    + ["training.ent_control=log", "training.sil_seed_epsilon=0.002"],
    # The anchors without the weights. The warm started arms hold near greedy
    # episodes in the imitation buffer as well as starting from imitated weights,
    # and on qft-100 they win by 8% while training from scratch loses to greedy,
    # so which of the two carries it has to be separated.
    "seed_only_big": ["policy.geometry=add"]
    + SEARCH_KNOBS
    + ["training.ent_control=log", "training.sil_seed_epsilon=0.002"],
    # What the benchmark measured before: the same budget without a warm start,
    # which on a board this size never leaves the quality of a random schedule.
    "scratch_big": ["policy.geometry=add"],
    "geometry_add": ["policy.geometry=add"],
    "geometry_only": ["policy.geometry=only"],
    "cross_task": ["policy.cross_task_layers=1"],
    "pool_weighted": ["policy.pool=weighted"],
    "bc_sharp": [
        SHARP_WEIGHTS,
        "training.target_entropy=0.06",
        "training.sil_coef=0.1",
        "training.sil_seed_episodes=8",
    ],
    "turn_actions": TURN,
    "turn_bc": TURN
    + [
        IMITATION_WEIGHTS,
        "training.target_entropy=0.06",
        "training.sil_coef=0.1",
        "training.sil_seed_episodes=8",
    ],
}

GROUPS = {
    "merge_move": ["control", "merge_move_off"],
    "optimization": [
        "control",
        "vf_coef_low",
        "grad_norm_high",
        "no_clip_vloss",
        "lr_high",
    ],
    "model": ["control", "merge_move_off", "orthogonal_init", "pool_attention"],
    "credit": ["control", "rollout_long", "ent_high", "minibatch_many"],
    "explore": [
        "control",
        "ent_high",
        "ent_adaptive_050",
        "ent_adaptive_075",
        "lr_floor",
    ],
    "search": [
        "control",
        "ent_low_003",
        "ent_low_006",
        "ent_low_012",
        "sil_010",
        "sil_030",
        "sil_ent_low",
        "seed_only",
        "sil_seeded",
    ],
    "warm": ["control", "bc_plain", "bc_init", "sil_seeded"],
    "turn": ["control", "bc_init", "turn_actions", "turn_bc"],
    "sharp": ["bc_init", "bc_plain", "bc_sharp"],
    "warmstart": ["bc_plain", "bc_plain_v2", "bc_init", "bc_init_v2", "bc_init_v3"],
    "big": ["bc_big", "bc_big_sharp", "scratch_big", "seed_only_big"],
    "geometry": ["control", "geometry_add", "geometry_only"],
    "aggregation": ["control", "pool_attention", "pool_weighted", "cross_task"],
    "all": list(ARMS),
}


def tag(arm, seed):
    # The instance is part of the name so that changing it cannot read back runs
    # trained on another one; two benchmarks can share a board shape.
    return f"abl_{SHAPE}_{arm}_s{seed}"


def launch(arm, seed, gpu, port):
    # Arms may point at per-seed artefacts, such as imitation weights, and may
    # also restate a baseline key, in which case the arm's value is the one used.
    settings = {}
    for override in BASELINE + ARMS[arm]:
        key, value = override.format(seed=seed, shape=SHAPE).split("=", 1)
        settings[key] = value
    settings["seed"] = str(seed)
    settings["exp_tag"] = tag(arm, seed)
    # One copy of the circuit for every process that shares the instance: building
    # it pulls a shared dataset file through a lock that concurrent builds fail on.
    settings["circuit_cache"] = str(search_bar.cache_path(INSTANCE, seed))
    overrides = [f"{key}={value}" for key, value in settings.items()]
    command = [
        TORCHRUN,
        "--nnodes=1",
        "--nproc_per_node=1",
        "--master_addr=127.0.0.1",
        f"--master_port={port}",
        "bin/train.py",
        CONFIG,
        *overrides,
    ]
    log_path = REPO / "outputs" / "ablation_logs" / f"{tag(arm, seed)}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    handle = log_path.open("w")
    process = subprocess.Popen(
        command,
        cwd=REPO,
        stdout=handle,
        stderr=subprocess.STDOUT,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu)},
    )
    return process, handle, log_path


def run(arms, seeds, gpus, stagger):
    pending = list(itertools.product(arms, seeds))
    print(f"{len(pending)} runs over {len(gpus)} GPUs", flush=True)
    free_gpus = list(gpus)
    active = []
    # Deriving the rendezvous port from the GPU keeps two harnesses out of each
    # other's way, since a GPU only ever holds one run.
    launches = {gpu: 0 for gpu in gpus}
    while pending or active:
        while pending and free_gpus:
            arm, seed = pending.pop(0)
            gpu = free_gpus.pop(0)
            port = 29600 + gpu * 50 + launches[gpu]
            launches[gpu] += 1
            process, handle, log_path = launch(arm, seed, gpu, port)
            active.append((arm, seed, gpu, process, handle, log_path))
            print(f"start {tag(arm, seed)} on GPU {gpu}", flush=True)
            # The MQT dataset is opened read-write, so two runs building their
            # circuit at the same moment fail to take the file lock.
            time.sleep(stagger)
        time.sleep(5)
        for entry in [e for e in active if e[3].poll() is not None]:
            arm, seed, gpu, process, handle, log_path = entry
            handle.close()
            status = "ok" if process.returncode == 0 else f"FAIL {process.returncode}"
            print(f"done  {tag(arm, seed)}: {status} ({log_path})", flush=True)
            active.remove(entry)
            free_gpus.append(gpu)
    print("all runs finished", flush=True)


def greedy_cost(seed):
    overrides = [
        CONFIG,
        f"seed={seed}",
        "output=/tmp/abl_greedy",
        "+heuristic=all",
        f"circuit_cache={search_bar.cache_path(INSTANCE, seed)}",
    ]
    result = subprocess.run(
        [PYTHON, "bin/benchmark_heuristic.py", *overrides],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=True,
    )
    # Qiskit logs lines that start with a bracket, so anchor on the array itself.
    start = re.search(r"^\[$", result.stdout, re.MULTILINE)
    payload = json.loads(result.stdout[start.start() :])
    return {entry["method"]: entry["cost"] for entry in payload}


BAR_POOL = REPO / "outputs" / "search_bar"


def load_bar(seeds):
    """The strongest restarted heuristic measured for each seed.

    Which level of noise restarts best depends on how many decisions an episode
    makes, so every level that has been measured competes and the cheapest is the
    bar an arm has to clear.
    """
    bar = {}
    for seed in seeds:
        records = [
            json.loads(path.read_text())
            for path in BAR_POOL.glob(f"{INSTANCE}_s{seed}_*.json")
        ]
        if records:
            bar[str(seed)] = min(records, key=lambda record: record["cost"])
    return bar


def measure_bar(seeds, workers, epsilon, heuristic="sampled_greedy"):
    """Best cost a restarted heuristic reaches on the episodes PPO is given.

    A single greedy rollout is not the bar: a run reports the cheapest of every
    episode it played, so the heuristic gets that many tries. ``search_bar``
    measures how many that is and splits them across processes.
    """
    BAR_POOL.mkdir(parents=True, exist_ok=True)
    for seed in seeds:
        search_bar.measure(INSTANCE, seed, workers, heuristic, epsilon, BAR_POOL)
    return load_bar(seeds)


def beam_path(seed):
    return REPO / "outputs" / "ablation_logs" / f"beam_{SHAPE}_s{seed}.json"


def load_beam(seeds):
    return {
        str(seed): json.loads(beam_path(seed).read_text())
        for seed in seeds
        if beam_path(seed).exists()
    }


def measure_beam(seeds, width, branch, turns):
    """Cost of planning each reconfiguration instead of each atom.

    Greedy and its restarts are both one atom at a time, so neither says what a
    classical search that respects the structure of the cost can do. This one
    does, and on the small instance a two turn lookahead beats every learned run
    in twenty four seconds, which is the comparison that has to be reported.
    """
    beam = load_beam(seeds)
    for seed in seeds:
        overrides = [
            CONFIG,
            f"seed={seed}",
            f"output=/tmp/abl_beam_{seed}",
            "+heuristic=beam",
            f"circuit_cache={search_bar.cache_path(INSTANCE, seed)}",
            f"beam.width={width}",
            f"beam.branch={branch}",
            f"beam.turns={turns}",
        ]
        result = subprocess.run(
            [PYTHON, "bin/benchmark_heuristic.py", *overrides],
            cwd=REPO,
            capture_output=True,
            text=True,
            check=True,
        )
        start = re.search(r"^\{$", result.stdout, re.MULTILINE)
        payload = json.loads(result.stdout[start.start() :])
        beam[str(seed)] = {
            "cost": payload["cost"],
            "width": width,
            "branch": branch,
            "turns": turns,
            "evaluations": payload["evaluations"],
            "runtime_seconds": payload["runtime_seconds"],
        }
        print(
            f"  seed {seed}: {payload['cost']:.2f} at width {width} turns {turns} "
            f"({payload['runtime_seconds'] / 60:.1f} min)",
            flush=True,
        )
        path = beam_path(seed)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(beam[str(seed)], indent=2))
    return beam


def metrics_path(arm, seed):
    root = REPO / "outputs" / "logs" / PROJECT
    candidates = sorted(root.glob(f"{PROJECT}_{tag(arm, seed)}_run_*/metrics.csv"))
    return candidates[-1] if candidates else None


COLLAPSE_ENTROPY = 0.1


def summarise_run(path):
    """Final values plus how long the run kept searching.

    The control arm reaches its best cost while entropy is still non-zero and
    never improves afterwards, so when exploration dies and whether the best
    cost still moves after that matter more than any end-of-run scalar.
    """
    rows = list(csv.DictReader(path.open()))
    summary = {"iterations": len(rows)}
    for key in (
        "min cost",
        "reconfig count",
        "explained_variance",
        "approx_kl",
        "entropy",
        "grad_norm",
        "v_loss",
        "pg_loss",
    ):
        values = [r[key] for r in rows if r.get(key) not in (None, "")]
        summary[key] = float(values[-1]) if values else math.nan

    collapse_iteration = math.nan
    cost_at_collapse = math.nan
    improvements_after = 0
    best_after_collapse = math.inf
    best = math.inf
    last_improvement = math.nan
    for row in rows:
        if row.get("iteration") in (None, ""):
            continue
        iteration = float(row["iteration"])
        entropy = float(row["entropy"]) if row.get("entropy") else math.nan
        cost = float(row["min cost"]) if row.get("min cost") else math.inf
        if math.isfinite(cost) and cost < best - 1e-9:
            best = cost
            last_improvement = iteration
        if math.isnan(collapse_iteration):
            if not math.isnan(entropy) and entropy < COLLAPSE_ENTROPY:
                collapse_iteration = iteration
                cost_at_collapse = cost
                best_after_collapse = cost
        elif cost < best_after_collapse - 1e-9:
            improvements_after += 1
            best_after_collapse = cost
    summary["collapse_iteration"] = collapse_iteration
    summary["cost_at_collapse"] = cost_at_collapse
    summary["improvements_after_collapse"] = improvements_after
    summary["last_improvement"] = last_improvement
    return summary


def collect(arms, seeds):
    baselines = {seed: greedy_cost(seed) for seed in seeds}
    bar = load_bar(seeds)
    beam = load_beam(seeds)
    print("\nheuristic reference")
    for seed, costs in baselines.items():
        line = "  ".join(f"{k}={v:.2f}" for k, v in costs.items())
        if str(seed) in bar:
            entry = bar[str(seed)]
            line += f"  restarted={entry['cost']:.2f} (n={entry['repeats']})"
        if str(seed) in beam:
            entry = beam[str(seed)]
            line += f"  beam={entry['cost']:.2f} (w{entry['width']}t{entry['turns']})"
        print(f"  seed {seed}: {line}")

    print(
        f"\n{'arm':<18}{'seed':>5}{'iters':>7}{'PPO cost':>11}{'vs greedy':>10}"
        f"{'vs best':>11}{'last drop@':>12}{'better after':>13}{'entropy':>9}"
    )
    gaps_by_arm = {}
    collapse_by_arm = {}
    improvements_by_arm = {}
    last_drop_by_arm = {}
    search_gaps_by_arm = {}
    for arm in arms:
        for seed in seeds:
            path = metrics_path(arm, seed)
            if path is None:
                print(f"{arm:<18}{seed:>5}   (no metrics)")
                continue
            run = summarise_run(path)
            greedy = baselines[seed]["greedy_one_step"]
            gap = 100.0 * (run["min cost"] - greedy) / greedy
            gaps_by_arm.setdefault(arm, []).append(gap)
            collapse = run["collapse_iteration"]
            if not math.isnan(collapse):
                collapse_by_arm.setdefault(arm, []).append(collapse)
            improvements_by_arm.setdefault(arm, []).append(
                run["improvements_after_collapse"]
            )
            last_drop = run["last_improvement"]
            if not math.isnan(last_drop):
                last_drop_by_arm.setdefault(arm, []).append(last_drop)
            last_drop_text = "-" if math.isnan(last_drop) else f"{last_drop:.0f}"
            # The claim is against the best baseline that exists, which is the
            # restarted heuristic on short circuits and the beam elsewhere.
            targets = [
                entry[str(seed)]["cost"]
                for entry in (bar, beam)
                if str(seed) in entry
            ]
            if targets:
                target = min(targets)
                search_text = f"{100.0 * (run['min cost'] - target) / target:>9.2f}%"
                search_gaps_by_arm.setdefault(arm, []).append(
                    100.0 * (run["min cost"] - target) / target
                )
            else:
                search_text = f"{'-':>10}"
            print(
                f"{arm:<18}{seed:>5}{run['iterations']:>7}{run['min cost']:>11.2f}"
                f"{gap:>9.2f}%{search_text:>11}{last_drop_text:>12}"
                f"{run['improvements_after_collapse']:>13}{run['entropy']:>9.3f}"
            )

    print("\nper arm (gap vs greedy: lower is better)")
    for arm, gaps in gaps_by_arm.items():
        mean = sum(gaps) / len(gaps)
        spread = max(gaps) - min(gaps)
        collapses = collapse_by_arm.get(arm, [])
        collapse_text = (
            "never"
            if not collapses
            else f"{sum(collapses) / len(collapses):.0f}"
            + ("" if len(collapses) == len(gaps) else f" ({len(collapses)}/{len(gaps)})")
        )
        drops = last_drop_by_arm.get(arm, [])
        drop_text = "-" if not drops else f"{sum(drops) / len(drops):.0f}"
        search = search_gaps_by_arm.get(arm, [])
        search_text = (
            "-" if not search else f"{sum(search) / len(search):+.2f}%"
        )
        print(
            f"  {arm:<18}vs greedy {mean:>7.2f}%  vs best {search_text:<9}"
            f"spread {spread:>5.2f}pt  collapse@{collapse_text:<10}"
            f"last drop@{drop_text:<7} n={len(gaps)}"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["run", "collect", "bar", "beam"])
    parser.add_argument("group", choices=list(GROUPS))
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--gpus", type=int, nargs="+", default=list(range(8)))
    # The bar's number of tries is measured from the instance, not chosen here.
    parser.add_argument("--workers", type=int, default=48)
    parser.add_argument("--epsilon", type=float, default=0.02)
    parser.add_argument("--instance", choices=INSTANCES, default="qaoa_14")
    parser.add_argument("--stagger", type=float, default=20.0)
    # Depth buys more than width: at width 4 a two turn lookahead reaches 235.5
    # on the small instance where width 16 at one turn reaches 238.6.
    parser.add_argument("--width", type=int, default=4)
    parser.add_argument("--branch", type=int, default=4)
    parser.add_argument("--turns", type=int, default=2)
    parser.add_argument(
        "--arms",
        nargs="+",
        choices=list(ARMS),
        help="run only these arms; the group still decides what collect compares",
    )
    args = parser.parse_args()

    use_instance(args.instance)
    arms = args.arms or GROUPS[args.group]
    if args.action == "run":
        run(arms, args.seeds, args.gpus, args.stagger)
    elif args.action == "bar":
        print(f"restarted heuristic, epsilon {args.epsilon}")
        measure_bar(args.seeds, args.workers, args.epsilon)
    elif args.action == "beam":
        print(f"beam width {args.width} branch {args.branch} turns {args.turns}")
        measure_beam(args.seeds, args.width, args.branch, args.turns)
    else:
        collect(arms, args.seeds)


if __name__ == "__main__":
    main()
