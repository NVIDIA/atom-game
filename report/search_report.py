# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""What each search bought for what it spent, pooled over seeds.

Reads the records bin/benchmark_search.py wrote and reports one frontier per instance:
the cheapest schedule any search reached at a given number of cost model
evaluations. The restarted greedy runs in outputs/search_bar are folded in, since
they already traced budgets far deeper than is worth repeating; their curves are
indexed by restarts, which converts to calls by multiplying by what one restart
spends, and that is measured here.

The last column is the comparison a learned policy has to be held to: what the best
search reaches when it is allowed the calls that training the policy cost.

    PYTHONPATH=. python report/search_report.py
    PYTHONPATH=. python report/search_report.py qaoa-14 hubbard-100
"""

import json
import statistics
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
BUDGET = REPO / "outputs" / "search_budget"
BAR = REPO / "outputs" / "search_bar"


def load(names):
    """Records grouped by benchmark, one entry per seed."""
    grouped = {}
    for path in sorted(BUDGET.glob("*_s*.json")):
        record = json.loads(path.read_text())
        if names and record["benchmark"] not in names:
            continue
        grouped.setdefault(record["benchmark"], []).append(record)
    return grouped


def bar_points(record):
    """The deep restart curve, priced in calls rather than in restarts."""
    restarts = record["policies"].get("sampled_greedy", {}).get("points")
    if not restarts:
        return []
    per_restart = restarts[0]["calls"] / restarts[0]["restarts"]
    name = record["benchmark"].replace("-", "_")
    matches = sorted(BAR.glob(f"{name}_s{record['seed']}_sampled_greedy_e*.json"))
    points = []
    for path in matches:
        curve = json.loads(path.read_text()).get("curve", {})
        for tries, cost in curve.items():
            points.append(
                {
                    # Named apart from the restarts traced here, because these were
                    # run at their own epsilon and split across shards: pooling the
                    # two under one label would average different experiments.
                    "label": f"restarted_greedy x{tries}",
                    "calls": int(tries) * per_restart,
                    "cost": cost,
                }
            )
    return points


def points(record):
    """Every measured point of one seed, as label, calls and cost."""
    collected = []
    for name, trace in record["policies"].items():
        for point in trace["points"]:
            if name == "beam":
                label = f"beam w{point['width']} b{point['branch']} t{point['turns']}"
            elif name == "sampled_greedy":
                label = f"sampled_greedy x{point['restarts']}"
            else:
                label = name
            collected.append(
                {
                    "label": label,
                    "calls": point["calls"],
                    "steps": point.get("steps", 0),
                    "cost": point["cost"],
                }
            )
    return collected + bar_points(record)


def pooled(records):
    """Mean over seeds of every point the seeds have in common."""
    per_label = {}
    for record in records:
        for point in points(record):
            per_label.setdefault(point["label"], []).append(point)
    rows = []
    for label, group in per_label.items():
        rows.append(
            {
                "label": label,
                "seeds": len(group),
                "calls": statistics.mean(p["calls"] for p in group),
                "steps": statistics.mean(p.get("steps", 0) for p in group),
                "cost": statistics.mean(p["cost"] for p in group),
            }
        )
    return sorted(rows, key=lambda row: row["calls"])


def pooled_learner(records):
    """The learned side's budget, averaged over seeds as the searches are."""
    keys = records[0]["learner"]
    return {
        key: statistics.mean(record["learner"][key] for record in records)
        for key in keys
    }


def best_within(rows, budget):
    """The cheapest schedule any search reached without exceeding a budget."""
    affordable = [row for row in rows if row["calls"] <= budget]
    return min(affordable, key=lambda row: row["cost"]) if affordable else None


def report(names=()):
    summary = {}
    for benchmark, records in load(names).items():
        rows = pooled(records)
        learner = pooled_learner(records)
        print(f"\n{benchmark} on {records[0]['board']}, {len(records)} seeds")
        print(
            f"  {'method':<24s} {'calls':>14s} {'steps':>12s} {'cost':>12s}  seeds"
        )
        for row in rows:
            print(
                f"  {row['label']:<24s} {row['calls']:14,.0f} "
                f"{row['steps']:12,.0f} {row['cost']:12.2f}  {row['seeds']}"
            )
        matched = best_within(rows, learner["total_calls"])
        print(
            f"  a learned policy of this config spends "
            f"{learner['total_calls']:,.0f} calls to train "
            f"({learner['episodes']:,.0f} episodes of "
            f"{learner['calls_per_episode']:,.0f} plus "
            f"{learner['demonstrations']:.0f} demonstrations) and "
            f"{learner['calls_per_episode']:,.0f} to run once, "
            f"stepping the environment {learner['training_steps']:,.0f} times to "
            f"train and {learner['steps_per_episode']:,.0f} to run once"
        )
        if matched:
            print(
                f"  cheapest cost a search reached within that budget: "
                f"{matched['cost']:.2f} ({matched['label']})"
            )
        summary[benchmark] = {
            "board": records[0]["board"],
            "seeds": len(records),
            "rows": rows,
            "learner": learner,
            "matched": matched,
        }
    path = BUDGET / "summary.json"
    path.write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {path}")


if __name__ == "__main__":
    report(tuple(sys.argv[1:]))
