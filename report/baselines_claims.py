# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Every aggregate the baseline section states, recomputed from the summary.

The prose quotes ranges and counts over the nine instances of the table, and they
have to be re-derived whenever a policy is refitted rather than carried over.

    PYTHONPATH=. python report/baselines_claims.py
"""

import csv
from collections import defaultdict
from pathlib import Path

SUMMARY = Path(__file__).resolve().parent.parent / "outputs" / "zeroshot" / "summary.csv"

INSTANCES = [
    "pool50_heldout",
    "qnn_50",
    "qaoa_14_on100",
    "groundstate_14_on100",
    "random_30_on100",
    "pool100_heldout",
    "qft_100",
    "hubbard_100",
    "random_30_on200",
]


def read():
    saved, calls = defaultdict(dict), defaultdict(dict)
    with SUMMARY.open() as handle:
        for row in csv.DictReader(handle):
            key = (row["instance"], row["setting"])
            if row["saved_vs_no_reconfiguration_percent_mean"]:
                saved[key][row["method"]] = float(
                    row["saved_vs_no_reconfiguration_percent_mean"]
                )
            if row["calls_mean"]:
                calls[key][row["method"]] = float(row["calls_mean"])
    return saved, calls


def methods(saved, instance, suffix):
    base = saved[(instance + suffix, "base")]
    guided = saved[(instance + suffix, "guided k8")]
    return {
        "greedy": base["greedy"],
        "guided greedy": base["learned"],
        "beam": base["beam"],
        "guided beam": guided["guided_beam"],
    }


def span(values):
    return f"{min(values):.2f} to {max(values):.2f}"


def main():
    saved, calls = read()
    for suffix, name in ((" T2", "T_G = 2"), ("", "T_G = 10")):
        rows = {i: methods(saved, i, suffix) for i in INSTANCES}
        print(f"== {name} ==")
        print("  greedy          ", span([r["greedy"] for r in rows.values()]))
        print("  guided greedy   ", span([r["guided greedy"] for r in rows.values()]))
        print("  beam            ", span([r["beam"] for r in rows.values()]))
        print("  guided beam     ", span([r["guided beam"] for r in rows.values()]))

        helps_greedy = [i for i, r in rows.items() if r["guided greedy"] > r["greedy"]]
        helps_beam = [i for i, r in rows.items() if r["guided beam"] > r["beam"]]
        best = [i for i, r in rows.items() if max(r, key=r.get) == "guided beam"]
        over_beam = [i for i, r in rows.items() if r["guided greedy"] > r["beam"]]
        print(f"  guided greedy beats greedy on {len(helps_greedy)}/9",
              f"misses {sorted(set(INSTANCES) - set(helps_greedy))}")
        print(f"  guided beam beats beam on {len(helps_beam)}/9",
              f"misses {sorted(set(INSTANCES) - set(helps_beam))}")
        print(f"  guided beam is the largest entry on {len(best)}/9",
              f"misses {sorted(set(INSTANCES) - set(best))}")
        print(f"  guided greedy beats plain beam on {len(over_beam)}/9: {over_beam}")

        gains = sorted(
            ((r["guided beam"] - r["beam"], i) for i, r in rows.items()), reverse=True
        )
        print("  guidance on the beam, in points, largest first:")
        for points, instance in gains:
            print(f"    {points:+6.2f}  {instance}")
        print()

    # The call counts are quoted from the three placement runs, as every other
    # number in the paper is, and those are complete at the near gate zone. A
    # budget is a count of candidates scored, so it does not depend on the zone.
    print("== cost model evaluations for one rollout, against greedy ==")
    ratios = {"guided greedy": [], "beam": [], "guided beam": [], "guided beam over beam": []}
    for instance in INSTANCES:
        base = calls[(instance + " T2", "base")]
        guided = calls[(instance + " T2", "guided k8")]
        if "greedy" not in base:
            print(f"  {instance}: no call counts")
            continue
        unit = base["greedy"]
        learned = base.get("learned")
        beam = base.get("beam")
        gbeam = guided.get("guided_beam")
        parts = [f"  {instance:24s} greedy {unit:12,.0f}"]
        if learned:
            ratios["guided greedy"].append(learned / unit)
            parts.append(f"learned {learned / unit:.2f}x")
        if beam:
            ratios["beam"].append(beam / unit)
            parts.append(f"beam {beam / unit:.2f}x")
        if gbeam:
            ratios["guided beam"].append(gbeam / unit)
            parts.append(f"guided beam {gbeam / unit:.2f}x")
        if beam and gbeam:
            ratios["guided beam over beam"].append(gbeam / beam)
            parts.append(f"guided/beam {gbeam / beam:.3f}x")
        print("  ".join(parts))
    print()
    for name, values in ratios.items():
        if values:
            print(f"  {name}: {span(values)} over {len(values)} instances")


if __name__ == "__main__":
    main()
