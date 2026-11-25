# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""What each scheduler is worth against what it spends.

The four schedulers of the baseline comparison sit at two budgets and not four.
Greedy and the shortlisted policy score the same candidate set at every decision,
so both cost one greedy rollout; plain and guided beam expand the same nodes and
shortlist the same candidates at each of them, so both cost the same search. The
guidance is a forward pass per decision either way, which the cost model never
sees. Plotting the reduction against that budget therefore asks the question the
table cannot: what a search buys, and what is instead had for free.

    PYTHONPATH=. python report/plot_budget.py
    PYTHONPATH=. python report/plot_budget.py --out somewhere/fig_budget.pdf

Reads outputs/zeroshot/summary.csv and writes outputs/fig_budget.pdf unless
--out says otherwise.
"""

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parent.parent
SUMMARY = REPO / "outputs" / "zeroshot" / "summary.csv"
FIGURE = REPO / "outputs" / "fig_budget.pdf"

# The instance rows of the paper's table, in its order, with the label to print.
INSTANCES = [
    ("pool50_heldout T2", "Pauli 50"),
    ("qnn_50 T2", "qnn 50"),
    ("qaoa_14_on100 T2", "qaoa 14"),
    ("groundstate_14_on100 T2", "groundstate 14"),
    ("random_30_on100 T2", "random 30"),
    ("pool100_heldout T2", "Pauli 100"),
    ("qft_100 T2", "qft 100"),
    ("hubbard_100 T2", "hubbard 100"),
    ("random_30_on200 T2", "random 30"),
]
# Method, the setting it was run under, and what a rollout of it costs in greedy
# rollouts. Read off ``calls_mean``, which is the mean over the three initial
# configurations, as every other number in the paper is: over the nine instances
# plain beam spends 7.80 to 9.92 greedy rollouts and the guided beam 7.93 to 10.53,
# whose mean is the 8.7 the searches are drawn at. On qft-100 and hubbard-100,
# where greedy takes 659,721 and 434,352 evaluations, the shortlisted policy takes
# 1.04 and 0.95 of that, plain beam 7.80 and 9.92, guided beam 7.93 and 10.53.
# A budget is a count of candidates scored and so does not depend on the device
# parameters the reduction is read at.
METHODS = [
    # Error bars are the standard error over the three initial configurations,
    # which is an uncertainty on one method's mean. The comparison within an
    # instance is paired and so is better determined than the bars suggest.
    ("greedy", "base", 1.0, "Greedy search", "tab:gray", "o", "none"),
    ("learned", "base", 1.0, "Guided greedy", "tab:blue", "o", "full"),
    ("beam", "base", 8.7, "Beam search", "tab:gray", "s", "none"),
    ("guided_beam", "guided k8", 8.7, "Guided beam", "tab:red", "s", "full"),
]
# Both members of a budget tier would land on one abscissa, so they are drawn a
# little apart. The axis is logarithmic, hence a factor rather than an offset.
NUDGE = 1.11


def load():
    rows = list(csv.DictReader(open(SUMMARY)))
    table = {}
    for row in rows:
        key = (row["instance"], row["method"], row["setting"])
        table[key] = row
    data = {}
    for instance, label in INSTANCES:
        # Three benchmarks appear once per pool grid, so the instance name rather
        # than the printed label has to be the key.
        points = []
        for method, setting, budget, *_ in METHODS:
            row = table.get((instance, method, setting))
            if row is None:
                raise KeyError(f"{instance} has no {method} at {setting}")
            points.append(
                (
                    budget,
                    float(row["saved_vs_no_reconfiguration_percent_mean"]),
                    float(row["saved_vs_no_reconfiguration_percent_sem"]),
                )
            )
        data[instance] = points
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=FIGURE)
    figure_path = parser.parse_args().out

    data = load()
    # Drawn at the width of one column of the journal's two column layout, so
    # that nothing is scaled down and the labels stay the size they were set in.
    plt.rcParams.update({"font.size": 8, "axes.labelsize": 8})
    figure, axis = plt.subplots(figsize=(3.4, 2.7))

    # One faint line per instance through all four of its points, in order of
    # budget. Only methods on the same line may be compared, since the instances
    # differ in what there is to save, and the line is what carries that.
    for points in data.values():
        axis.plot(
            [
                points[0][0] / NUDGE,
                points[1][0] * NUDGE,
                points[2][0] / NUDGE,
                points[3][0] * NUDGE,
            ],
            [points[0][1], points[1][1], points[2][1], points[3][1]],
            color="0.82",
            linewidth=0.7,
            zorder=1,
        )

    for index, (method, setting, budget, label, colour, marker, fill) in enumerate(
        METHODS
    ):
        x = budget / NUDGE if index % 2 == 0 else budget * NUDGE
        values = [points[index][1] for points in data.values()]
        errors = [points[index][2] for points in data.values()]
        axis.errorbar(
            [x] * len(values),
            values,
            yerr=errors,
            fmt=marker,
            fillstyle=fill,
            color=colour,
            markersize=4,
            markeredgewidth=1.0,
            elinewidth=0.7,
            capsize=0,
            linestyle="none",
            label=label,
            zorder=3,
        )

    axis.set_xscale("log")
    axis.set_xticks([1.0, 8.7])
    axis.set_xticklabels(["1", "8.7"])
    axis.set_xlim(0.72, 12.5)
    axis.minorticks_off()
    axis.set_xlabel("Cost model evaluations, in greedy rollouts")
    axis.set_ylabel("Cost removed vs.\nno reconfiguration (%)")
    axis.legend(frameon=False, fontsize=7, loc="lower right", ncol=2, handletextpad=0.3, columnspacing=1.0)
    axis.spines[["top", "right"]].set_visible(False)
    figure.tight_layout()
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(figure_path)
    print(f"wrote {figure_path}")

    # The claim the figure is there to make, stated in numbers as well.
    beaten = [
        label
        for label, points in data.items()
        if points[1][1] > points[2][1]
    ]
    print(
        f"guided greedy beats plain beam on {len(beaten)} of {len(data)} instances: "
        + ", ".join(beaten)
    )


if __name__ == "__main__":
    main()
