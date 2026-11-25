# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tables from the zero shot runs, read off the logs `experiments/zeroshot.sh` leaves.

Every transfer number this project reports comes from here, so that a table is a
function of the runs on disk rather than of what was pasted into a message. One
log is one instance at one placement under one setting; a row is those
pooled across placements, and a percentage is always against greedy on exactly the
placements both methods were run on.

    PYTHONPATH=. python report/zeroshot_report.py
    PYTHONPATH=. python report/zeroshot_report.py --logs outputs/zeroshot/logs
    PYTHONPATH=. python report/zeroshot_report.py --csv outputs/zeroshot

The markdown tables quote one number per cell, which is the pooled mean; `--csv`
writes the spread as well. The unit there is a placement, because that is what was
repeated three times, so a standard deviation is over three values and a comparison
is paired, taken per placement and only then averaged. `episodes.csv` holds every
circuit of every placement, which is what to recompute from if the unit should be
an episode instead.

Runs made before `bin/evaluate.py` recorded its own settings are labelled from the file
name instead, which is what `experiments/zeroshot.sh` encodes there; the mapping is in
``label``. Ad hoc runs are skipped rather than guessed at.
"""

import argparse
import csv
import json
import re
import statistics as st
from pathlib import Path

LOGS = Path("outputs/zeroshot/logs")

# The order a table reads in: the floor, the baseline everything is measured
# against, the model as the scheduler, the search, the search with the model in it.
SCHEDULERS = ("no_reconfiguration", "greedy", "learned", "beam", "guided_beam")

# Suffixes a file name can carry, and what setting each one was. ``spent`` was an
# ad hoc pair of runs whose beam shortlist is not in the name, so it is only read
# for what it measured that nothing else did, which is the call counts.
NAMED = {"": "base", "_spent": "calls"}

# Instances whose board the paper already measures on come first, because their
# rows are the only ones comparable with anything outside this file.
FIRST = ("qft_100", "hubbard_100", "qnn_50")

# The zone travel time every earlier run used, and so the one a row carries no
# marker for. Anything else becomes part of the instance name.
DEFAULT_ZONE = 10.0


def read(path):
    """The JSON `bin/evaluate.py` prints, or nothing if the run did not finish."""
    text = path.read_text(errors="ignore")
    start = text.find("{")
    if start < 0:
        return None
    try:
        return json.loads(text[start:])
    except json.JSONDecodeError:
        return None


def label(name, data):
    """Instance, placement and setting for one log.

    The setting comes from the run itself when it recorded one. Otherwise it comes
    from the name, because that is where `experiments/zeroshot.sh` put it. The zone travel time
    is part of the instance rather than of the setting: it is charged whatever a
    scheduler decides, so two zones are two different problems and nothing may be
    pooled or compared across them.
    """
    match = re.match(r"(.+?)_s(\d+)(.*)", name)
    if match is None:
        return None
    instance, seed, suffix = match.group(1), int(match.group(2)), match.group(3)
    settings = data.get("settings")
    zone = (settings or {}).get("inter_zone_time")
    if zone is None:
        named = re.search(r"_z([\d.]+)", suffix)
        zone = named.group(1) if named else DEFAULT_ZONE
    if float(zone) != DEFAULT_ZONE:
        instance = f"{instance} T{float(zone):g}"
    suffix = re.sub(r"_z[\d.]+", "", suffix)
    if settings and suffix not in NAMED:
        beam = settings.get("beam", {})
        setting = (
            f"guided k{beam.get('shortlist', '?')}"
            f"{' critic' if float(beam.get('value_weight', 0) or 0) else ''}"
        )
    elif suffix in NAMED:
        setting = NAMED[suffix]
    elif suffix.startswith("_guided"):
        width = re.search(r"_k(\d+)", suffix)
        weight = re.search(r"_v([\d.]+)", suffix)
        critic = " critic" if weight and float(weight.group(1)) else ""
        setting = f"guided k{width.group(1) if width else 8}{critic}"
    elif suffix.startswith("_b"):
        setting = f"budget {suffix[2:]}"
    else:
        return None
    return instance, seed, setting


def collect(logs):
    """Costs and call counts per instance, setting, placement and policy.

    The second dictionary keeps every circuit rather than the placement's mean,
    which the markdown tables never need and a standard deviation might.
    """
    runs = {}
    episodes = {}
    for path in sorted(Path(logs).glob("*.log")):
        if path.stem.startswith("runner"):
            continue
        data = read(path)
        if data is None:
            continue
        named = label(path.stem, data)
        if named is None:
            continue
        instance, seed, setting = named
        entry = runs.setdefault(instance, {}).setdefault(setting, {})
        for policy, cost in data["mean_cost"].items():
            entry.setdefault(policy, {})[seed] = cost
        for policy, spent in data.get("spent", {}).items():
            entry.setdefault(policy + " calls", {})[seed] = spent["calls"]
        for k, cost in data.get("best_of_curve", {}).get("learned", {}).items():
            entry.setdefault(f"learned best of {k}", {})[seed] = cost
        for policy, series in data.get("per_episode", {}).items():
            place = episodes.setdefault(instance, {}).setdefault(setting, {})
            place.setdefault(policy, {})[seed] = [item["cost"] for item in series]
    order = sorted(runs, key=ordering)
    return {name: runs[name] for name in order}, episodes


def ordering(instance):
    name, _, zone = instance.partition(" T")
    return (
        bool(zone),
        FIRST.index(name) if name in FIRST else len(FIRST),
        name,
    )


def paired(left, right):
    """Means over the placements both were run on, and how often left won."""
    shared = sorted(set(left) & set(right))
    if not shared:
        return None
    return (
        st.fmean(left[seed] for seed in shared),
        st.fmean(right[seed] for seed in shared),
        sum(left[seed] < right[seed] for seed in shared),
        len(shared),
    )


def against(entry, policy, reference="greedy"):
    """A policy as a percentage of greedy, on the placements they share."""
    if policy not in entry or reference not in entry:
        return "—"
    pair = paired(entry[policy], entry[reference])
    if pair is None:
        return "—"
    return f"{100 * (pair[0] / pair[1] - 1):+.2f}%"


def spread(values):
    """Mean, sample standard deviation and standard error, over placements."""
    values = list(values)
    if not values:
        return None
    deviation = st.stdev(values) if len(values) > 1 else 0.0
    return st.fmean(values), deviation, deviation / len(values) ** 0.5


def relative(left, right):
    """The percentage difference taken per placement, then summarised.

    Pairing before averaging is the point: the placements differ from each other by
    far more than the methods differ on any one of them, so a difference of pooled
    means carries a spread that is mostly the placements and not the methods.
    """
    shared = sorted(set(left) & set(right))
    if not shared:
        return None
    return spread(100 * (left[seed] / right[seed] - 1) for seed in shared)


def improvement(left, right):
    """How much of `right` a method saves, positive meaning cheaper.

    This is the same paired quantity as `relative` with its sign turned over, and it
    is the one every table leads with. Greedy is the wrong denominator for saying
    what a scheduler is worth, because a percentage of greedy hides how much of the
    problem greedy had already solved and inflates as the constant nobody can
    schedule away is thinned. The stay in place cost is a property of the instance
    and of nothing else, so it is what everything here is judged against.
    """
    shared = sorted(set(left) & set(right))
    if not shared:
        return None
    return spread(100 * (1 - left[seed] / right[seed]) for seed in shared)


def points(left, right, floor):
    """The gap between two methods, as a share of the stay in place cost.

    A difference of two savings, so it is in the same currency as the columns it
    sits beside: how many more points of the floor `left` takes off than `right`.
    """
    shared = sorted(set(left) & set(right) & set(floor))
    if not shared:
        return None
    return spread(100 * (right[seed] - left[seed]) / floor[seed] for seed in shared)


def cell(pair):
    return f"{pair[0]:.2f} ± {pair[1]:.2f}" if pair else "—"


def columns(row, name, pair):
    """Three columns for one summarised quantity, blank where it does not apply."""
    for field, number in zip(("mean", "sd", "sem"), pair or ()):
        row[f"{name}_{field}"] = round(number, 4)
    for field in ("mean", "sd", "sem"):
        row.setdefault(f"{name}_{field}", "")


def summary(runs, episodes, path):
    """One row per instance, method and setting, with the spread over placements.

    The saving off the stay in place cost comes first because that is what a method
    is worth. The percentages against greedy and against beam are kept after it,
    for reading one scheduler against another.
    """
    rows = []
    for instance, settings in runs.items():
        base = settings.get("base", {})
        floor = base.get("no_reconfiguration", {})
        reference = {"greedy": base.get("greedy", {}), "beam": base.get("beam", {})}
        for setting, entry in sorted(settings.items()):
            for policy in SCHEDULERS:
                values = entry.get(policy)
                # A guided run replays the model's own rollout to have something
                # beside its search, and that row already exists under ``base``.
                if not values or (setting != "base" and policy in base):
                    continue
                counts = episodes.get(instance, {}).get(setting, {}).get(policy, {})
                cost = spread(values.values())
                calls = spread((entry.get(policy + " calls") or {}).values())
                beam = reference["beam"]
                row = {
                    "instance": instance,
                    "method": policy,
                    "setting": setting,
                    "placements": len(values),
                    "circuits_per_placement": (
                        len(next(iter(counts.values()))) if counts else ""
                    ),
                    "cost_mean": round(cost[0], 4),
                    "cost_sd": round(cost[1], 4),
                    "cost_sem": round(cost[2], 4),
                }
                columns(
                    row,
                    "saved_vs_no_reconfiguration_percent",
                    improvement(values, floor) if floor is not values else None,
                )
                columns(
                    row,
                    "gained_over_beam_points",
                    points(values, beam, floor) if floor and beam is not values else None,
                )
                for name, other in reference.items():
                    columns(
                        row,
                        f"vs_{name}_percent",
                        relative(values, other) if other is not values else None,
                    )
                shared = sorted(set(values) & set(beam))
                row["placements_beating_beam"] = (
                    f"{sum(values[seed] < beam[seed] for seed in shared)}/{len(shared)}"
                    if shared and beam is not values
                    else ""
                )
                row["calls_mean"] = round(calls[0]) if calls else ""
                row["calls_sd"] = round(calls[1]) if calls else ""
                rows.append(row)
    fields = list(rows[0])
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fields)
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def raw(episodes, path):
    """Every circuit of every placement, the values everything else is made of."""
    rows = []
    for instance, settings in sorted(episodes.items(), key=lambda item: item[0]):
        for setting, policies in sorted(settings.items()):
            for policy, places in sorted(policies.items()):
                for seed, costs in sorted(places.items()):
                    for index, cost in enumerate(costs):
                        rows.append(
                            {
                                "instance": instance,
                                "setting": setting,
                                "method": policy,
                                "placement": seed,
                                "circuit": index,
                                "cost": round(cost, 4),
                            }
                        )
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def table(headers, rows):
    print("| " + " | ".join(headers) + " |")
    print("| " + " | ".join("---" for _ in headers) + " |")
    for row in rows:
        print("| " + " | ".join(str(cell) for cell in row) + " |")
    print()


def schedulers(runs, setting="guided k8"):
    """One row per instance: what each method takes off the stay in place cost.

    The guided column is one uniform setting rather than each instance's best,
    because picking the best per instance is a choice made after seeing the answer.
    """
    rows = []
    for instance, settings in runs.items():
        base = settings.get("base", {})
        floor = base.get("no_reconfiguration")
        if not floor:
            continue
        guided = settings.get(setting, {}).get("guided_beam", {})
        row = [instance, f"{st.fmean(floor.values()):.2f}"]
        for policy in SCHEDULERS:
            if policy == "no_reconfiguration":
                continue
            values = guided if policy == "guided_beam" else base.get(policy)
            row.append(cell(improvement(values, floor)) if values else "—")
        beam = base.get("beam", {})
        row.append(cell(points(guided, beam, floor)) if guided and beam else "—")
        pair = paired(guided, beam) if guided else None
        row.append(f"{pair[2]}/{pair[3]}" if pair else "—")
        calls = paired(
            settings.get(setting, {}).get("guided_beam calls", {}),
            base.get("beam calls", {}),
        )
        row.append(f"{calls[0] / calls[1]:.3f}x" if calls else "—")
        rows.append(row)
    print(f"## What reconfiguring buys, and the guided beam at {setting}\n")
    print(
        "The no reconf column is the cost of never moving an atom, in the cost\n"
        "model's own units, and every other column is the percentage of it that\n"
        "method takes off, larger being better, taken per placement and then\n"
        "averaged, ± one standard deviation over the placements. Guided − beam is\n"
        "the same currency: how many more points of the floor the guidance is\n"
        "worth. Placements is how often the guided beam beat plain beam.\n"
    )
    table(
        [
            "instance",
            "no reconf",
            "greedy",
            "the weights",
            "beam",
            "guided",
            "guided − beam",
            "placements",
            "calls vs beam",
        ],
        rows,
    )


def sweep(runs):
    """What the shortlist width and the critic are worth, over plain beam."""
    settings = sorted(
        {name for entry in runs.values() for name in entry if name.startswith("guided")},
        key=lambda name: (("critic" in name), int(re.search(r"k(\d+)", name).group(1))),
    )
    rows = []
    for instance, found in runs.items():
        base = found.get("base", {})
        beam, floor = base.get("beam"), base.get("no_reconfiguration")
        if not beam or not floor:
            continue
        row = [instance, cell(improvement(beam, floor))]
        for setting in settings:
            guided = found.get(setting, {}).get("guided_beam", {})
            row.append(cell(points(guided, beam, floor)) if guided else "—")
        rows.append(row)
    print("## The guided beam's settings, over plain beam\n")
    print(
        "The beam column is what plain beam takes off the stay in place cost, and\n"
        "each setting is the points of that floor the guidance adds to it.\n"
    )
    table(["instance", "beam"] + settings, rows)


def budgets(runs):
    """Where a budget of rollouts was measured, what it bought."""
    rows = []
    for instance, settings in runs.items():
        for setting, entry in sorted(settings.items()):
            if not setting.startswith("budget"):
                continue
            base = settings.get("base", {})
            curve = sorted(
                (int(name.rsplit(" ", 1)[1]), values)
                for name, values in entry.items()
                if name.startswith("learned best of ")
            )
            for k, values in curve:
                if k not in (1, 8, 64):
                    continue
                rows.append(
                    [
                        instance,
                        k,
                        f"{st.fmean(values.values()):.2f}",
                        against({**entry, "best": values}, "best"),
                        against(base, "beam"),
                        against(entry, "sampled_greedy"),
                    ]
                )
    if not rows:
        return
    print("## Sampling the model against restarting greedy, at a budget\n")
    print(
        "The model's own best of k, against greedy on the same placements, with\n"
        "plain beam's single rollout alongside for the budget it costs.\n"
    )
    table(
        [
            "instance",
            "rollouts",
            "the weights, best of k",
            "vs greedy",
            "beam vs greedy",
            "restarted greedy vs greedy",
        ],
        rows,
    )


def calls(runs):
    """What a rollout of each method asks of the cost model."""
    rows = []
    for instance, settings in runs.items():
        # Greedy is the unit, and only one setting per instance ever runs it, so a
        # guided beam measured in its own run is still priced against that one.
        reference = next(
            (
                entry["greedy calls"]
                for entry in settings.values()
                if "greedy calls" in entry
            ),
            None,
        )
        for setting, entry in sorted(settings.items()):
            for policy in ("greedy", "learned", "beam", "guided_beam"):
                values = entry.get(policy + " calls")
                if values is None:
                    continue
                pair = paired(values, reference) if reference else None
                rows.append(
                    [
                        instance,
                        setting,
                        policy,
                        f"{st.fmean(values.values()):,.0f}",
                        f"{pair[0] / pair[1]:.2f}x" if pair else "—",
                    ]
                )
    if not rows:
        return
    print("## Cost model evaluations for one rollout\n")
    table(["instance", "setting", "method", "calls", "vs greedy"], rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--logs", default=LOGS)
    parser.add_argument("--guided", default="guided k8")
    parser.add_argument("--csv", help="write summary.csv and episodes.csv here")
    arguments = parser.parse_args()
    runs, episodes = collect(arguments.logs)
    if not runs:
        raise SystemExit(f"No finished runs under {arguments.logs}")
    if arguments.csv:
        folder = Path(arguments.csv)
        folder.mkdir(parents=True, exist_ok=True)
        for name, written in (
            ("summary.csv", summary(runs, episodes, folder / "summary.csv")),
            ("episodes.csv", raw(episodes, folder / "episodes.csv")),
        ):
            print(f"{folder / name}: {written} rows")
        return
    schedulers(runs, arguments.guided)
    sweep(runs)
    budgets(runs)
    calls(runs)


if __name__ == "__main__":
    main()
