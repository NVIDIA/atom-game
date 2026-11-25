# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The rows of the paper's baseline table, read off the regenerated summary.

The table quotes the standard error over initial configurations, which the summary
already carries, and marks in bold a guided scheduler that takes more off the no
reconfiguration cost than the baseline it guides.

    PYTHONPATH=. python report/baselines_rows.py
"""

import csv
from collections import defaultdict
from pathlib import Path

SUMMARY = Path(__file__).resolve().parent.parent / "outputs" / "zeroshot" / "summary.csv"

# Printed label, qubit count, and the instance key at each inter-zone time.
ROWS = [
    ("Pauli, held out", "50", "pool50_heldout"),
    ("qnn", "50", "qnn_50"),
    (r"qaoa$^\dagger$", "14", "qaoa_14_on100"),
    (r"groundstate$^{\dagger\ddagger}$", "14", "groundstate_14_on100"),
    (r"random$^\dagger$", "30", "random_30_on100"),
    ("Pauli, held out", "100", "pool100_heldout"),
    ("qft", "100", "qft_100"),
    ("hubbard", "100", "hubbard_100"),
    (r"random$^\dagger$", "30", "random_30_on200"),
]
# Baseline, and the guided scheduler whose entry is bold when it beats that baseline.
PAIRS = [("greedy", "learned", "base"), ("beam", "guided_beam", "guided k8")]


def read():
    saved = defaultdict(dict)
    with SUMMARY.open() as handle:
        for row in csv.DictReader(handle):
            mean = row["saved_vs_no_reconfiguration_percent_mean"]
            if not mean:
                continue
            saved[(row["instance"], row["setting"])][row["method"]] = (
                float(mean),
                float(row["saved_vs_no_reconfiguration_percent_sem"]),
            )
    return saved


def cell(entry, bold):
    mean, sem = entry
    body = f"{mean:.2f} \\pm {sem:.2f}"
    return f"$\\mathbf{{{body}}}$" if bold else f"${body}$"


def half(saved, instance, suffix):
    out = []
    for baseline, guided, setting in PAIRS:
        plain = saved[(instance + suffix, "base")].get(baseline)
        with_model = saved[(instance + suffix, setting)].get(guided)
        if plain is None or with_model is None:
            return None
        out.append(cell(plain, False))
        out.append(cell(with_model, with_model[0] > plain[0]))
    return out


def main():
    saved = read()
    for label, qubits, instance in ROWS:
        near = half(saved, instance, " T2")
        far = half(saved, instance, "")
        if near is None or far is None:
            print(f"% missing {instance}")
            continue
        print(f"{label} & {qubits} & " + " & ".join(near) + " & & " + " & ".join(far) + r" \\")


if __name__ == "__main__":
    main()
