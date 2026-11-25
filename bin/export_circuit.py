# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Write a fixed benchmark's circuit in the form the transferable path reads.

A policy fitted on a pool can be run on any circuit, but only if the circuit
arrives the way the pool does, one JSON object per line. Exporting the benchmark
circuits is therefore what lets the transferable weights be evaluated on qft or
hubbard instead of on more random Pauli circuits, which is the strongest test of
transfer available without training anything new.

The board is not exported. Site embeddings are learned per site index, so weights
only mean anything on the grid they were fitted on: the pool's grid and the
benchmark's have to agree, which is why the shape and the atom count are printed.

    PYTHONPATH=. python bin/export_circuit.py qft_100 hubbard_100
"""

import sys
from pathlib import Path

from report.episode_budget import resolve
from qds.task.container import Container

OUT = Path(__file__).resolve().parent.parent / "outputs" / "zeroshot"


def export(name, seed=0):
    resolved = resolve(
        name, seed, overrides=[f"circuit_cache=outputs/circuits/{name}_s{seed}.pkl"]
    )
    env = Container(resolved.mode).get_env(resolved).unwrapped
    path = OUT / f"{name}_s{seed}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(env.reader.to_json() + "\n")
    print(
        f"{name} seed {seed}: {resolved.benchmark.id}, "
        f"{env.reader.num_qubits()} atoms, {env.reader.n_components()} components, "
        f"board {resolved.board.row}x{resolved.board.col} "
        f"({resolved.board.row * resolved.board.col} sites) -> {path}",
        flush=True,
    )
    env.close()
    return path


if __name__ == "__main__":
    for argument in sys.argv[1:]:
        name, _, seed = argument.partition(":")
        export(name, int(seed or 0))
