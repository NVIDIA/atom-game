#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Cost against cost model evaluations, for every benchmark in the paper.
#
# One process per instance, three seeds each, written to outputs/search_budget as
# one file per instance and seed. The restart budgets differ because a rollout does
# not: on qnn-50 one greedy rollout evaluates the model two million times, so a
# hundred restarts there is not the same experiment as a hundred on qaoa-14. The
# deep end of the restart curve is not repeated here, since it is already recorded
# in outputs/search_bar and converts to calls by multiplying by the calls a rollout
# spends, which this measures.
#
# The project environment has to be active, as it does for bin/train.py, or an
# interpreter has to be named:
#
#     bash experiments/search_budget.sh
#     PYTHON=/path/to/env/bin/python bash experiments/search_budget.sh
set -u
# Work from the repository root whatever directory the caller is in, and put it
# on the import path so the scripts under bin/ and report/ can import qds.
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export PYTHONPATH=.${PYTHONPATH:+:$PYTHONPATH}


python=${PYTHON:-python}
instances=(qaoa_14 groundstate_14 random_30 qnn_50 qft_100 hubbard_100)
restarts=(100 100 10 3 5 5)

mkdir -p outputs/search_budget/logs

for index in "${!instances[@]}"; do
  instance=${instances[$index]}
  (
    for seed in 0 1 2; do
      "$python" bin/benchmark_search.py \
        --config-name "$instance" \
        "seed=$seed" \
        "search.restarts=${restarts[$index]}" \
        "circuit_cache=outputs/circuits/${instance}_s${seed}.pkl"
    done
  ) >"outputs/search_budget/logs/${instance}.log" 2>&1 &
  echo "launched $instance (pid $!)"
done

wait
echo "done"
