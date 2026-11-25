#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Demonstrations for the 50 qubit pool, by the lookahead teacher.
#
# The same construction the appendix describes for the smaller pool: twenty four
# independent collectors of three episodes each, differing only in their seed, and
# pooled at fit time. The seeds are the ones the 40--50 pool was collected under, so
# the initial configurations each collector draws are the same sequence and only the
# circuits differ.
#
# The pool's circuits must already exist; experiments/pool50/generate_pool.py draws them.
#
#     PYTHON=/path/to/env/bin/python bash experiments/pool50/collect_pool50q.sh
set -u
# Work from the repository root whatever directory the caller is in, and put it
# on the import path so the scripts under bin/ and report/ can import qds.
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
export PYTHONPATH=.${PYTHONPATH:+:$PYTHONPATH}


python=${PYTHON:-python}
pool=${POOL:-outputs/pool50q}
prefix=${PREFIX:-tr50q}
logs=outputs/demo_data/logs
mkdir -p outputs/demo_data "$logs"

for seed in $(seq 101 124); do
  CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 "$python" bin/imitate.py \
    --config-name=transfer_50 \
    "circuit_path=$pool/circuit.json" \
    "seed=$seed" \
    imitation.episodes=3 \
    imitation.demo_policy=lookahead \
    '+lookahead={k:3,turns:2}' \
    imitation.collect_only=true \
    "imitation.dataset_path=outputs/demo_data/${prefix}_${seed}.npz" \
    "output=/tmp/collect_${prefix}_${seed}" \
    >"$logs/${prefix}_${seed}.log" 2>&1 &
done
wait
echo "COLLECTION_DONE $(ls outputs/demo_data/${prefix}_*.npz 2>/dev/null | wc -l) files"
