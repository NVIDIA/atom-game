#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# The two policies the paper reports, from nothing but the configuration files.
#
# The defaults in conf/train.yaml are not the reported recipe. They are the
# starting point several arms of the search were explored from, so a bare
# `python bin/imitate.py --config-name=transfer_100` fits a different model: without
# `policy.geometry=add` the pair geometry of the paper's Section IV B 1 c is not in
# the network at all, without `imitation.demo_policy=lookahead` the demonstrations
# come from a sampled greedy scheduler rather than the lookahead teacher, and
# without `imitation.shortlist=3` the policy is asked to rank the whole board. This
# script carries every override that separates the two, so that what it produces is
# what Table I was measured on.
#
#     POOL=50  PYTHON=/path/to/env/bin/python bash experiments/reproduce.sh collect
#     POOL=50  PYTHON=/path/to/env/bin/python bash experiments/reproduce.sh fit
#     POOL=100 PYTHON=/path/to/env/bin/python bash experiments/reproduce.sh collect
#     POOL=100 PYTHON=/path/to/env/bin/python bash experiments/reproduce.sh fit
#
# `collect` is CPU only and embarrassingly parallel; it is where the hours go, of
# order a hundred core-hours for the larger pool. `fit` wants one GPU and takes a
# few hours. Both pools must be collected before either is fit, since the fit reads
# every demonstration file its glob matches.
set -eu

# Work from the repository root whatever directory the caller is in, and put it
# on the import path so the scripts under bin/ and report/ can import qds.
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export PYTHONPATH=.${PYTHONPATH:+:$PYTHONPATH}

python=${PYTHON:-python}
pool=${POOL:-100}

case "$pool" in
  50)
    config=transfer_50
    # ``pool50q`` is the pool at a single qubit count of 50, which is what the
    # paper reports; ``pool50`` is the earlier draw over 40 to 50 qubits, kept so
    # that the two can be compared.
    circuits=outputs/pool50q/circuit.json
    # Three episodes per collector, twenty four collectors: 72 episodes and
    # 208,591 demonstrated decisions, as Appendix B of the paper reports.
    seeds=$(seq 101 124)
    episodes=3
    demos='outputs/demo_data/tr50q_1*.npz'
    prefix=outputs/demo_data/tr50q
    weights=outputs/pool50q/la_mw2.pt
    # Deciding to move at all is rarer than deciding where to move, and the
    # weight that buys the first without spoiling the second differs with the
    # pool. Thirty epochs here against twenty on the larger pool.
    move_weight=2
    epochs=30
    eval_episodes=5
    ;;
  100)
    config=transfer_100
    circuits=outputs/pool100/circuit.json
    # Two episodes per collector, thirty two collectors: 64 episodes and 391,366
    # decisions. Seeds 401-432 were collected afterwards and fed a second fit
    # that the paper does not report, so the glob below is restricted to 3xx.
    seeds=$(seq 301 332)
    episodes=2
    demos='outputs/demo_data/tr100la_3*.npz'
    prefix=outputs/demo_data/tr100la
    weights=outputs/pool100/mw25.pt
    move_weight=2.5
    epochs=20
    eval_episodes=3
    ;;
  *)
    echo "POOL must be 50 or 100" >&2
    exit 1
    ;;
esac

collect() {
  mkdir -p outputs/demo_data
  for seed in $seeds; do
    # The teacher is the lookahead search: the three cheapest sites by the
    # estimator, two whole turns of greedy play from each, cheapest continuation
    # kept. It cuts at whole turns because a reconfiguration is paid for when its
    # last atom has moved.
    CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 "$python" bin/imitate.py \
      --config-name="$config" \
      "circuit_path=$circuits" \
      "seed=$seed" \
      "imitation.episodes=$episodes" \
      imitation.demo_policy=lookahead \
      '+lookahead={k:3,turns:2}' \
      imitation.collect_only=true \
      "imitation.dataset_path=${prefix}_${seed}.npz" \
      "exp_tag=pool${pool}" &
  done
  wait
}

fit() {
  "$python" bin/imitate.py \
    --config-name="$config" \
    "circuit_path=$circuits" \
    seed=0 \
    policy.geometry=add \
    "imitation.dataset_inputs=$demos" \
    imitation.shortlist=3 \
    "imitation.move_weight=$move_weight" \
    imitation.demo_weight=1.0 \
    imitation.soft_temperature=0.5 \
    imitation.final_temperature=0.5 \
    imitation.select=move \
    "imitation.epochs=$epochs" \
    "imitation.eval_episodes=$eval_episodes" \
    "imitation.save_path=$weights" \
    "exp_tag=p${pool}_mw${move_weight/./}"
}

case "${1:-}" in
  collect) collect ;;
  fit) fit ;;
  *)
    echo "usage: POOL={50|100} bash experiments/reproduce.sh {collect|fit}" >&2
    exit 1
    ;;
esac
