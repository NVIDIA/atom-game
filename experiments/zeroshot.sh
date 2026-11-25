#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# The pool trained policy on the paper's benchmark circuits, which it never saw.
#
# The weights carry a learned embedding per site, so they only mean anything on the
# grid they were fitted on: transfer_100 is 20x10 and transfer_50 is 20x5. Three of
# the paper's instances are already measured on one of those grids, so their rows
# are directly comparable with the paper's. The other three are measured on smaller
# boards, and putting them on a pool's grid changes the instance: it is a different
# and easier problem than the paper's, and only the methods within a run may be
# compared. It is still worth measuring, because 14 atoms on 100 sites is nothing
# like the 40 to 50 the pool holds, and the atom count is the part of the
# distribution nothing else here varies.
#
# The atom count only has to fit under ``benchmark.max_atoms``; the observation is
# padded up to it. Nothing is trained here. Every method in a run plays the same
# circuit from the same board, so a comparison is internal to the run.
#
# A budget above one lets the policy search as well, by sampling that many rollouts
# and keeping the cheapest. Under a shortlist each rollout costs about what greedy
# costs, so the budget converts to cost model evaluations and can be set against
# beam's: on qft-100 beam spends roughly eight rollouts' worth.
#
# ``GUIDED`` runs the beam with the weights inside it instead of beside it, which is
# the comparison a sampled budget cannot make: the same search, the same number of
# nodes, the same calls to the estimator, and the model deciding which sites get a
# subtree and which half built plans survive. ``VALUE_WEIGHT=0`` keeps only the
# first of those, which separates what the actor contributes from what the critic
# does.
#
# ``ZONE`` overrides ``device.inter_zone_time``, the travel to the gate zone and
# back, which is charged once per gate step whatever a scheduler decides and so
# enters every cost as a constant. Lowering it does not change the problem, only how
# much of each cost is the part nobody can schedule away, and every ratio is then
# read against a smaller denominator. Runs at a different zone are logged and
# reported separately, never pooled with the default of 10.
#
#     PYTHON=/path/to/env/bin/python bash experiments/zeroshot.sh
#     PYTHON=/path/to/env/bin/python bash experiments/zeroshot.sh qnn_50
#     BUDGET=64 PYTHON=/path/to/env/bin/python bash experiments/zeroshot.sh qft_100
#     GUIDED=1 PYTHON=/path/to/env/bin/python bash experiments/zeroshot.sh qft_100
#     DEVICES=4,5 GUIDED=1 PYTHON=/path/to/env/bin/python bash experiments/zeroshot.sh qnn_50
#     ZONE=2 bash experiments/zeroshot.sh && ZONE=2 GUIDED=1 VALUE_WEIGHT=0 bash experiments/zeroshot.sh
set -u
# Work from the repository root whatever directory the caller is in, and put it
# on the import path so the scripts under bin/ and report/ can import qds.
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export PYTHONPATH=.${PYTHONPATH:+:$PYTHONPATH}


python=${PYTHON:-python}
budget=${BUDGET:-1}
guided=${GUIDED:-0}
value_weight=${VALUE_WEIGHT:-1.0}
shortlist=${BEAM_SHORTLIST:-8}
zone=${ZONE:-10}
# Which initial configurations to schedule from. The uncertainty quoted in the
# paper is over these, so an instance whose spread is wide can be given more of
# them without redoing the rest.
#
#     SEEDS="3 4 5 6 7 8 9" bash experiments/zeroshot.sh groundstate_14_on100
seeds=${SEEDS:-"0 1 2"}
# The guided beam asks the model once per node and a node is one decision deep, so a
# run is hundreds of thousands of forward passes of one row each. Left on a single
# device the jobs of a grid queue behind each other on it, so they are dealt round
# robin over these instead. ``system.device`` is plain ``cuda``, which is whichever
# one card each job is shown.
devices=${DEVICES:-0,1,2,3}
IFS=, read -r -a cards <<<"$devices"
# The smaller pool's circuits and weights, overridable so that regenerating that
# pool does not touch the rows fitted on the larger one. ``pool50`` is the 40 to 50
# qubit pool the first submission reported; ``pool50q`` is the same pool at a single
# qubit count of 50.
pool50_circuits=${POOL50_CIRCUITS:-outputs/pool50q/circuit_test.json}
pool50_weights=${POOL50_WEIGHTS:-outputs/pool50q/la_mw2.pt}
mkdir -p outputs/zeroshot/logs

# A name to select and to log under, the circuits to schedule, how many of them,
# the pool whose board they go on, and the weights that pool's recipe won with:
# move weight 2 on 100 sites, 2.5 on 200 sites.
#
# The first two rows are the pool's own held out circuits, sparse Pauli time
# evolution, which is the only in distribution case here and so the row every other
# row is a departure from. Then the three instances the paper already measures on
# one of these grids. Then three the paper measures on smaller boards, which on a
# pool's grid are a different and easier instance than the paper's.
targets=(
  "pool50_heldout $pool50_circuits 10 transfer_50 $pool50_weights"
  "pool100_heldout outputs/pool100/circuit_test.json 10 transfer_100 outputs/pool100/mw25.pt"
  "qft_100 outputs/zeroshot/qft_100_s0.jsonl 1 transfer_100 outputs/pool100/mw25.pt"
  "hubbard_100 outputs/zeroshot/hubbard_100_s0.jsonl 1 transfer_100 outputs/pool100/mw25.pt"
  "qnn_50 outputs/zeroshot/qnn_50_s0.jsonl 1 transfer_50 $pool50_weights"
  "qaoa_14_on100 outputs/zeroshot/qaoa_14_s0.jsonl 1 transfer_50 $pool50_weights"
  "groundstate_14_on100 outputs/zeroshot/groundstate_14_s0.jsonl 1 transfer_50 $pool50_weights"
  "random_30_on100 outputs/zeroshot/random_30_s0.jsonl 1 transfer_50 $pool50_weights"
  "qaoa_14_on200 outputs/zeroshot/qaoa_14_s0.jsonl 1 transfer_100 outputs/pool100/mw25.pt"
  "groundstate_14_on200 outputs/zeroshot/groundstate_14_s0.jsonl 1 transfer_100 outputs/pool100/mw25.pt"
  "random_30_on200 outputs/zeroshot/random_30_s0.jsonl 1 transfer_100 outputs/pool100/mw25.pt"
)

run() {
  local name=$1 circuits=$2 count=$3 config=$4 weights=$5 seed=$6 card=$7
  local tag="${name}_s${seed}"
  if [ "$zone" != 10 ]; then
    tag="${tag}_z${zone}"
  fi
  local extra=('+policies=[greedy,no_reconfiguration,beam]' beam.turns=2)
  if [ "$guided" -eq 1 ]; then
    # Plain beam is deterministic and already measured on this circuit from this
    # board at these settings, so it is not paid for twice; what changes here is
    # only where the ranking comes from.
    tag="${tag}_guided_v${value_weight}_k${shortlist}"
    extra=(
      '+policies=[guided_beam]'
      beam.turns=2
      "+beam.shortlist=$shortlist"
      "+beam.value_weight=$value_weight"
    )
  elif [ "$budget" -gt 1 ]; then
    # Beam is a single deterministic rollout, so it is measured once at budget one
    # and not repeated here; what a budget adds is a restarted greedy to compare
    # the sampled policy against at the same number of rollouts.
    tag="${tag}_b${budget}"
    extra=(
      '+policies=[greedy,sampled_greedy]'
      "+budget=$budget"
      +sample_temperature=0.05
      +epsilon=0.002
    )
  fi
  CUDA_VISIBLE_DEVICES=$card "$python" bin/evaluate.py \
    --config-name="$config" \
    task=test \
    "seed=$seed" \
    policy.geometry=add \
    "test_circuit_path=$circuits" \
    "benchmark.test_count=$count" \
    imitation.shortlist=3 \
    "device.inter_zone_time=$zone" \
    "init_model=$weights" \
    "${extra[@]}" \
    "output=/tmp/zeroshot_${tag}" \
    >"outputs/zeroshot/logs/${tag}.log" 2>&1
}

launched=0
for entry in "${targets[@]}"; do
  read -r name circuits count config weights <<<"$entry"
  if [ $# -gt 0 ] && [[ " $* " != *" $name "* ]]; then
    continue
  fi
  for seed in $seeds; do
    card=${cards[$((launched % ${#cards[@]}))]}
    run "$name" "$circuits" "$count" "$config" "$weights" "$seed" "$card" &
    launched=$((launched + 1))
  done
done
wait
echo "done"
