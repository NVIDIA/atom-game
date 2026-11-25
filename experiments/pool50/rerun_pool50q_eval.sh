#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Re-measure every zero shot row that the smaller pool's weights decide.
#
# The larger pool's rows are not touched: their weights and their circuits are
# unchanged, so re-running them would only add noise to a table they already fill.
# The matrix below is the one the archived logs hold, so the regenerated table has
# the same cells as the submitted one:
#
#   five instances x {T_G = 10, T_G = 2} x {plain, guided k8 without the critic}
#   three initial configurations each, ten for groundstate, whose spread is widest
#   plus the one guided run that reads the critic, on qnn, for the settings sweep
#
#     PYTHON=/path/to/env/bin/python bash experiments/pool50/rerun_pool50q_eval.sh
set -u
# Work from the repository root whatever directory the caller is in, and put it
# on the import path so the scripts under bin/ and report/ can import qds.
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
export PYTHONPATH=.${PYTHONPATH:+:$PYTHONPATH}


export PYTHON=${PYTHON:-python}
export DEVICES=${DEVICES:-0,1,2,3,4,5,6,7}
rows=${ROWS:-"pool50_heldout qnn_50 qaoa_14_on100 random_30_on100"}
wide=${WIDE:-groundstate_14_on100}
wide_seeds=${WIDE_SEEDS:-"0 1 2 3 4 5 6 7 8 9"}

step() {
  echo "STEP_START $*"
  env "$@" bash experiments/zeroshot.sh $rows
  echo "STEP_DONE $*"
}

wide_step() {
  echo "STEP_START wide $*"
  env SEEDS="$wide_seeds" "$@" bash experiments/zeroshot.sh $wide
  echo "STEP_DONE wide $*"
}

# Plain, both zones. The baselines live here: greedy, never moving, plain beam and
# the weights as the scheduler.
step ZONE=10
step ZONE=2

# The same beam with the model choosing which sites get a subtree. VALUE_WEIGHT=0
# keeps the actor only, which is the setting the paper's guided column reports.
step ZONE=10 GUIDED=1 VALUE_WEIGHT=0 BEAM_SHORTLIST=8
step ZONE=2 GUIDED=1 VALUE_WEIGHT=0 BEAM_SHORTLIST=8

wide_step ZONE=10
wide_step ZONE=2
wide_step ZONE=10 GUIDED=1 VALUE_WEIGHT=0 BEAM_SHORTLIST=8
wide_step ZONE=2 GUIDED=1 VALUE_WEIGHT=0 BEAM_SHORTLIST=8

# The critic's own row, which only the settings sweep reads. Three initial
# configurations, as the submitted sweep had, though the logs it came from predate
# the shortlist going into the file name and so land under a different name here.
echo "STEP_START qnn critic"
env ZONE=10 GUIDED=1 VALUE_WEIGHT=1.0 BEAM_SHORTLIST=8 bash experiments/zeroshot.sh qnn_50
echo "STEP_DONE qnn critic"

echo "EVAL_ALL_DONE"
