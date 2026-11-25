#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Everything left between the fitted weights and a regenerated table, in one run.
#
# Three of the archived logs were written before the beam shortlist went into the
# file name, so the rerun lands beside them rather than on top of them, and the
# report would then read the old pool's numbers and the new pool's under one key.
# They are dropped here; the archive still holds them.
#
#     PYTHON=/path/to/env/bin/python bash experiments/pool50/finish_pool50q.sh
set -u
# Work from the repository root whatever directory the caller is in, and put it
# on the import path so the scripts under bin/ and report/ can import qds.
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
export PYTHONPATH=.${PYTHONPATH:+:$PYTHONPATH}


python=${PYTHON:-python}
export PYTHON="$python"
logs=outputs/zeroshot/logs
archive=outputs/zeroshot/logs_q40_50_archive

echo "STALE_START"
"$python" - "$logs" "$archive" <<'PY'
import os
import sys

logs, archive = sys.argv[1], sys.argv[2]
# Named rather than matched, so that a rerun cannot widen what this deletes.
stale = [
    "qnn_50_s0_guided_v1.0.log",
    "qnn_50_s1_guided_v1.0.log",
    "qnn_50_s2_guided_v1.0.log",
]
for name in stale:
    path = os.path.join(logs, name)
    if not os.path.exists(path):
        print(f"  already gone {name}")
        continue
    if not os.path.exists(os.path.join(archive, name)):
        raise SystemExit(f"refusing to delete {name}: not in {archive}")
    os.remove(path)
    print(f"  dropped {name}")
PY
echo "STALE_DONE"

bash experiments/pool50/rerun_pool50q_eval.sh

echo "REPORT_START"
"$python" report/zeroshot_report.py --csv outputs/zeroshot
echo "REPORT_DONE"
echo "FINISH_ALL_DONE"
