# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Draw a transferable pool's circuits once, before any collector asks for them.

The pool is generated lazily by ``Container`` the first time a run needs it, which
is fine for one process and a race for the twenty four collectors that fit a
policy: each would find the file missing and start drawing its own. So the cache is
built here, in one process, and the collectors only ever read it.

    PYTHONPATH=. python experiments/pool50/generate_pool.py --config-name=transfer_50 \
        circuit_path=outputs/pool50q/circuit.json \
        test_circuit_path=outputs/pool50q/circuit_test.json
"""

import random

import hydra
import numpy as np
from ml_collections import ConfigDict
from omegaconf import DictConfig, OmegaConf

from qds.task.container import Container


@hydra.main(version_base="1.3", config_path="../../conf", config_name="train")
def main(config: DictConfig) -> None:
    resolved = ConfigDict(OmegaConf.to_container(config, resolve=True))
    seed = resolved.get("seed", 0)
    for task in ("train", "test"):
        # A fresh Container per task: it memoises the circuits it loaded, and the
        # path it reads depends on the task.
        random.seed(seed)
        np.random.seed(seed)
        resolved.task = task
        circuits = Container(resolved.mode)._load_transferable_circuits(resolved)
        path = resolved.test_circuit_path if task == "test" else resolved.circuit_path
        print(f"{task}: {len(circuits)} circuits at {path}", flush=True)


if __name__ == "__main__":
    main()
