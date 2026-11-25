# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# torchrun --standalone --nproc_per_node=1 bin/train.py --config-name=transfer_100 seed=0 training.num_envs=100 training.num_iterations=10 task=test load_model=true exp_tag=v2_0_speed
# torchrun --standalone --nproc_per_node=1 bin/train.py --config-name=transfer_100 seed=1 training.num_envs=100 training.num_iterations=10 task=test load_model=true exp_tag=v2_1_speed
# torchrun --standalone --nproc_per_node=1 bin/train.py --config-name=transfer_100 seed=2 training.num_envs=100 training.num_iterations=20 task=test load_model=true exp_tag=v2_2_speed
# Work from the repository root whatever directory the caller is in, and put it
# on the import path so the scripts under bin/ and report/ can import qds.
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export PYTHONPATH=.${PYTHONPATH:+:$PYTHONPATH}

torchrun --standalone --nproc_per_node=1 bin/train.py --config-name=transfer_100 seed=0 training.num_envs=10 training.num_iterations=10 task=test load_model=true exp_tag=v2_0_speed
torchrun --standalone --nproc_per_node=1 bin/train.py --config-name=transfer_100 seed=2 training.num_envs=1 training.num_iterations=20 task=test load_model=true exp_tag=v2_2_speed
