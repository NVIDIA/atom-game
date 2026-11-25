# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Work from the repository root whatever directory the caller is in, and put it
# on the import path so the scripts under bin/ and report/ can import qds.
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export PYTHONPATH=.${PYTHONPATH:+:$PYTHONPATH}

torchrun --standalone --nproc_per_node=1 bin/train.py --config-name=transfer_50 seed=0 training.num_envs=20 training.num_iterations=100 task=test load_model=true exp_tag=v1_0_load
torchrun --standalone --nproc_per_node=1 bin/train.py --config-name=transfer_50 seed=0 training.num_envs=20 training.num_iterations=100 task=test load_model=false exp_tag=v1_0_noload
torchrun --standalone --nproc_per_node=1 bin/train.py --config-name=transfer_50 seed=1 training.num_envs=20 training.num_iterations=100 task=test load_model=true exp_tag=v1_1_load
torchrun --standalone --nproc_per_node=1 bin/train.py --config-name=transfer_50 seed=1 training.num_envs=20 training.num_iterations=100 task=test load_model=false exp_tag=v1_1_noload
torchrun --standalone --nproc_per_node=1 bin/train.py --config-name=transfer_50 seed=2 training.num_envs=20 training.num_iterations=100 task=test load_model=true exp_tag=v1_2_load
torchrun --standalone --nproc_per_node=1 bin/train.py --config-name=transfer_50 seed=2 training.num_envs=20 training.num_iterations=100 task=test load_model=false exp_tag=v1_2_noload
torchrun --standalone --nproc_per_node=1 bin/train.py --config-name=transfer_100 seed=0 training.num_envs=20 training.num_iterations=100 task=test load_model=true exp_tag=v2_0_load
torchrun --standalone --nproc_per_node=1 bin/train.py --config-name=transfer_100 seed=0 training.num_envs=20 training.num_iterations=100 task=test load_model=false exp_tag=v2_0_noload
torchrun --standalone --nproc_per_node=1 bin/train.py --config-name=transfer_100 seed=1 training.num_envs=20 training.num_iterations=100 task=test load_model=true exp_tag=v2_1_load
torchrun --standalone --nproc_per_node=1 bin/train.py --config-name=transfer_100 seed=1 training.num_envs=20 training.num_iterations=100 task=test load_model=false exp_tag=v2_1_noload
torchrun --standalone --nproc_per_node=1 bin/train.py --config-name=transfer_100 seed=2 training.num_envs=20 training.num_iterations=100 task=test load_model=true exp_tag=v2_2_load
torchrun --standalone --nproc_per_node=1 bin/train.py --config-name=transfer_100 seed=2 training.num_envs=20 training.num_iterations=100 task=test load_model=false exp_tag=v2_2_noload
