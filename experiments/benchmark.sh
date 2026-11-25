# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# torchrun --standalone --nproc_per_node=8 bin/train.py --multirun benchmark.id=qaoa-14 seed=0 board.row=10 board.col=4 training.num_envs=20 training.num_iterations=500
# torchrun --standalone --nproc_per_node=8 bin/train.py --multirun benchmark.id=groundstate-14 seed=0 board.row=10 board.col=4 training.num_envs=20 training.num_iterations=500
# torchrun --standalone --nproc_per_node=8 bin/train.py --multirun benchmark.id=random-30 seed=0 board.row=10 board.col=6 training.num_envs=20 training.num_iterations=500
# torchrun --standalone --nproc_per_node=8 bin/train.py --multirun benchmark.id=qnn-50 seed=0 board.row=20 board.col=5 training.num_envs=20 training.num_iterations=2000
# torchrun --standalone --nproc_per_node=8 bin/train.py --multirun benchmark.id=qnn-lite-100 seed=0 board.row=20 board.col=10 training.num_envs=20 training.num_iterations=2000
# torchrun --standalone --nproc_per_node=8 bin/train.py --multirun benchmark.id=hubbard-100 seed=0 board.row=20 board.col=10 training.num_envs=10 training.num_iterations=4000
# torchrun --standalone --nproc_per_node=8 bin/train.py --multirun benchmark.id=qft-100 seed=0 board.row=20 board.col=10 training.num_envs=20 training.num_iterations=2000
# torchrun --standalone --nproc_per_node=8 bin/train.py --multirun benchmark.id=qaoa-14 seed=1 board.row=10 board.col=4 training.num_envs=20 training.num_iterations=2000
# torchrun --standalone --nproc_per_node=8 bin/train.py --multirun benchmark.id=groundstate-14 seed=1 board.row=10 board.col=4 training.num_envs=20 training.num_iterations=2000
# torchrun --standalone --nproc_per_node=8 bin/train.py --multirun benchmark.id=random-30 seed=1 board.row=10 board.col=6 training.num_envs=20 training.num_iterations=2000
# torchrun --standalone --nproc_per_node=8 bin/train.py --multirun benchmark.id=qnn-50 seed=1 board.row=20 board.col=5 training.num_envs=20 training.num_iterations=10000
# torchrun --standalone --nproc_per_node=8 bin/train.py --multirun benchmark.id=hubbard-100 seed=1 board.row=20 board.col=10 training.num_envs=10 training.num_iterations=4000
# torchrun --standalone --nproc_per_node=8 bin/train.py --multirun benchmark.id=qft-100 seed=1 board.row=20 board.col=10 training.num_envs=20 training.num_iterations=10000
# torchrun --standalone --nproc_per_node=8 bin/train.py --multirun benchmark.id=qaoa-14 seed=2 board.row=10 board.col=4 training.num_envs=20 training.num_iterations=2000
# torchrun --standalone --nproc_per_node=8 bin/train.py --multirun benchmark.id=groundstate-14 seed=2 board.row=10 board.col=4 training.num_envs=20 training.num_iterations=2000
# torchrun --standalone --nproc_per_node=8 bin/train.py --multirun benchmark.id=random-30 seed=2 board.row=10 board.col=6 training.num_envs=20 training.num_iterations=2000
# torchrun --standalone --nproc_per_node=8 bin/train.py --multirun benchmark.id=qnn-50 seed=2 board.row=20 board.col=5 training.num_envs=20 training.num_iterations=10000
# torchrun --standalone --nproc_per_node=8 bin/train.py --multirun benchmark.id=hubbard-100 seed=2 board.row=20 board.col=10 training.num_envs=10 training.num_iterations=10000
# Work from the repository root whatever directory the caller is in, and put it
# on the import path so the scripts under bin/ and report/ can import qds.
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export PYTHONPATH=.${PYTHONPATH:+:$PYTHONPATH}

torchrun --standalone --nproc_per_node=6 bin/train.py --multirun benchmark.id=qft-100 seed=2 board.row=20 board.col=10 training.num_envs=20 training.num_iterations=10000


# python bin/train.py --multirun benchmark.id=qft-100 seed=2,3,4 board.row=20 board.col=10
# python bin/train.py --multirun benchmark.id=hubbard-200 seed=2,3,4 board.row=40 board.col=10
# python bin/train.py --multirun benchmark.id=hubbard-100 seed=2,3,4 board.row=20 board.col=10
# python bin/train.py --multirun benchmark.id=hubbard-100 seed=4 board.row=20 board.col=10
# python bin/train.py --multirun benchmark.id=qnn-50 seed=0,1,2,3,4 board.row=20 board.col=5
# python bin/train.py --multirun benchmark.id=qnn-50 seed=3,4 board.row=20 board.col=5
# python bin/train.py --multirun benchmark.id=random-50 seed=0,1,2,3,4 board.row=20 board.col=5
# python bin/train.py --multirun benchmark.id=qaoa-14 seed=0,1,2,3,4 board.row=10 board.col=4
# python bin/train.py --multirun benchmark.id=groundstate-14 seed=0,1,2,3,4 board.row=10 board.col=4
# python bin/train.py --multirun benchmark.id=grover-noancilla seed=0,1,2,3,4 board.row=10 board.col=4
# python bin/train.py --multirun benchmark.id=hubbard-200 seed=0,1,2,3,4 board.row=40 board.col=10 device.inter_zone_time=5
# python bin/train.py --multirun benchmark.id=hubbard-200 seed=3,4 board.row=40 board.col=10 device.inter_zone_time=5
# python bin/train.py --multirun benchmark.id=qft-100 seed=0,1,2,3,4 board.row=20 board.col=10
# python bin/train.py --multirun benchmark.id=hubbard-100 seed=0,1,2,3,4 board.row=20 board.col=10 device.inter_zone_time=5
# python bin/train.py --multirun benchmark.id=qnn-50 seed=0,1,2,3,4 board.row=20 board.col=5 device.inter_zone_time=5
# python bin/train.py --multirun benchmark.id=random-50 seed=0,1,2,3,4 board.row=20 board.col=5 device.inter_zone_time=5
# python bin/train.py --multirun benchmark.id=qaoa-14 seed=0,1,2,3,4 booard.row=10 board.col=4 device.inter_zone_time=5
# python bin/train.py --multirun benchmark.id=groundstate-14 seed=0,1,2,3,4 booard.row=10 board.col=4 device.inter_zone_time=5
