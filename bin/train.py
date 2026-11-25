# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import re
from datetime import datetime
from pathlib import Path

import gymnasium as gym
import hydra
import torch
import torch.distributed as dist
from ml_collections import ConfigDict
from omegaconf import DictConfig, OmegaConf

from qds.task.container import Container
from qds.task.ppo_ddp import PPOTask


CHECKPOINT_PATTERN = re.compile(r"model_iter_(\d+)\.pt")


def make_env_fn(container: Container, config: ConfigDict):
    def make():
        return gym.wrappers.RecordEpisodeStatistics(container.get_env(config))

    return make


def build_vector_env(container: Container, config: ConfigDict):
    """Step the environments in worker processes when asked.

    Observation construction dominates environment time, so spreading the
    environments over cores keeps large ``num_envs`` affordable.
    """
    if not config.training.get("async_envs", False):
        factories = [
            make_env_fn(container, config) for _ in range(config.training.num_envs)
        ]
        return gym.vector.SyncVectorEnv(factories)

    # Build one environment here so the circuit and board land in the container
    # cache; workers then inherit them by pickle instead of rebuilding.
    container.get_env(config).close()
    factories = [
        make_env_fn(container, config) for _ in range(config.training.num_envs)
    ]
    return gym.vector.AsyncVectorEnv(
        factories, shared_memory=True, context="spawn"
    )


def latest_checkpoint(path: str | os.PathLike[str]):
    model_dir = Path(path)
    if not model_dir.is_dir():
        return None, 0
    checkpoints = []
    for candidate in model_dir.iterdir():
        match = CHECKPOINT_PATTERN.fullmatch(candidate.name)
        if match:
            checkpoints.append((int(match.group(1)), candidate))
    if not checkpoints:
        return None, 0
    iteration, checkpoint_path = max(checkpoints, key=lambda item: item[0])
    return checkpoint_path, iteration


def prepare_config(config: DictConfig) -> ConfigDict:
    resolved = ConfigDict(OmegaConf.to_container(config, resolve=True))
    if resolved.mode == "fixed":
        problem, n_atoms = resolved.benchmark.id.rsplit("-", maxsplit=1)
        resolved.benchmark.problem = problem
        resolved.benchmark.n_atoms = int(n_atoms)
    resolved.run_name = (
        resolved.run_name_prefix + datetime.now().strftime("run_%m%d_%H_%M")
    )
    return resolved


def load_pretrained_policy(policy, config: ConfigDict) -> None:
    checkpoint_path, _ = latest_checkpoint(config.test.training_model_path)
    if checkpoint_path is None:
        raise ValueError(
            f"No pretrained model found in {config.test.training_model_path}"
        )
    state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    policy.load_state_dict(state_dict, strict=True)
    print(f"Loaded pretrained model from {checkpoint_path}")


@hydra.main(version_base="1.3", config_path="../conf", config_name="train")
def main(config: DictConfig) -> None:
    config = prepare_config(config)
    Path(config.tmp_dir).mkdir(parents=True, exist_ok=True)
    container = Container(config.mode)

    if config.task == "warmup":
        if config.mode != "transferable":
            raise ValueError("warmup is only supported in transferable mode")
        if int(os.environ.get("RANK", "0")) == 0:
            container.prepare(config)
        return

    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ.get("LOCAL_RANK", dist.get_rank()))
    torch.cuda.set_device(local_rank)

    envs = None
    try:
        envs = build_vector_env(container, config)
        policy = container.get_policy(config)
        if config.task == "test" and config.load_model:
            load_pretrained_policy(policy, config)
        elif config.get("init_model"):
            # Imitation weights reach the heuristic's quality on their own, so
            # starting from them turns training into a search around it.
            policy.load_state_dict(
                torch.load(config.init_model, map_location="cpu", weights_only=True),
                strict=True,
            )
            print(f"Initialised policy from {config.init_model}")

        PPOTask(config, envs, container.get_logger(config)).run(
            policy, local_rank
        )
    finally:
        if envs is not None:
            envs.close()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
