# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Conversion between the dict observation and the flat tensor the policy sees."""

import gymnasium as gym
import numpy as np


def unflatten_batch(space: gym.spaces.Dict, flat_tensor):
    """Split a ``[batch, features]`` tensor back into the observation entries."""
    observation = {}
    index = 0
    batch_size = flat_tensor.shape[0]
    for key, subspace in space.spaces.items():
        size = int(np.prod(subspace.shape))
        shape = subspace.shape
        if isinstance(subspace, gym.spaces.Discrete):
            size = subspace.n
            shape = (subspace.n,)
        observation[key] = flat_tensor[:, index : index + size].reshape(
            (batch_size,) + shape
        )
        index += size
    return observation
