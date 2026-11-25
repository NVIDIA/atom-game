# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import unittest

import torch
from ml_collections import ConfigDict

from qds.task.ppo_ddp import BestEpisodeBuffer, PPOTask


def task(gamma=1.0, gae_lambda=1.0, adv_scope="batch"):
    instance = PPOTask.__new__(PPOTask)
    instance.config = ConfigDict(
        {
            "training": {
                "gamma": gamma,
                "gae_lambda": gae_lambda,
                "adv_scope": adv_scope,
            }
        }
    )
    return instance


class TurnGroupingTest(unittest.TestCase):
    def test_groups_start_after_the_previous_boundary(self):
        ends = torch.tensor(
            [
                [False, False],
                [True, False],
                [False, True],
                [False, False],
                [True, False],
            ]
        )

        groups = PPOTask.turn_groups(ends)

        self.assertEqual(groups[0], [(0, 2), (2, 5)])
        # The second environment's turn is still open at the end of the rollout.
        self.assertEqual(groups[1], [(0, 3)])

    def test_advantage_uses_the_first_value_and_the_whole_turn_reward(self):
        """One reconfiguration is one transition worth its own reward.

        With no discounting the advantage of the last turn is the reward it
        earned plus the value of the board it leaves behind, minus the value of
        the board its first atom saw.
        """
        rewards = torch.tensor([[0.0], [0.0], [-3.0], [0.0], [-5.0]])
        values = torch.tensor([[1.0], [2.0], [3.0], [4.0], [5.0]])
        buffers = {
            "rewards": rewards,
            "dones": torch.zeros(5, 1),
            "values": values,
        }
        groups = [[(0, 3), (3, 5)]]

        spans, advantages, returns, group_values = task().turn_advantages(
            groups, buffers, torch.tensor([7.0]), torch.zeros(1), task().config.training
        )

        self.assertEqual(spans, [(0, 3, 0), (3, 5, 0)])
        # Last turn: -5 + bootstrap 7 - value 4.
        self.assertAlmostEqual(advantages[1].item(), -2.0)
        # First turn: (-3 + 4 - 1) plus the advantage carried back from the last.
        self.assertAlmostEqual(advantages[0].item(), -2.0)
        self.assertAlmostEqual(returns[0].item(), -1.0)
        self.assertAlmostEqual(group_values[0].item(), 1.0)

    def test_a_terminal_boundary_stops_the_bootstrap(self):
        buffers = {
            "rewards": torch.tensor([[-3.0], [-5.0]]),
            "dones": torch.tensor([[0.0], [1.0]]),
            "values": torch.tensor([[1.0], [9.0]]),
        }

        _, advantages, _, _ = task().turn_advantages(
            [[(0, 1)]],
            buffers,
            torch.tensor([7.0]),
            torch.zeros(1),
            task().config.training,
        )

        self.assertAlmostEqual(advantages[0].item(), -4.0)

    def test_discounting_applies_per_turn_not_per_atom(self):
        buffers = {
            "rewards": torch.tensor([[0.0], [-2.0], [0.0], [-2.0]]),
            "dones": torch.zeros(4, 1),
            "values": torch.zeros(4, 1),
        }

        _, advantages, _, _ = task(gamma=0.5, gae_lambda=1.0).turn_advantages(
            [[(0, 2), (2, 4)]],
            buffers,
            torch.zeros(1),
            torch.zeros(1),
            task(gamma=0.5).config.training,
        )

        # Two turns of -2 seen from the first: the second is discounted once.
        self.assertAlmostEqual(advantages[0].item(), -3.0)


class AdvantageScopeTest(unittest.TestCase):
    def test_batch_scope_leaves_the_advantages_alone(self):
        advantages = torch.tensor([[1.0, 100.0], [3.0, 300.0]])

        scoped = task().scope_advantages(advantages)

        torch.testing.assert_close(scoped, advantages)

    def test_env_scope_makes_two_circuits_comparable(self):
        """Columns are environments, and with a pool they hold different circuits.

        The right hand circuit is a hundred times more expensive; without a per
        column scope its advantages dominate every minibatch it appears in.
        """
        advantages = torch.tensor([[1.0, 100.0], [3.0, 300.0]])

        scoped = task(adv_scope="env").scope_advantages(advantages)

        torch.testing.assert_close(scoped[:, 0], scoped[:, 1])
        self.assertAlmostEqual(scoped.mean().item(), 0.0, places=6)


class BestEpisodeBufferTest(unittest.TestCase):
    def episode(self, buffer, cost, group):
        buffer.add_episode(cost, torch.zeros(2, 3), torch.zeros(2), group)

    def test_a_cheap_circuit_does_not_evict_an_expensive_one(self):
        """Costs are only comparable within a circuit.

        A pool of circuits spans a wide range of costs, so a single ranking would
        keep only the cheapest circuit and the imitation term would never see the
        rest of the pool again.
        """
        buffer = BestEpisodeBuffer(capacity=1, num_envs=1)

        self.episode(buffer, 300.0, group=0)
        self.episode(buffer, 10.0, group=1)

        self.assertEqual(sorted(cost for cost, _, _ in buffer.entries), [10.0, 300.0])

    def test_within_a_circuit_the_cheaper_episode_wins(self):
        buffer = BestEpisodeBuffer(capacity=1, num_envs=1)

        self.episode(buffer, 300.0, group=0)
        self.episode(buffer, 290.0, group=0)

        self.assertEqual([cost for cost, _, _ in buffer.entries], [290.0])


if __name__ == "__main__":
    unittest.main()
