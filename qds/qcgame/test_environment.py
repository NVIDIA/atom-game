# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import copy
import random
import unittest

from ml_collections import ConfigDict

from qds.circuit import BatchCircuitReader
from qds.circuit.generator import MQTGenerator
from qds.circuit.reader import MultiBatchCircuitReader
from qds.qcgame.device import Board, Point
from qds.qcgame.environment import FixedCircuitEnv, TransferableEnv
from qds.qcgame.heuristic import BeamSearchPolicy, GreedyPolicy, GuidedBeamPolicy
from qds.qcgame.meter import counting, evaluations, steps


def environment_config(mode):
    return ConfigDict(
        {
            "mode": mode,
            "task": "test",
            "benchmark": {
                "n_atoms": 14,
                "max_atoms": 16,
            },
            "board": {"row": 10, "col": 4},
            "env": {"feature_count": 2, "player_layer": 2},
            "device": {
                "idle_error": 0.02,
                "move_error": 0.002,
                "inter_zone_time": 10,
            },
        }
    )


def make_board(board_class, n_atoms):
    points = [Point(column, row) for column in range(4) for row in range(10)]
    return board_class(points, dict(enumerate(range(n_atoms))))


class EnvironmentTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        circuit = MQTGenerator(14).generate(0, benchmark="qaoa")
        cls.reader = BatchCircuitReader.from_circuit(circuit)

    def test_fixed_reset_and_step(self):
        config = environment_config("fixed")
        reader = BatchCircuitReader(self.reader.components)
        env = FixedCircuitEnv(
            make_board(Board, reader.num_qubits()), reader, config
        )
        observation, _ = env.reset()
        self.assertEqual(observation["board"].shape, (14,))
        self.assert_valid_terminal_observation(env)

    def test_transferable_reset_and_step(self):
        config = environment_config("transferable")
        batch_reader = MultiBatchCircuitReader(
            [BatchCircuitReader(self.reader.components)]
        )
        board = make_board(Board, batch_reader.num_qubits())
        random.seed(0)
        env = TransferableEnv(batch_reader, config, board)
        observation, _ = env.reset()
        self.assertEqual(observation["board"].shape, (16,))
        self.assertEqual(observation["moves"].shape, (16, 2))
        self.assert_valid_terminal_observation(env)

    def test_clone_plays_out_without_touching_the_original(self):
        config = environment_config("fixed")
        reader = BatchCircuitReader(self.reader.components)
        env = FixedCircuitEnv(
            make_board(Board, reader.num_qubits()), reader, config
        )
        env.reset()
        for _ in range(3):
            env.step(env.board.atom_map[env.turn.current_player()])
        before = dict(env.board.atom_map)
        player = env.turn.current_player()
        twin = copy.deepcopy(env)
        # Send the copy somewhere the original never goes, then to the end.
        twin.step(twin.board.empties()[0])
        self.assertNotEqual(twin.board.atom_map, before)
        self.assert_valid_terminal_observation(twin)
        self.assertEqual(env.board.atom_map, before)
        self.assertEqual(env.turn.current_player(), player)
        self.assertFalse(env.reader.is_done())

    def test_beam_search_never_loses_to_greedy(self):
        costs = {}
        for name, policy in (
            ("greedy", GreedyPolicy(None)),
            ("beam", BeamSearchPolicy(None, width=2, branch=2, turns=2)),
        ):
            config = environment_config("fixed")
            reader = BatchCircuitReader(self.reader.components)
            env = FixedCircuitEnv(
                make_board(Board, reader.num_qubits()), reader, config
            )
            env.reset()
            while True:
                _, _, terminated, truncated, _ = env.step(policy.action(env))
                if terminated or truncated:
                    break
            costs[name] = env.log.cost
        self.assertLessEqual(costs["beam"], costs["greedy"])

    def test_guiding_the_beam_costs_the_estimator_nothing_extra(self):
        """A guided beam has to be measurable against the plain one directly.

        The claim being made is that the model changes which sites get a subtree
        and not how many are opened, so a proposer that prefers what the estimator
        prefers has to reproduce the plain search down to the last evaluation.
        One that prefers the opposite then changes the schedule, and its total is
        allowed to move, because a schedule that leaves the atoms better placed
        needs fewer reconfigurations and so fewer nodes: what is fixed per node is
        not fixed per episode.
        """
        spent = {}
        for name, build in (
            ("beam", lambda: BeamSearchPolicy(None, width=2, branch=2, turns=2)),
            (
                "agreeing",
                lambda: GuidedBeamPolicy(
                    None,
                    width=2,
                    branch=2,
                    turns=2,
                    shortlist=8,
                    proposer=lambda state, actions, costs: [-cost for cost in costs],
                ),
            ),
            (
                "opposing",
                lambda: GuidedBeamPolicy(
                    None,
                    width=2,
                    branch=2,
                    turns=2,
                    shortlist=8,
                    proposer=lambda state, actions, costs: list(costs),
                ),
            ),
        ):
            config = environment_config("fixed")
            reader = BatchCircuitReader(self.reader.components)
            env = FixedCircuitEnv(
                make_board(Board, reader.num_qubits()), reader, config
            )
            policy = build()
            env.reset()
            with counting() as counts:
                while True:
                    _, _, terminated, truncated, _ = env.step(policy.action(env))
                    if terminated or truncated:
                        break
            spent[name] = (evaluations(counts), steps(counts), env.log.cost)

        self.assertEqual(spent["beam"], spent["agreeing"])
        self.assertNotEqual(spent["beam"][2], spent["opposing"][2])

    def assert_valid_terminal_observation(self, env):
        for _ in range(10_000):
            action = env.board.atom_map[env.turn.current_player()]
            observation, _, terminated, truncated, _ = env.step(action)
            self.assertFalse(truncated)
            if terminated:
                self.assertTrue(env.observation_space.contains(observation))
                return
        self.fail("Environment did not terminate within 10,000 steps")


if __name__ == "__main__":
    unittest.main()
