# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import sys

sys.path.append(".")
from qds.qcgame.estimator import *
from qds.qcgame.device import Board, Point
import unittest


class TestRewardEstimator(unittest.TestCase):
    def test_is_parallel_executable(self):
        estimator = CostEstimator(10)
        points = []
        for i in range(5):
            for j in range(5):
                points.append(Point(i, j))
        self.assertTrue(
            estimator.is_parallel_executable(
                Board(points, atom_map={0: 1, 1: 7, 2: 3, 3: 9}), [(0, 1), (2, 3)]
            )
        )
        self.assertTrue(
            estimator.is_parallel_executable(
                Board(points, atom_map={0: 1, 1: 7, 2: 3, 3: 9, 4: 10, 5: 11}),
                [(0, 2), (1, 3), (4, 5)],
            )
        )
        self.assertFalse(
            estimator.is_parallel_executable(
                Board(points, atom_map={0: 1, 1: 7, 2: 3, 3: 9, 4: 10, 5: 11}),
                [(0, 1), (2, 4), (3, 5)],
            )
        )

    def test_gate_gain(self):
        points = []
        for i in range(5):
            for j in range(5):
                points.append(Point(i, j))
        estimator = CostEstimator(10)
        cost = estimator.gate_cost(
            Board(points, atom_map={0: 1, 1: 7, 2: 3, 3: 9, 4: 10, 5: 11}),
            [(0, 2), (1, 3), (4, 5)],
        )
        cost2 = estimator.gate_cost(
            Board(points, atom_map={0: 1, 1: 7, 2: 3, 3: 9, 4: 10, 5: 11}),
            [(0, 1), (2, 4), (3, 5)],
        )
        print(cost2, cost)
        self.assertGreater(cost2, cost)

    def test_move_gain(self):
        points = []
        for i in range(5):
            for j in range(5):
                points.append(Point(i, j))
        estimator = CostEstimator(10)
        cost = estimator.reconfiguration_cost(
            [
                AtomMove(0, Point(1, 0), Point(2, 1)),
                AtomMove(1, Point(3, 0), Point(4, 1)),
            ]
        )
        cost2 = estimator.reconfiguration_cost(
            [
                AtomMove(0, Point(1, 0), Point(2, 1)),
                AtomMove(1, Point(3, 0), Point(1, 2)),
            ]
        )
        self.assertGreater(cost2, cost)


if __name__ == "__main__":
    unittest.main()
