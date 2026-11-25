# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest import TestCase
from qds.qcgame.device import Board, Point
import numpy as np


class TestBoard(TestCase):
    def test_transform(self):
        points = []
        for i in range(3):
            for j in range(4):
                points.append(Point(i, j))
        atom_map = {0: 11, 1: 4, 2: 3, 3: 5, 4: 6, 5: 7, 6: 0}
        original_board = Board(points, atom_map)
        board = original_board.transform(a_id=4, grid_id=10)
        np.testing.assert_array_equal(original_board.vector(), [6, -1, -1, 2, 1, 3, 4, 5, -1, -1, -1, 0])
        np.testing.assert_array_equal(board.vector(), [6, -1, -1, 2, 1, 3, -1, 5, -1, -1, 4, 0])

        board2 = original_board.transform(a_id=4, grid_id=10)
        np.testing.assert_array_equal(board2.vector(), [6, -1, -1, 2, 1, 3, -1, 5, -1, -1, 4, 0])
        self.assertRaises(AttributeError, original_board.transform, a_id=4, grid_id=0)
