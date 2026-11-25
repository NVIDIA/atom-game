# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import unittest
import sys

import numpy as np

sys.path.append(".")
from qds.qcgame.cost import (
    reconfiguation_conflicts,
    reconfigure_layout,
    gate_forward,
    estimate_cost_batch,
    RewardModel,
    Qubit,
)


class TestCost(unittest.TestCase):
    def test_reconfiguration_conflicts(self):
        # Test 1: no conflicts
        self.assertEqual(
            1,
            reconfiguation_conflicts(
                [((0, 0), (0, 1)), ((0, 2), (0, 3)), ((0, 4), (0, 5))]
            ),
        )
        # Test 2: one conflict
        self.assertEqual(
            2,
            reconfiguation_conflicts(
                [((0, 0), (0, 1)), ((0, 1), (0, 0)), ((0, 4), (0, 5))]
            ),
        )
        # Test 3: Many-to-one conflict
        self.assertEqual(
            4,
            reconfiguation_conflicts(
                [((0, 0), (0, 1)), ((0, 1), (0, 1)), ((0, 2), (0, 1))]
            ),
        )
        # Test 4: One-to-many conflict
        self.assertEqual(
            4,
            reconfiguation_conflicts(
                [((0, 2), (0, 1)), ((0, 2), (0, 2)), ((0, 2), (0, 3))]
            ),
        )

    def test_gate_forward_batch(self):
        B, N = 2, 1  # バッチ5件、各サンプルに4つのactive_sites
        model = RewardModel(N=10)
        r = gate_forward(
            model,
            state={Qubit(0): (0, 0), Qubit(1): (0, 1)},
            gates={(Qubit(0), Qubit(1))},
        )
        
        r2 = estimate_cost_batch(
            model, np.array([[[[0, 0], [0, 1]]]]), np.array([1]), is_gate=True
        )
        self.assertEqual(r["cost"], r2["cost"][0])

    def test_reconfigure_layout_batch(self):
        B, N = 200, 4  # バッチ5件、各サンプルに4つのactive_sites
        np.random.seed(0)
        model = RewardModel(N=10)
        active_sites = np.random.randint(0, 10, size=(B, N, 2, 2))
        result = estimate_cost_batch(model, active_sites, np.full(B, N))
        # (B, N, 2, 2): 適当に 0~9 の座標を生成
        from qds.qcgame.cost import do_reconfigure_layout

        values = []
        for site in active_sites:
            values.append(do_reconfigure_layout(model, site))
        for j in range(B):
            self.assertEqual(result["cost"][j], values[j]["cost"])

    def test_reconfigure_layout(self):
        model = RewardModel(N=10)
        # Test 1: Nothing happens
        self.assertEqual(
            0.0,
            reconfigure_layout(
                model, start_state={Qubit(0): (0, 0)}, end_state={Qubit(0): (0, 0)}
            )["cost"],
        )
        # Test 2: single qubit moves, another does not
        self.assertEqual(
            14.24213562373095,
            reconfigure_layout(
                model,
                start_state={Qubit(0): (0, 0), Qubit(1): (0, 1)},
                end_state={Qubit(0): (0, 2), Qubit(1): (0, 1)},
            )["cost"],
        )
        # Test 3: Unconflicted reconfiguration
        self.assertEqual(
            10.2,
            reconfigure_layout(
                model,
                start_state={Qubit(0): (0, 0), Qubit(1): (0, 1)},
                end_state={Qubit(0): (1, 1), Qubit(1): (1, 2)},
            )["cost"],
        )
        # Test 4: Slightly conflicted reconfiguration (move to different rows)
        self.assertAlmostEqual(
            24.283346494528416,
            reconfigure_layout(
                model,
                start_state={Qubit(0): (0, 0), Qubit(1): (0, 1)},
                end_state={Qubit(0): (1, 1), Qubit(1): (2, 2)},
            )["cost"],
            delta=1e-5,
        )
        model = RewardModel(N=20)
        # Test 5: Invert order 16
        self.assertAlmostEqual(
            472.9365085206941,
            reconfigure_layout(
                model,
                start_state={Qubit(i): (i, 0) for i in range(16)},
                end_state={Qubit(i): (0, 16 - i) for i in range(16)},
            )["cost"],
            delta=1e-5,
        )
        model = RewardModel(N=80)
        # Test 6: Invert order 64
        self.assertAlmostEqual(
            5565.0945301336,
            reconfigure_layout(
                model,
                start_state={Qubit(i): (i, 0) for i in range(64)},
                end_state={Qubit(i): (0, 64 - i) for i in range(64)},
            )["cost"],
            delta=1e-5,
        )
        # Test 7: Two qubits move, another gets caught in the starting cross
        self.assertAlmostEqual(
            238.060460550383,
            reconfigure_layout(
                model,
                start_state={Qubit(0): (0, 0), Qubit(1): (1, 1), Qubit(2): (0, 1)},
                end_state={Qubit(0): (2, 2), Qubit(1): (3, 3), Qubit(2): (0, 1)},
            )["cost"],
            delta=1e-5,
        )
        # Test 8: Three qubits move on a grid
        self.assertAlmostEqual(
            113.4370849898476,
            reconfigure_layout(
                model,
                start_state={Qubit(0): (0, 0), Qubit(1): (1, 1), Qubit(2): (0, 1)},
                end_state={Qubit(0): (2, 2), Qubit(1): (3, 3), Qubit(2): (2, 3)},
            )["cost"],
            delta=1e-5,
        )

    def test_gate_forward(self):
        model = RewardModel(N=10)
        # Test 1: Single unconflicted gate
        self.assertAlmostEqual(
            110.1,
            gate_forward(
                model,
                state={Qubit(0): (0, 0), Qubit(1): (0, 1)},
                gates={(Qubit(0), Qubit(1))},
            )["cost"],
        )
        # Test 2: Single partially unconflicted gate
        self.assertAlmostEqual(
            110.1,
            gate_forward(
                model,
                state={Qubit(0): (0, 0), Qubit(1): (1, 1)},
                gates={(Qubit(0), Qubit(1))},
            )["cost"],
        )

        # Test 2: Two conflicted gates on the same row
        self.assertAlmostEqual(
            134.20157304731092,
            gate_forward(
                model,
                state={
                    Qubit(0): (0, 0),
                    Qubit(1): (4, 0),
                    Qubit(2): (1, 0),
                    Qubit(3): (2, 0),
                },
                gates={(Qubit(0), Qubit(1)), (Qubit(2), Qubit(3))},
            )["cost"],
            delta=0.001,
        )
        model = RewardModel(N=20)
        # Test 3: no conflicts
        self.assertAlmostEqual(
            220.8,
            gate_forward(
                model,
                state={Qubit(i): (i, 0) for i in range(16)},
                gates={(Qubit(2 * i), Qubit(2 * i + 1)) for i in range(8)},
            )["cost"],
        )
        # Test 4: complete conflicts
        self.assertAlmostEqual(
            541.783117013963,
            gate_forward(
                model,
                state={Qubit(i): (i, 0) for i in range(16)},
                gates={(Qubit(i), Qubit(15 - i)) for i in range(8)},
            )["cost"],
            delta=0.001,
        )


if __name__ == "__main__":
    unittest.main()
