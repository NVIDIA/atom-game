# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest import TestCase
from qds.circuit.circuit import init_circuit, decompose_circuit


class Test(TestCase):
    def test_decompose_circuit(self):
        qc = init_circuit(4, "default")
        qc.x(0)
        qc.cnot(0, 1)
        qc.y(0)
        qc.cnot(0, 1)
        qc.y(0)

        qc.cnot(0, 1)
        qc.y(0)
        qc.cnot(0, 1)
        qc.y(0)

        qc.cnot(2, 3)
        qc.z(0)
        qc.cnot(3, 2)
        qc.y(0)

        circuits = decompose_circuit(qc, 3)

        self.assertEqual(str(circuits[0]), "x(0) cx(0,1) y(0) cx(0,1)")
        self.assertEqual(str(circuits[1]), "y(0) cx(0,1) y(0) cx(0,1)")
        self.assertEqual(str(circuits[2]), "y(0) cx(2,3) z(0) cx(3,2) y(0)")
