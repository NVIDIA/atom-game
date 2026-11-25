# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import sys
sys.path.append(".")
from unittest import TestCase
import unittest
from qds.circuit.circuit import GateSequence
from qds.circuit.reader import BatchCircuitReader


class TestBatchCircuitReader(TestCase):
    def test_save(self):
        g1 = GateSequence(5)
        g2 = GateSequence(5)
        g3 = GateSequence(5)
        g4 = GateSequence(5)

        g1.cnot(3, 4)
        g2.cnot(3, 1)
        g2.cnot(2, 4)
        g3.cnot(1, 2)
        g4.cnot(3, 2)

        circuits = [g1, g2, g3, g4]
        reader = BatchCircuitReader(circuits)
        self.assertEqual(reader.to_json(), '{"circuit": [[[3, 4]], [[3, 1], [2, 4]], [[1, 2]], [[3, 2]]], "nqubit": 5}')
        reader2 = BatchCircuitReader.from_json(reader.to_json())
        self.assertEqual(reader2.to_json(), '{"circuit": [[[3, 4]], [[3, 1], [2, 4]], [[1, 2]], [[3, 2]]], "nqubit": 5}')
        
    
    def test_reset(self):
        circuits = [GateSequence(10) for _ in range(0, 50)]
        reader = BatchCircuitReader(circuits)
        reader.reset(10, 30)

        count = 0
        while True:
            circuit = reader.read()
            if circuit is None:
                break
            count += 1
        self.assertEqual(count, 20)


if __name__ == "__main__":
    unittest.main()