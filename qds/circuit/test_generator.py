# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import unittest

from qds.circuit.generator import SyntheticQNNGenerator
from qds.circuit.reader import BatchCircuitReader


class TestSyntheticQNNGenerator(unittest.TestCase):
    def test_builds_parallel_random_matching_layers(self):
        generator = SyntheticQNNGenerator(
            nqubit=6,
            layers=5,
            gates_per_layer=2,
        )
        reader = BatchCircuitReader(generator.generate_components(seed=7))

        self.assertEqual(reader.n_components(), 5)
        for component in reader.components:
            gates = component.twoq_gates(tuple=True)
            self.assertEqual(len(gates), 2)
            self.assertEqual(
                len({qubit for gate in gates for qubit in gate}),
                4,
            )

    def test_seed_is_reproducible(self):
        generator = SyntheticQNNGenerator(nqubit=8, layers=4)

        first = generator.generate(seed=3).twoq_gates(tuple=True)
        second = generator.generate(seed=3).twoq_gates(tuple=True)
        other = generator.generate(seed=4).twoq_gates(tuple=True)

        self.assertEqual(first, second)
        self.assertNotEqual(first, other)


if __name__ == "__main__":
    unittest.main()
