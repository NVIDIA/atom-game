# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import unittest
from pathlib import Path
import re

import torch
from ml_collections import ConfigDict
from omegaconf import OmegaConf

from qds.qcgame.policy import (
    FixedPolicy,
    GateEmbedding,
    GateTransformer,
    MergeLayer,
    MoveEmbedding,
    MoveTransformer,
    PairGeometry,
    TransferablePolicy,
)
from qds.task.container import Container


def policy_config(mode):
    return ConfigDict(
        {
            "mode": mode,
            "n_components": 5,
            "benchmark": {"n_atoms": 4, "max_atoms": 6},
            "board": {"row": 2, "col": 4},
            "env": {"feature_count": 2},
            "policy": {
                "move_embedding_dim": 10,
                "gate_embedding_dim": 10,
                "n_heads": 2,
                "move_layers": 1,
                "gate_layers": 1,
                "move_hidden_dim": 20,
                "gate_hidden_dim": 20,
                "mlp_depth": 1,
                "p_hsize": 4,
                "player_embedding_dim": 4,
                "time_embedding_dim": 4,
            },
        }
    )


class PolicyComponentTest(unittest.TestCase):
    def test_components_support_both_modes(self):
        for mode, atom_count in (("fixed", 4), ("transferable", 6)):
            with self.subTest(mode=mode):
                config = policy_config(mode)
                board = torch.tensor(
                    [[0, 5, 1, 3] + ([8, 8] if mode == "transferable" else [])]
                )
                tasks = torch.tensor(
                    [
                        [
                            [0, 1, 3, 2]
                            + ([4, 5] if mode == "transferable" else []),
                            [2, 3, 0, 1]
                            + ([4, 5] if mode == "transferable" else []),
                        ]
                    ]
                )
                moves = torch.stack([board, board], dim=-1)
                player = torch.tensor([2])
                player_position = torch.tensor([1])

                move_embedding, _ = MoveEmbedding(config)(
                    moves, player_position
                )
                gate_embedding, _ = GateEmbedding(config)(board, tasks, player)
                self.assertEqual(
                    move_embedding.shape, (1, atom_count + 8, 10)
                )
                self.assertEqual(
                    gate_embedding.shape, (2, atom_count + 8, 10)
                )
                self.assertEqual(
                    MoveTransformer(config)(moves, player_position).shape,
                    (1, 8, 10),
                )
                gate_out = GateTransformer(config)(board, tasks, player)
                self.assertEqual(gate_out.shape, (1, 2, 8, 10))
                self.assertEqual(
                    MergeLayer(config)(
                        gate_out, torch.rand(1, 8, 10)
                    ).shape,
                    (1, 8, 10),
                )

    def test_policy_state_dict_interfaces_remain_distinct(self):
        fixed = FixedPolicy(policy_config("fixed"))
        transferable = TransferablePolicy(policy_config("transferable"))
        self.assertIn("global_block.fc.2.weight", fixed.state_dict())
        self.assertNotIn("global_block.fc.2.weight", transferable.state_dict())
        self.assertIn("global_critic.2.weight", transferable.state_dict())

    def test_fixed_board_encoding_is_isolated_per_batch(self):
        policy = FixedPolicy(policy_config("fixed"))
        state = {
            "board": torch.tensor([[0, 1, 2, 3], [4, 5, 6, 7]]),
            "tasks": torch.zeros(2, 2, 4),
            "moves": torch.zeros(2, 4, 2),
            "player": torch.tensor([[1, 0, 0, 0], [1, 0, 0, 0]]),
            "current_index": torch.tensor(
                [[1, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0]]
            ),
        }
        grids = policy.build_obs(state)["grid"]
        self.assertFalse(torch.equal(grids[0], grids[1]))

    def test_existing_transferable_checkpoint_loads_strictly(self):
        model_dir = Path(
            "outputs/neutral-atom-final-transferable/v1/model"
        )
        checkpoints = list(model_dir.glob("model_iter_*.pt"))
        if not checkpoints:
            self.skipTest("Existing v1 checkpoints are not available")
        checkpoint = max(
            checkpoints,
            key=lambda path: int(re.search(r"\d+", path.stem).group()),
        )
        config = policy_config("transferable")
        # Existing v1 checkpoints use the production 40-50 atom dimensions.
        config.benchmark.max_atoms = 50
        config.board.row = 20
        config.board.col = 5
        config.policy.move_embedding_dim = 20
        config.policy.gate_embedding_dim = 20
        config.policy.n_heads = 4
        config.policy.move_layers = 3
        config.policy.gate_layers = 3
        config.policy.move_hidden_dim = 20
        config.policy.gate_hidden_dim = 20
        policy = TransferablePolicy(config)
        state_dict = torch.load(
            checkpoint, map_location="cpu", weights_only=True
        )
        policy.load_state_dict(state_dict, strict=True)


def geometry_config(**policy):
    base = OmegaConf.load("conf/train.yaml")
    base.board.row = 5
    base.board.col = 4
    return OmegaConf.merge(base, {"policy": policy})


class PairGeometryTest(unittest.TestCase):
    def test_features_match_board_coordinates(self):
        """The coordinate decoding has to agree with how the board is laid out.

        ``Container._board_points`` walks columns on the outside, so a position
        index splits as ``column = index // row``. Getting this backwards would
        silently feed the policy a transposed board.
        """
        config = geometry_config(geometry="add")
        points = Container._board_points(config)
        geometry = PairGeometry(config, dim=3)
        size = config.board.row * config.board.col

        pairs = torch.tensor(
            [[a, b] for a in range(size) for b in range(size)], dtype=torch.long
        )
        features = geometry.features(pairs)

        for index, (a, b) in enumerate(pairs.tolist()):
            reference, candidate = points[a], points[b]
            dx = reference.x - candidate.x
            dy = reference.y - candidate.y
            row = features[index]
            self.assertAlmostEqual(row[0].item() * geometry.scale, dx, places=5)
            self.assertAlmostEqual(row[1].item() * geometry.scale, dy, places=5)
            self.assertAlmostEqual(
                row[4].item() * geometry.scale, max(abs(dx), abs(dy)), places=5
            )
            self.assertEqual(row[7].item(), float(dx == 0))
            self.assertEqual(row[8].item(), float(dy == 0))
            self.assertEqual(row[9].item(), 1.0)

    def test_off_adds_no_parameters_and_passes_the_difference_through(self):
        geometry = PairGeometry(geometry_config(), dim=3)
        diff = torch.randn(2, 3)

        self.assertEqual(list(geometry.parameters()), [])
        torch.testing.assert_close(
            geometry(diff, torch.zeros(2, 2, dtype=torch.long)), diff
        )

    def test_only_discards_the_learned_difference(self):
        geometry = PairGeometry(geometry_config(geometry="only"), dim=3)
        pairs = torch.tensor([[0, 7], [3, 3]], dtype=torch.long)
        diff = torch.randn(2, 3)

        torch.testing.assert_close(
            geometry(diff, pairs), geometry(torch.zeros_like(diff), pairs)
        )

    def test_outside_positions_are_flagged_and_zeroed(self):
        config = geometry_config(geometry="add")
        geometry = PairGeometry(config, dim=3)
        size = config.board.row * config.board.col

        features = geometry.features(torch.tensor([[0, size]], dtype=torch.long))

        self.assertEqual(features[0, 9].item(), 0.0)
        self.assertEqual(features[0, :9].abs().sum().item(), 0.0)


if __name__ == "__main__":
    unittest.main()
