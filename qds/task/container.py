# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import os
import pickle
import random
from pathlib import Path

import torch
import torch.distributed as dist
from ml_collections import ConfigDict
from pytorch_lightning.loggers import CSVLogger, WandbLogger

from qds.circuit import BatchCircuitReader
from qds.circuit.generator import (
    FHGenerator,
    MQTGenerator,
    SparseRandomPauliGenerator,
    SyntheticQNNGenerator,
)
from qds.circuit.reader import MultiBatchCircuitReader
from qds.qcgame.device import Board, Point
from qds.qcgame.environment import FixedCircuitEnv, TransferableEnv
from qds.qcgame.policy import FixedPolicy, TransferablePolicy


class Container:
    """Build the active fixed-circuit or transferable PPO stack."""

    MODES = {"fixed", "transferable"}

    def __init__(self, mode: str):
        if mode not in self.MODES:
            raise ValueError(f"Unsupported mode {mode!r}; expected one of {self.MODES}")
        self.mode = mode
        self.reader = None
        self.circuit = None
        self.circuits = None
        self.board = None

    def get_logger(self, config: ConfigDict):
        if config.get("logger") == "wandb":
            return WandbLogger(
                config=config.to_dict(),
                project=config.project.name,
                name=config.run_name,
            )
        return CSVLogger(
            config.logger_path,
            name=config.project.name,
            version=config.run_name,
            flush_logs_every_n_steps=10,
        )

    def get_env(self, config: ConfigDict):
        seed = config.seed
        if self.mode == "transferable" and config.task == "test":
            seed = 31 * (seed + 1) + 17
        random.seed(seed)
        torch.manual_seed(seed)

        if self.mode == "fixed":
            return self._get_fixed_env(config)
        return self._get_transferable_env(config)

    def get_policy(self, config: ConfigDict):
        if self.mode == "fixed":
            return FixedPolicy(config)
        return TransferablePolicy(config)

    def prepare(self, config: ConfigDict):
        if self.mode != "transferable":
            raise ValueError("Only transferable mode has a circuit cache")
        return self._load_transferable_circuits(config)

    def _get_fixed_env(self, config: ConfigDict):
        circuit = self._get_fixed_circuit(config)
        reader = self._get_fixed_reader(config, circuit)
        config.benchmark.n_atoms = reader.num_qubits()
        if config.benchmark.truncate > 0:
            reader.truncate(config.benchmark.truncate)
        board = self._get_fixed_board(config, circuit, reader)
        return FixedCircuitEnv(board, reader, config)

    def _get_transferable_env(self, config: ConfigDict):
        circuits = self._get_transferable_circuits(config)
        # Held out circuits are visited in order so that a cost can be compared
        # circuit by circuit against a heuristic run over the same file, while
        # training keeps drawing at random.
        strategy = "sequential" if config.task == "test" else "random"
        reader = MultiBatchCircuitReader.from_jsons(circuits, strategy)
        if len(circuits) == 1:
            config.benchmark.n_atoms = reader.num_qubits()
            config.n_components = reader.n_components()
        # A shared board is only defined when every episode places the same
        # atoms, so a held out set of differing sizes draws its board per reset.
        board = (
            self._get_transferable_board(config)
            if config.task == "test" and len(circuits) == 1
            else None
        )
        return TransferableEnv(reader, config, board)

    def _get_fixed_reader(self, config: ConfigDict, circuit):
        if self.reader is None:
            if isinstance(circuit, list):
                self.reader = BatchCircuitReader(circuit)
            else:
                self.reader = BatchCircuitReader.from_circuit(circuit)
            return self.reader
        return BatchCircuitReader(self.reader.components)

    def _get_fixed_circuit(self, config: ConfigDict):
        # Ranks only need to agree once; later envs reuse the cache so that
        # environment construction stays free of collectives and of CUDA.
        if self.circuit is not None:
            return self.circuit
        if self._rank() == 0:
            circuit = self._generate_fixed_circuit(config)
            serialized = pickle.dumps(circuit)
        else:
            serialized = None
        return self._broadcast_object(serialized, config.system.device)

    def _generate_fixed_circuit(self, config: ConfigDict):
        if self.circuit is not None:
            return self.circuit
        cache = config.get("circuit_cache")
        if cache and Path(cache).exists():
            self.circuit = pickle.loads(Path(cache).read_bytes())
            return self.circuit
        if config.benchmark.problem == "hubbard":
            generator = FHGenerator(
                nqubit=config.benchmark.n_atoms, **config.benchmark.kwargs
            )
            time = config.benchmark.kwargs.get("t") or random.random()
            self.circuit = generator.generate(time, seed=config.seed)
        elif config.benchmark.problem == "qnn-lite":
            generator = SyntheticQNNGenerator(
                nqubit=config.benchmark.n_atoms,
                layers=config.benchmark.layers,
                gates_per_layer=config.benchmark.gates_per_layer,
            )
            self.circuit = generator.generate_components(seed=config.seed)
        else:
            generator = MQTGenerator(config.benchmark.n_atoms)
            self.circuit = generator.generate(
                config.seed, benchmark=config.benchmark.problem
            )
        if cache:
            path = Path(cache)
            path.parent.mkdir(parents=True, exist_ok=True)
            # Replace is atomic, so a reader either sees the previous copy or this
            # one and never a half written file.
            partial = path.with_suffix(f".{os.getpid()}")
            partial.write_bytes(pickle.dumps(self.circuit))
            partial.replace(path)
        return self.circuit

    def _get_fixed_board(self, config: ConfigDict, circuit, reader):
        if self.board is not None:
            return self.board.copy()
        points = self._board_points(config)
        if config.board.init != "random":
            raise ValueError(f"Unsupported board initializer: {config.board.init}")
        atom_map = self._random_atom_map(len(points), config.benchmark.n_atoms)
        self.board = Board(points, atom_map)
        config.atom_map = json.dumps(self.board.atom_map)
        config.n_components = reader.n_components()
        return self.board

    def _get_transferable_circuits(self, config: ConfigDict):
        if self._rank() == 0:
            circuits = self._load_transferable_circuits(config)
            serialized = pickle.dumps(circuits)
        else:
            serialized = None
        return self._broadcast_object(serialized, config.system.device)

    def _load_transferable_circuits(self, config: ConfigDict):
        if self.circuits is not None:
            return self.circuits
        path = Path(
            config.test_circuit_path
            if config.task == "test"
            else config.circuit_path
        )
        if not path.exists():
            self._generate_transferable_cache(config, path)
        self.circuits = path.read_text().splitlines(keepends=True)
        return self.circuits

    def _generate_transferable_cache(self, config: ConfigDict, path: Path):
        generator = SparseRandomPauliGenerator(
            min_nqubit=config.benchmark.min_atoms,
            max_nqubit=config.benchmark.max_atoms,
            n_terms=config.benchmark.n_terms,
            k_local=config.benchmark.k_local,
        )
        # One held out circuit says nothing about transfer, so the evaluation set
        # is sized on its own key and the default stays what it was.
        count = (
            config.benchmark.get("test_count", 1)
            if config.task == "test"
            else config.benchmark.count
        )
        readers = [
            BatchCircuitReader.from_circuit(generator.generate())
            for _ in range(count)
        ]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            "".join(f"{line}\n" for line in MultiBatchCircuitReader(readers).to_jsons())
        )

    def _get_transferable_board(self, config: ConfigDict):
        if self.board is None:
            points = self._board_points(config)
            atom_map = self._random_atom_map(
                len(points), config.benchmark.n_atoms
            )
            self.board = Board(points, atom_map)
        return self.board

    @staticmethod
    def _board_points(config: ConfigDict):
        return [
            Point(column, row)
            for column in range(config.board.col)
            for row in range(config.board.row)
        ]

    @staticmethod
    def _random_atom_map(board_size: int, n_atoms: int):
        available = list(range(board_size))
        atom_map = {}
        for atom in range(n_atoms):
            position = random.choice(available)
            available.remove(position)
            atom_map[atom] = position
        return atom_map

    @staticmethod
    def _broadcast_object(serialized, device):
        if not dist.is_available() or not dist.is_initialized():
            return pickle.loads(serialized)
        if dist.get_rank() == 0:
            size = len(serialized)
        else:
            size = 0
        size_tensor = torch.tensor([size], dtype=torch.int, device=device)
        dist.broadcast(size_tensor, src=0)
        if dist.get_rank() == 0:
            buffer = torch.tensor(list(serialized), dtype=torch.uint8, device=device)
        else:
            buffer = torch.empty(size_tensor.item(), dtype=torch.uint8, device=device)
        dist.broadcast(buffer, src=0)
        if dist.get_rank() != 0:
            serialized = buffer.cpu().numpy().tobytes()
        return pickle.loads(serialized)

    @staticmethod
    def _rank():
        if dist.is_available() and dist.is_initialized():
            return dist.get_rank()
        return 0
