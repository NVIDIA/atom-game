# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from abc import ABC, abstractmethod
import copy
from enum import Enum
from queue import Queue
import random
from typing import Any

import gymnasium as gym
import numpy as np

from qds.qcgame.device import Board, Point
from qds.qcgame.estimator import CostEstimator
from qds.qcgame.log import ActivityLog


class Event(Enum):
    GAME_START = 0
    RECONFIGURATION_END = 1


class AtomGameAI(ABC):
    def __init__(self, env) -> None:
        self.env = env
        self.queue = Queue()

    def register(self, event: Event):
        self.queue.put(event)

    def handle(self):
        reward = 0
        while not self.queue.empty():
            gain, done = self.do_handle(self.queue.get())
            reward += gain
            if done:
                return reward, done
        return reward, False

    @abstractmethod
    def do_handle(self, event: Event):
        raise NotImplementedError


class Turn:
    TURN_END = -1

    def __init__(self, players, check_point, reader, feature_count, **kwargs):
        self.reader = reader
        self.feature_count = feature_count
        self.moves = []
        self.moves_raw = []
        self.next_task = kwargs.get("next_task")
        self._player_index = 0
        self._players = players
        self._check_point = check_point

    @property
    def players(self):
        return self._players

    def num_players(self):
        return len(self._players)

    def current_player(self):
        return self._players[self._player_index]

    def next_player(self):
        self._player_index += 1
        if self._player_index == self._check_point and not self.moves:
            return self.TURN_END
        if self._player_index == len(self._players):
            return self.TURN_END
        return self._players[self._player_index]

    def tasks(self):
        if (
            self.next_task is not None
            and self.next_task.has_twoq_gates(self.current_player())
            and len(self.next_task.gates) > 1
        ):
            tasks = [self.next_task]
            tasks.extend(
                self.reader.see_next(
                    self.feature_count - 1, player=self.current_player()
                )
            )
        else:
            tasks = self.reader.see_next(
                self.feature_count, player=self.current_player()
            )
        return [task.twoq_gates(tuple=True) for task in tasks]


class TurnBuilder:
    def __init__(self, reader, feature_count):
        self.reader = reader
        self.feature_count = feature_count

    def reconfiguration(self, tasks):
        players = []
        included = set()
        check_point = None
        for index, task in enumerate(tasks):
            for gate in task.gates:
                for atom in (gate.c_index, gate.t_index):
                    if atom not in included:
                        players.append(atom)
                        included.add(atom)
            if index == 0:
                check_point = len(players)
        return Turn(
            players,
            check_point,
            self.reader,
            self.feature_count,
            next_task=tasks[0],
        )


class ReconfigurationOnlyAI(AtomGameAI):
    def do_handle(self, event: Event):
        if event == Event.GAME_START:
            return 0, self.prepare_next_task()
        if event == Event.RECONFIGURATION_END:
            self.env.log.add_reconfiguration_moves(self.env.turn.moves)
            reward = self.env.configuration_reward()
            return reward, self.prepare_next_task()
        raise ValueError(f"Unsupported event: {event}")

    def prepare_next_task(self):
        if self.env.reader.is_done():
            return True
        tasks = self.env.reader.see_next(self.env.player_layer)
        self.env.next_task = self.env.reader.read()
        if self.env.estimator.is_parallel_executable(
            self.env.board, self.env.next_task.twoq_gates(tuple=True)
        ):
            self.env.turn = None
            self.register(Event.RECONFIGURATION_END)
        self.env.turn = self.env.turn_builder.reconfiguration(tasks)
        return False


class BaseCircuitEnv(gym.Env, ABC):
    metadata = {"render_modes": []}

    def __init__(self, reader, config):
        super().__init__()
        self.config = config
        self.reader = reader
        self.row = config.board.row
        self.col = config.board.col
        self.player_layer = config.env.player_layer
        self.feature_count = config.env.feature_count
        self.robot = ReconfigurationOnlyAI(self)
        self.estimator = None
        self.board = None
        self.turn = None
        self.next_task = None
        self.state = None
        self.log = None
        self.turn_builder = None

    def step(self, action):
        player_reward, reconfiguration_end = self._reconfiguration_step(action)
        if reconfiguration_end:
            self.robot.register(Event.RECONFIGURATION_END)
        robot_reward, done = self.robot.handle()
        reward = player_reward + robot_reward
        if done:
            return self.state, reward, done, False, {}
        self.state = self._build_state()
        # Every atom in a reconfiguration is asked in turn but only the last one
        # earns the reward, so learners that treat a whole turn as one decision
        # need to know where the boundary is.
        return (
            self.state,
            reward,
            done,
            False,
            {
                "log": self.log,
                "turn_end": reconfiguration_end,
                # Costs from different circuits are not comparable, so anything
                # that ranks episodes has to know which circuit produced one.
                "circuit": getattr(self.reader, "current_reader_indx", 0),
            },
        )

    # The config and the spaces are fixed once a game is under way, so a copy
    # shares them and only pays for the state that moves. The reader is not among
    # them: it carries how far into the circuit the game has read.
    SHARED = ("config", "action_space", "observation_space")

    def __deepcopy__(self, memo):
        """A detached game, so a planner can play a move out and take it back.

        The robot is rebuilt rather than copied: it holds an event queue with a
        lock, which cannot be copied at all, and a reference back to the game it
        drives, which has to point at the copy.
        """
        twin = self.__class__.__new__(self.__class__)
        memo[id(self)] = twin
        for name, value in self.__dict__.items():
            if name == "robot":
                continue
            copied = value if name in self.SHARED else copy.deepcopy(value, memo)
            setattr(twin, name, copied)
        twin.robot = ReconfigurationOnlyAI(twin)
        return twin

    def _reconfiguration_step(self, action):
        player = self.turn.current_player()
        move = self.board.get_atom_move(player, action)
        if move is not None:
            self.log.reconfig_count += 1
            self.turn.moves.append(move)
            self.turn.moves_raw.append((player, self.board.atom_map[player], action))
        self.board = self.board.transform(player, action)
        return 0, self.turn.next_player() == Turn.TURN_END

    def configuration_reward(self):
        gate_detail = self.estimator.gate_cost_detail(
            self.board, self.next_task.twoq_gates(tuple=True)
        )
        duration = gate_detail["total_time"]
        n_touches = gate_detail["total_touches"]
        cost = gate_detail["cost"]
        reward = (
            self.estimator.gate_cost(
                self.original_board, self.next_task.twoq_gates(tuple=True)
            )
            - gate_detail["cost"]
        )
        if self.turn is not None:
            reconfiguration_detail = self.estimator.reconfiguration_cost_detail(
                self.turn.moves
            )
            duration += reconfiguration_detail["total_time"]
            n_touches += reconfiguration_detail["total_touches"]
            cost += reconfiguration_detail["cost"]
            reward -= reconfiguration_detail["cost"]
        self.log.add_task(self.next_task, duration, n_touches, reward, cost)
        return reward

    def _build_state(self):
        board = self._board_array(self.board)
        return {
            "board": board,
            "tasks": self._tasks_array(self.turn.tasks()),
            "moves": self._moves_array(self.turn.moves_raw, board.copy()),
            "player": self.turn.current_player(),
            "current_index": self._current_index(),
        }

    @abstractmethod
    def _current_index(self):
        raise NotImplementedError

    @abstractmethod
    def _board_array(self, board):
        raise NotImplementedError

    def _tasks_array(self, tasks):
        size = self.observation_space["board"].shape[0]
        result = np.tile(np.arange(size), (self.feature_count, 1))
        for index, task in enumerate(tasks):
            for control, target in task:
                result[index][control] = target
                result[index][target] = control
        return result

    @staticmethod
    def _moves_array(moves_raw, board):
        result = np.stack([board.copy(), board.copy()])
        for player, before, _ in moves_raw:
            result[0][player] = before
        return result.T.astype(np.int32)

    def render(self):
        pass

    def close(self):
        pass


class FixedCircuitEnv(BaseCircuitEnv):
    def __init__(self, board, reader, config):
        super().__init__(reader, config)
        n_atoms = len(board.atom_map)
        self.action_space = gym.spaces.Discrete(board.size())
        self.observation_space = gym.spaces.Dict(
            {
                "board": gym.spaces.Box(
                    0, board.size() - 1, shape=(n_atoms,), dtype=np.int64
                ),
                "tasks": gym.spaces.Box(
                    0,
                    n_atoms - 1,
                    shape=(self.feature_count, n_atoms),
                    dtype=np.int64,
                ),
                "moves": gym.spaces.Box(
                    0,
                    board.size() - 1,
                    shape=(n_atoms, 2),
                    dtype=np.int32,
                ),
                "player": gym.spaces.Discrete(n_atoms),
                "current_index": gym.spaces.Discrete(reader.n_components() + 1),
            }
        )
        self.original_board = board
        self.estimator = CostEstimator(N=n_atoms, **config.device)
        self.turn_builder = TurnBuilder(reader, self.feature_count)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ):
        super().reset(seed=seed)
        self.reader.reset()
        self.log = ActivityLog(self.original_board)
        self.turn = None
        self.next_task = None
        self.board = self.original_board
        self.robot.register(Event.GAME_START)
        reward, _ = self.robot.handle()
        self.state = self._build_state()
        return self.state, {"initial_reward": reward}

    def _current_index(self):
        return self.reader.current_index

    def _board_array(self, board):
        result = np.zeros(len(board.atom_map), dtype=np.int32)
        for atom, position in board.atom_map.items():
            result[atom] = position
        return result


class TransferableEnv(BaseCircuitEnv):
    def __init__(self, reader, config, board=None):
        super().__init__(reader, config)
        self.max_atoms = config.benchmark.max_atoms
        board_size = self.row * self.col
        self.action_space = gym.spaces.Discrete(board_size)
        self.observation_space = gym.spaces.Dict(
            {
                "board": gym.spaces.Box(
                    0, board_size, shape=(self.max_atoms,), dtype=np.int64
                ),
                "tasks": gym.spaces.Box(
                    0,
                    self.max_atoms - 1,
                    shape=(self.feature_count, self.max_atoms),
                    dtype=np.int64,
                ),
                "moves": gym.spaces.Box(
                    0,
                    board_size,
                    shape=(self.max_atoms, 2),
                    dtype=np.int32,
                ),
                "player": gym.spaces.Discrete(self.max_atoms),
                "current_index": gym.spaces.Discrete(
                    reader.max_n_components() + 1
                ),
            }
        )
        # Kept apart from original_board, which reset rewrites with the board the
        # current episode started from: a board handed in here has to survive
        # every reset, and it only fits circuits with as many atoms as it holds.
        self.fixed_board = board

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ):
        super().reset(seed=seed)
        self.reader.reset()
        self.estimator = CostEstimator(
            N=self.reader.num_qubits(), **self.config.device
        )
        self.turn_builder = TurnBuilder(self.reader, self.feature_count)
        self.turn = None
        self.next_task = None
        if self.fixed_board is not None:
            self.board = self.fixed_board.copy()
        else:
            self.board = self._random_board()
        self.original_board = self.board.copy()
        self.robot.register(Event.GAME_START)
        self.log = ActivityLog(self.board)
        reward, _ = self.robot.handle()
        self.state = self._build_state()
        return self.state, {"initial_reward": reward}

    def _random_board(self):
        points = [
            Point(column, row)
            for column in range(self.col)
            for row in range(self.row)
        ]
        available = list(range(len(points)))
        atom_map = {}
        for atom in range(self.reader.num_qubits()):
            position = random.choice(available)
            available.remove(position)
            atom_map[atom] = position
        return Board(points, atom_map)

    def _current_index(self):
        return self.reader.current_index()

    def _board_array(self, board):
        result = np.full(self.max_atoms, board.size(), dtype=np.int32)
        for atom, position in board.atom_map.items():
            result[atom] = position
        return result
