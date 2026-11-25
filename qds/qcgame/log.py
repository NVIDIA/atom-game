# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from qds.qcgame.device import Board
from qds.circuit import GateSequence
import json


class ActivityLog:
    def __init__(self, board: Board):
        self.board = board
        self.verbose = False
        self.duration = 0
        self.n_touches = 0
        self.cost = 0
        self.reconfig_count = 0
        self.reward = 0
        points = []
        for p in board.points:
            points.append((p.x, p.y))
        self.lines = [
            json.dumps(
                {"category": "board", "points": points, "atom_map": board.atom_map}
            )
        ]

    def copy(self):
        log = ActivityLog(self.board)
        log.verbose = self.verbose
        log.duration = self.duration
        log.n_touches = self.n_touches
        log.cost = self.cost
        log.reward = self.reward
        log.reconfig_count = self.reconfig_count
        log.lines = self.lines.copy()
        return log

    def add_line(self, line):
        if self.verbose:
            print(line)
        self.lines.append(line)

    def add_reconfiguration_moves(self, moves):
        self.add_line(
            json.dumps({"category": "reconfiguration_move", "move": str(moves)})
        )

    def add_task(self, task: GateSequence, duration, n_touches, reward, cost):
        self.duration += duration
        self.n_touches += n_touches
        self.reward += reward
        self.cost += cost
        gates = []
        for gate in task.twoq_gates():
            gates.append((gate.c_index, gate.t_index))
        self.add_line(
            json.dumps({"category": "gate", "gates": gates, "reward": reward})
        )

    def add_state(self, player, state):
        self.add_line(
            json.dumps({"category": "state", "player": player, "state": state})
        )

    def add_dict(self, dic):
        self.add(json.dumps(dic))

    def add(self, line):
        self.add_line(f"{line}")

    def print(self):
        for line in self.lines:
            print(line)

    def save(self, filename):
        with open(filename, "w") as f:
            for id, line in enumerate(self.lines):
                f.write(f"{id}\t{line}\n")
