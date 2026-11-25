# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import List
from qds.qcgame.cost import (
    gate_forward,
    reconfigure_layout,
    reconfiguation_conflicts,
    estimate_cost_batch,
    RewardModel,
)
import numpy as np
from qds.qcgame.device import Board, AtomMove


class CostEstimator:
    def __init__(self, N, **kwargs):
        self.model = RewardModel(N)
        if "idle_error" in kwargs:
            self.model.idle_error = kwargs["idle_error"]
        if "move_error" in kwargs:
            self.model.move_error = kwargs["move_error"]
        if "inter_zone_time" in kwargs:
            self.model.inter_zone_time = kwargs["inter_zone_time"]

    def is_parallel_executable(self, board: Board, two_qubit_gates):
        active_sites = []  # TODO
        for g in two_qubit_gates:
            p0 = board.get_atom_position(g[0])
            p1 = board.get_atom_position(g[1])
            active_sites.append(((p0.x, p0.y), (p1.x, p1.y)))
        return reconfiguation_conflicts(active_sites) == 1

    def gate_cost(self, board: Board, two_qubit_gates):
        return self.gate_cost_detail(board, two_qubit_gates)["cost"]

    def reconfiguration_cost(self, moves: List[AtomMove]):
        return self.reconfiguration_cost_detail(moves)["cost"]

    def reconfigure_cost_batch(self, active_sites, num_active_sites=None):
        if num_active_sites is None:
            num_active_sites = np.full(active_sites.shape[0], active_sites.shape[1])
        return estimate_cost_batch(self.model, active_sites, num_active_sites)["cost"]

    def gate_cost_batch(self, active_sites, num_active_sites=None):
        if num_active_sites is None:
            num_active_sites = np.full(active_sites.shape[0], active_sites.shape[1])
        return estimate_cost_batch(
            self.model, active_sites, num_active_sites, is_gate=True
        )["cost"]

    def gate_cost_detail(self, board: Board, two_qubit_gates):
        state = {}
        for g in two_qubit_gates:
            p0 = board.get_atom_position(g[0])
            p1 = board.get_atom_position(g[1])
            state[g[0]] = (p0.x, p0.y)
            state[g[1]] = (p1.x, p1.y)
        return gate_forward(self.model, state, two_qubit_gates)

    def reconfiguration_cost_detail(self, moves: List[AtomMove]):
        start_state = {}
        end_state = {}
        import time

        start = time.time()
        for m in moves:
            start_state[m.atom_id] = (m.frm.x, m.frm.y)
            end_state[m.atom_id] = (m.to.x, m.to.y)
        mid = time.time()
        result = reconfigure_layout(self.model, start_state, end_state)
        return result
