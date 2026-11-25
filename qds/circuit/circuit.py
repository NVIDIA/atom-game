# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import TypeVar

import networkx
from abc import ABC, abstractmethod
from networkx import DiGraph, draw_networkx, get_node_attributes

__all__ = [
     "Gate",
     "OneQubitGate",
     "TwoQubitGate",
     "GateSequence",
     "CircuitDag",
     "init_circuit",
     "decompose_circuit",
]

class Gate(ABC):
    @abstractmethod
    def is_two_gate(self):
        pass

    @abstractmethod
    def can_reorder(self, g):
        pass


class OneQubitGate(Gate):
    def __init__(self, name, index):
        self.name = name
        self.index = index

    def is_two_gate(self):
        return False

    def can_reorder(self, g):
        if g.is_two_gate:
            return self.index != g.c_index and self.index != g.t_index
        return self.index != g.index

    def __repr__(self):
        return f"{self.name}({self.index})"

    def __eq__(self, __value):
        if not isinstance(__value, OneQubitGate):
            return False
        return self.name == __value.name and self.index == __value.index


class TwoQubitGate(Gate):
    def __init__(self, name, c_index, t_index):
        self.name = name
        self.c_index = c_index
        self.t_index = t_index

    def is_two_gate(self):
        return True

    def can_reorder(self, g):
        if g.is_two_gate:
            c_no_match = self.c_index != g.c_index and self.c_index != g.t_index
            t_no_match = self.t_index != g.c_index and self.t_index != g.t_index
            return c_no_match and t_no_match
        return g.can_reorder(self)

    def __repr__(self):
        return f"{self.name}({self.c_index},{self.t_index})"

    def __eq__(self, __value):
        if not isinstance(__value, TwoQubitGate):
            return False
        return self.name == __value.name and self.c_index == __value.c_index and self.t_index == __value.t_index


class GateSequence:
    def __init__(self, nqubit, drawable=False):
        self.nqubit = nqubit
        self.gates = []
        self.drawable = drawable
        if drawable:
            from qwrapper.circuit import QWrapper, QiskitCircuit
            self.qc2 = QiskitCircuit(nqubit)

    @staticmethod
    def frm(gates, nqubit):
        seq = GateSequence(nqubit)
        seq.gates = gates
        return seq

    def has_twoq_gates(self, q):
        for g in self.gates:
            if g.is_two_gate and (q == g.c_index or q == g.t_index):
                return True
        return False

    def twoq_gates(self, player=None, tuple=False):
        results = []
        for g in self.gates:
            if not g.is_two_gate:
                continue
            if player is not None and (g.c_index != player and g.t_index != player):
                continue
            if tuple:
                results.append((g.c_index, g.t_index))
            else:
                results.append(g)
        return results

    def cx(self, c_index, t_index):
        self.cnot(c_index, t_index)
        if self.drawable:
            self.qc2.cx(c_index, t_index)

    def h(self, index):
        self.gates.append(OneQubitGate("h", index))
        if self.drawable:
            self.qc2.h(index)

    def s(self, index):
        self.gates.append(OneQubitGate("s", index))
        if self.drawable:
            self.qc2.s(index)

    def sdag(self, index):
        self.gates.append(OneQubitGate("sdag", index))
        if self.drawable:
            self.qc2.sdag(index)

    def hsdag(self, index):
        self.sdag(index)
        self.h(index)
        if self.drawable:
            self.qc2.hsdag(index)

    def x(self, index):
        self.gates.append(OneQubitGate("x", index))
        if self.drawable:
            self.qc2.x(index)

    def y(self, index):
        self.gates.append(OneQubitGate("y", index))
        if self.drawable:
            self.qc2.y(index)

    def z(self, index):
        self.gates.append(OneQubitGate("z", index))
        if self.drawable:
            self.qc2.z(index)

    def rx(self, theta, index):
        self.gates.append(OneQubitGate(f"rx_{theta}", index))
        if self.drawable:
            self.qc2.rx(theta, index)

    def ry(self, theta, index):
        self.gates.append(OneQubitGate(f"ry_{theta}", index))
        if self.drawable:
            self.qc2.ry(theta, index)

    def rz(self, theta, index):
        self.gates.append(OneQubitGate(f"rz_{theta}", index))
        if self.drawable:
            self.qc2.rz(theta, index)

    def cnot(self, c_index, t_index):
        self.gates.append(TwoQubitGate("cx", c_index, t_index))
        if self.drawable:
            self.qc2.cnot(c_index, t_index)

    def cy(self, c_index, t_index):
        self.gates.append(TwoQubitGate("cy", c_index, t_index))
        if self.drawable:
            self.qc2.cy(c_index, t_index)

    def cz(self, c_index, t_index):
        self.gates.append(TwoQubitGate("cz", c_index, t_index))
        if self.drawable:
            self.qc2.cz(c_index, t_index)

    def copy(self):
        raise NotImplementedError

    def get_async_samples(self, nshot):
        raise NotImplementedError

    def get_q_register(self):
        raise NotImplementedError

    def measure_all(self):
        raise NotImplementedError

    def barrier(self):
        raise NotImplementedError

    def draw(self, output="mpl"):
        if self.drawable:
            return self.qc2.draw(output)
        raise NotImplementedError

    def draw_and_show(self):
        if self.drawable:
            return self.qc2.draw_and_show()
        raise NotImplementedError

    def get_samples(self, nshot):
        raise NotImplementedError

    def get_counts(self, nshot):
        raise NotImplementedError

    def post_select(self, target, index):
        raise NotImplementedError

    def get_state_vector(self):
        raise NotImplementedError

    @classmethod
    def execute_post_selects(cls, vector, post_selects, n_qubit):
        raise NotImplementedError

    @classmethod
    def execute_post_select(cls, state_vector, i, bit, n_qubit):
        raise NotImplementedError

    def __repr__(self):
        strings = []
        for gate in self.gates:
            strings += [f"{gate}"]
        return " ".join(strings)


T = TypeVar('T')


class GateWithDepth:
    def __init__(self, val: Gate, depth) -> None:
        self.val = val
        self.depth = depth

    def __repr__(self) -> str:
        return str(self.val)

    def __lt__(self, other):
        if not isinstance(other, type(self)):
            return NotImplemented
        return id(self) < id(other)


class CircuitDag(DiGraph):
    @staticmethod
    def from_circuit(circuit: GateSequence):
        result = CircuitDag()
        for g in circuit.gates:
            if g.is_two_gate():
                result.append(g)
        return result

    @staticmethod
    def make_node(gate: Gate):
        return GateWithDepth(gate, None)

    def append(self, gate: Gate):
        new_node = self.make_node(gate)
        max_depth = -1
        for node in list(self.nodes):
            if not gate.can_reorder(node.val):
                self.add_edge(node, new_node)
                if max_depth < node.depth:
                    max_depth = node.depth
                for pred in self.pred[node]:
                    self.add_edge(pred, new_node)

        new_node.depth = max_depth + 1
        if gate.is_two_gate():
            index = gate.c_index
        else:
            index = gate.index
        self.add_node(new_node, pos=(new_node.depth, index))

    def rearrange(self):
        """
        Recalculate the depth value
        """
        old_position = networkx.get_node_attributes(self, 'pos')
        new_position = {}
        nodes = self.get_initial_nodes()
        for node in nodes:
            node.depth = 0
            self.adaptive_fill_depth(node)
        for n in list(self.nodes):
            new_position[n] = (n.depth, old_position[n][1])
        networkx.set_node_attributes(self, new_position, 'pos')

    def adaptive_fill_depth(self, node):
        results = []
        for n in self.succ[node]:
            add = True
            for n2 in self.pred[n]:
                if n2.depth > node.depth + 1:
                    add = False
            if add:
                n.depth = node.depth + 1
                results.append(n)
        for r in results:
            self.adaptive_fill_depth(r)

    def get_initial_nodes(self):
        results = []
        for node in self.nodes:
            if len(self.pred[node]) == 0:
                results.append(node)
        return results

    def draw(self):
        pos = get_node_attributes(self, 'pos')
        draw_networkx(self, pos)


def init_circuit(nqubit, tool):
    if tool == 'qiskit':
        from qwrapper.circuit import QWrapper, QiskitCircuit
        return QiskitCircuit(nqubit)
    return GateSequence(nqubit)


def decompose_circuit(circuit, n_components: int):
    if not isinstance(circuit, GateSequence):
        raise NotImplementedError("Circuit must be an instance of GateSequence")
    two_gates_count = 0
    for g in circuit.gates:
        if g.is_two_gate():
            two_gates_count += 1
    base_count = int(two_gates_count / n_components)
    extra = two_gates_count % n_components
    counts = [base_count] * n_components
    for i in range(extra):
        counts[i] += 1

    results = []
    result = []
    results.append(result)
    current_index = 0
    current_count = 0
    for g in circuit.gates:
        result.append(g)
        if g.is_two_gate():
            current_count += 1
        if current_index != len(counts) - 1 and current_count == counts[current_index]:
            current_count = 0
            current_index += 1
            result = []
            results.append(result)

    circuits = []
    for r in results:
        c = GateSequence(circuit.nqubit)
        c.gates = r
        circuits.append(c)
    return circuits
