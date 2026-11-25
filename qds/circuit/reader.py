# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import random, json
from typing import List

from abc import ABC, abstractmethod
from qds.circuit.circuit import GateSequence, CircuitDag


__all__ = ["CircuitReader", "BatchCircuitReader", "MultiBatchCircuitReader"]

class CircuitReader(ABC):
    @abstractmethod
    def read(self) -> GateSequence:
        """
        Reads the circuit and returns it as a GateSequence. The position is incremented.
        """
        pass

    @abstractmethod
    def reset(self, start=0, end=None):
        pass

    @abstractmethod
    def see_next(self, n_components, player=None):
        pass

    @abstractmethod
    def n_components(self):
        pass

    @abstractmethod
    def is_done(self):
        pass

    @abstractmethod
    def num_qubits(self):
        pass

    @abstractmethod
    def current_index(self):
        pass

    @abstractmethod
    def truncate(self, count):
        pass

    @abstractmethod
    def count_twoq_gates(self, player):
        pass


class BatchCircuitReader(CircuitReader):
    def __init__(self, circuits: [GateSequence]):
        self._current_index = 0
        self.components = circuits
        self.end = self.n_components()

    def __deepcopy__(self, memo):
        """Share the circuit and copy only the position in it.

        Playing never changes a component, so a planner that copies the game to
        try a move out would otherwise duplicate all eight hundred layers of the
        circuit on every node it expands.
        """
        twin = self.__class__.__new__(self.__class__)
        memo[id(self)] = twin
        twin.components = self.components
        twin._current_index = self._current_index
        twin.end = self.end
        return twin

    def to_json(self):
        results = []
        for component in self.components:
            results.append(component.twoq_gates(tuple=True))
        return json.dumps({"circuit": results, "nqubit": self.num_qubits()})
    
    @classmethod
    def from_json(cls, v):
        circuits = []
        obj = json.loads(v)
        for two_q_gates in obj["circuit"]:
            sequence = GateSequence(obj["nqubit"])
            for g in two_q_gates:
                sequence.cnot(g[0], g[1])
            circuits.append(sequence)
        return BatchCircuitReader(circuits)
                

    @staticmethod
    def from_circuit(circuit: GateSequence):
        circuits = BatchCircuitReader.build(circuit)
        return BatchCircuitReader(circuits)

    @property
    def current_index(self):
        return self._current_index

    def count_twoq_gates(self, player):
        count = 0
        for component in self.components[self.current_index:]:
            count += len(component.twoq_gates(player))
        return count

    def truncate(self, count):
        self.components = self.components[:count]

    def n_components(self):
        return len(self.components)

    def is_done(self):
        return self.current_index == self.end

    def num_qubits(self):
        return self.components[0].nqubit

    def read(self) -> GateSequence:
        """
        Reads the circuit and returns it as a GateSequence. The position is incremented.
        """
        if self.current_index == self.end:
            return None
        result = self.components[self.current_index]
        self._current_index += 1
        return result

    def reset(self, start=0, end=None):
        self._current_index = start
        if end is None:
            self.end = self.n_components()
        else:
            self.end = min(self.n_components(), end)

    def see_next(self, count, player=None) -> [GateSequence]:
        """
        See the next {count} gate components
        """
        if player is None:
            results = self.components[self.current_index: min(self.current_index + count, self.n_components())]
        else:
            results = []
            for component in self.components[self.current_index:]:
                if not component.has_twoq_gates(player):
                    continue
                if len(component.gates) == 1:
                    continue
                results.append(component)
                if len(results) == count:
                    return results
        if len(results) < count:
            for _ in range(count - len(results)):
                results.append(GateSequence(self.num_qubits()))
        return results

    @staticmethod
    def build(circuit: GateSequence):
        results = []
        dag = CircuitDag.from_circuit(circuit)
        while True:
            nodes = dag.get_initial_nodes()
            gates = [n.val for n in nodes]
            for n in nodes:
                dag.remove_node(n)
            if len(nodes) == 0:
                return results
            sequence = GateSequence.frm(gates, circuit.nqubit)
            results.append(sequence)


class MultiBatchCircuitReader(CircuitReader):
    def __init__(self, circuit_readers: List[CircuitReader], reset_strategy="random"):
        """
        Reset strategy can be either "random" or "sequential".
        """
        self.circuit_readers = circuit_readers
        self.reset_strategy = reset_strategy

        if reset_strategy == "random":
            self.current_reader_indx = random.randint(0, len(circuit_readers) - 1)
        elif reset_strategy == "sequential":
            self.current_reader_indx = 0
        else:
            raise ValueError(f"Invalid reset strategy: {reset_strategy}")
        
        self.current_reader = self.circuit_readers[self.current_reader_indx]
        self.max_n_qubits = max([reader.num_qubits() for reader in self.circuit_readers])
        
    def to_jsons(self):
        results = []
        for reader in self.circuit_readers:
            results.append(reader.to_json())
        return results
    
    @classmethod
    def from_jsons(cls, jsons, reset_strategy="random"):
        results = []
        for json in jsons:
            reader = BatchCircuitReader.from_json(json)
            results.append(reader)
        return MultiBatchCircuitReader(results, reset_strategy=reset_strategy)
    
    def reset(self, start=0, end=None):
        if self.reset_strategy == "random":
            self.current_reader_indx = random.randint(0, len(self.circuit_readers) - 1)
        elif self.reset_strategy == "sequential":
            self.current_reader_indx = (self.current_reader_indx + 1) % len(self.circuit_readers)

        self.current_reader = self.circuit_readers[self.current_reader_indx]
        self.current_reader.reset(start, end)

    @staticmethod
    def from_circuits(circuits: List[GateSequence], reset_strategy="random"):
        readers = [BatchCircuitReader.from_circuit(circuit) for circuit in circuits]
        return MultiBatchCircuitReader(readers, reset_strategy=reset_strategy)


    def read(self) -> GateSequence:
        return self.current_reader.read()
    
    def n_components(self):
        return self.current_reader.n_components()
    
    def max_n_components(self):
        return max([reader.n_components() for reader in self.circuit_readers])
    
    def is_done(self):
        return self.current_reader.is_done()
    
    def num_qubits(self):
        return self.current_reader.num_qubits()
    
    def current_index(self):
        return self.current_reader.current_index
    
    def count_twoq_gates(self, player):
        return self.current_reader.count_twoq_gates(player)
    
    def truncate(self, count):
        for reader in self.circuit_readers:
            reader.truncate(count)
        self.current_reader.truncate(count)

    def see_next(self, count, player=None) -> [GateSequence]:
        return self.current_reader.see_next(count, player)

    def has_twoq_gates(self, player):
        return self.current_reader.has_twoq_gates(player)

    # def __getattribute__(self, name):
    #     if name in ['circuit_readers', 'current_reader']:
    #         return object.__getattribute__(self, name)
    #     return getattr(self.current_reader, name)