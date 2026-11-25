# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import random
from functools import partial

import numpy as np

from qds.circuit.circuit import init_circuit

# Imports for MQT benchmark
import pennylane as qml
from pennylane.transforms import commute_controlled, cancel_inverses

# Imports for Fermi-Hubbard model
from qds.circuit._fermihubbard import fermihubbard_model
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.synthesis.evolution import (
    LieTrotter,
    SuzukiTrotter,
)
from qiskit.quantum_info import SparsePauliOp


__all__ = [
    "RandomCircuitGenerator",
    "SyntheticQNNGenerator",
    "MQTGenerator",
    "FHGenerator",
    "SparseRandomPauliGenerator",
]


class RandomCircuitGenerator:
    def __init__(self, nqubit, tool="default"):
        self.nqubit = nqubit
        self.tool = tool

    def generate(self, depth, seed=None):
        max_try = 10 * depth
        if seed is not None:
            random.seed(seed)
        count = 0
        results = []
        prev_map = {}
        for _ in range(max_try):
            control = random.randint(0, self.nqubit - 1)
            target = random.randint(0, self.nqubit - 2)
            if target >= control:
                target = control + 1
            if control in prev_map:
                if prev_map[control] == target:
                    continue
            prev_map[control] = target
            results.append((control, target))
            count += 1
            if count == depth:
                break
        circuit = init_circuit(self.nqubit, self.tool)
        for r in results:
            circuit.cnot(r[0], r[1])
        return circuit


class SyntheticQNNGenerator:
    """Generate a QNN-like schedule without loading or compiling a framework circuit."""

    def __init__(
        self,
        nqubit,
        layers=343,
        gates_per_layer=None,
        tool="default",
    ):
        if nqubit < 2:
            raise ValueError("nqubit must be at least 2")
        if layers < 1:
            raise ValueError("layers must be positive")
        if gates_per_layer is None:
            gates_per_layer = nqubit // 2
        if not 1 <= gates_per_layer <= nqubit // 2:
            raise ValueError(
                "gates_per_layer must be between 1 and nqubit // 2"
            )
        self.nqubit = nqubit
        self.layers = layers
        self.gates_per_layer = gates_per_layer
        self.tool = tool

    def generate_components(self, seed=None):
        rng = random.Random(seed)
        qubits = list(range(self.nqubit))
        components = []

        for _ in range(self.layers):
            rng.shuffle(qubits)
            component = init_circuit(self.nqubit, self.tool)
            for gate_index in range(self.gates_per_layer):
                index = 2 * gate_index
                component.cnot(qubits[index], qubits[index + 1])
            components.append(component)

        return components

    def generate(self, seed=None):
        circuit = init_circuit(self.nqubit, self.tool)
        for component in self.generate_components(seed):
            for gate in component.twoq_gates():
                circuit.cnot(gate.c_index, gate.t_index)
        return circuit


class MQTGenerator:
    def __init__(
        self, nqubit, tool="default", pipeline=None, num_passes=None, basis_set=None
    ):
        """
        This class generates circuits from the MQT benchmark dataset.

        Args:
            nqubit (int): Number of qubits in the circuit.
            tool (str): The quantum device to use. Default is "default".

        Example:

        ```python
            from qds.circuit.generator import MQTGenerator
            gen = MQTGenerator(5)
            gen.generate()
        ```
        """
        [ds] = qml.data.load("other", name="mqt-bench")

        self.nqubit = nqubit
        self.tool = tool
        self.ds = ds
        self.type_benchmarks = set(ds.attr_info.keys())

        # Compiler settings
        self.pipeline = pipeline or [commute_controlled, cancel_inverses]
        self.num_passes = num_passes or 4
        self.basis_set = basis_set or ["CZ", "RX", "RY", "RZ"]

        self.unallowed_benchmarks = set()

    def search_available_nqubits(self, benchmark, max_qubits):
        results = []
        for j in range(max_qubits):
            if str(j) in benchmark:
                results.append(j)
        return results

    def generate(self, seed=None, benchmark=None):
        """
        Generate a circuit from the MQT benchmark dataset. The benchmarks available are in self.type_benchmarks.
        If benchmark is None, a random benchmark is selected.
        """
        random.seed(seed)

        # if benchmark is None, randomly select one
        user_benchmark = None
        if benchmark is None:
            benchmark = random.choice(list(self.type_benchmarks))
        else:
            user_benchmark = benchmark

        ops = None
        while ops is None:
            try:
                benchmark = getattr(self.ds, benchmark)
            except KeyError:
                if user_benchmark is not None:
                    raise ValueError(f"Unknown benchmark {user_benchmark}")
                self.unallowed_benchmarks.add(benchmark)
                benchmark = random.choice(
                    list(self.type_benchmarks - self.unallowed_benchmarks)
                )
            if str(self.nqubit) not in benchmark:
                raise AttributeError(
                    "The number of qubits {} is not available in the benchmark. Please choose from {}".format(
                        self.nqubit,
                        self.search_available_nqubits(benchmark, 2 * self.nqubit),
                    )
                )
            ops = self.get_compiled_ops(benchmark[str(self.nqubit)])

        circuit = init_circuit(self.nqubit, self.tool)

        for op in ops:
            if op.name == "CZ":
                circuit.cz(op.wires[0], op.wires[1])
            elif op.name == "CNOT":
                circuit.cnot(op.wires[0], op.wires[1])
            elif op.name == "RX":
                circuit.rx(op.parameters[0], op.wires[0])
            elif op.name == "RY":
                circuit.ry(op.parameters[0], op.wires[0])
            elif op.name == "RZ":
                circuit.rz(op.parameters[0], op.wires[0])
            elif op.name == "GlobalPhase":
                pass
            else:
                raise ValueError(
                    f"Unknown operation {op.name} (benchmark {benchmark}, nqubit {self.nqubit})"
                )

        return circuit

    def get_compiled_ops(self, ops):
        dev = qml.device("default.qubit")

        def make_circuit(ops):
            @partial(
                qml.compile,
                pipeline=self.pipeline,
                basis_set=self.basis_set,
                num_passes=self.num_passes,
            )
            @qml.qnode(dev)
            def circuit():
                for op in ops:
                    qml.apply(op)
                return qml.state()

            return circuit

        qnode = make_circuit(ops)
        qnode.construct([], {})
        tapes, _ = qnode.transform_program([qnode.tape])
        program, _ = qnode.device.preprocess()
        tapes, _ = program(tapes)
        return tapes[0].operations


class FHGenerator:
    """Generator of circuit for the Fermi-Hubbard model"""

    def __init__(
        self,
        nqubit=8,
        J=-1.0,
        U=2.0,
        periodic=False,
        encoding="jw",
        method="lt",
        reps=10,
        tool="default",
        basis_set=None,
    ):
        """
        Args:
            nqubit (int): Number of qubits in the circuit. Must be even. Number of sites is nqubit // 2.
            J (float): Hopping term.
            U (float): Interaction term.
            periodic (bool): Whether the lattice is periodic.
            encoding (str): Encoding of the Hamiltonian.
            method (str): Method for evolution. Options are 'lt' (Lie-Trotter) or 'st' (Suzuki-Trotter).
            reps (int): Number of repetitions for the evolution.
            tool (str): The quantum device to use. Default is "default".
            basis_set (list): Basis gates for the circuit. Default is ['rx', 'ry', 'rz', 'cz'].
        """
        if nqubit % 2 != 0:
            raise ValueError("Number of qubits must be even")
        nsites = nqubit // 2
        fhm, h1, h2 = fermihubbard_model(nsites, J, U, periodic, encoding)
        self.fhm = fhm
        self.reps = reps
        self.nsites = nsites
        self.nqubit = nqubit
        self.basis_set = basis_set or ["rx", "ry", "rz", "cz"]
        self.method = method
        self.tool = tool

    def generate(self, t=None, seed=None):
        """
        Args:
            t (float): Time of evolution. If None, a random time is selected.
            seed (int): Seed for random number generator.
        """
        random.seed(seed)
        t = t or random.random()

        evol = PauliEvolutionGate(self.fhm, t)
        if self.method == "lt":
            trotter = LieTrotter(self.reps).synthesize(evol)
        elif self.method == "st":
            trotter = SuzukiTrotter(self.reps).synthesize(evol)
        else:
            raise ValueError(f"Unknown method {self.method}")

        pm = generate_preset_pass_manager(
            basis_gates=self.basis_set, optimization_level=0
        )
        compiled_circ = pm.run(trotter)

        circuit = init_circuit(self.nqubit, self.tool)

        for gate in compiled_circ:
            op = gate.operation
            wires = [
                compiled_circ.find_bit(qubit).index for qubit in gate.qubits
            ]
            if op.name == "cz":
                circuit.cz(wires[0], wires[1])
            elif op.name == "cx":
                circuit.cnot(wires[0], wires[1])
            elif op.name == "rx":
                circuit.rx(float(op.params[0]), wires[0])
            elif op.name == "ry":
                circuit.ry(float(op.params[0]), wires[0])
            elif op.name == "rz":
                circuit.rz(float(op.params[0]), wires[0])
            else:
                raise ValueError(f"Unknown operation {op.name}")

        return circuit

class SparseRandomPauliGenerator:
    """Generates Hamiltonian simulaton for sparse random Pauli Hamiltonian.
    
    Family of Hamiltonians defined in https://arxiv.org/pdf/2302.03394.
    """
    def __init__(
        self,
        min_nqubit=10,
        max_nqubit=50,
        n_terms=10,
        method="lt",
        reps=4,
        tool="default",
        basis_set=None,
        k_local=None,
        force_k_local=False,
    ):
        self.reps = reps
        self.min_nqubit = min_nqubit
        self.max_nqubit = max_nqubit
        self.n_terms = n_terms
        self.method = method
        self.tool = tool
        self.basis_set = basis_set or ["rx", "ry", "rz", "cz"]
        assert k_local <= self.min_nqubit, "k_local must be less than or equal to nqubit"
        self.k_local = k_local
        self.force_k_local = force_k_local

    def generate(self, t=10):
        """
        Args:
            t (float): Time of evolution. 10 is the default value.
            seed (int): Seed for random number generator.
        """
        pauli_list = []
        coeff_list = []

        # make a random matrix of size n_terms x nqubit
        nqubit = random.randint(self.min_nqubit, self.max_nqubit)
        if self.k_local is None:
            ops = np.random.randint(0, 4, (self.n_terms, nqubit))
        else:
            ops = np.zeros((self.n_terms, nqubit), dtype=int)

            for i in range(self.n_terms):
                non_zero_positions = np.random.choice(nqubit, size=self.k_local, replace=False)
                if self.force_k_local:
                    ops[i, non_zero_positions] = np.random.randint(1, 4, size=self.k_local)
                else:
                    ops[i, non_zero_positions] = np.random.randint(0, 4, size=self.k_local)

        # make coefficients
        coeffs = np.random.randint(0, 2, self.n_terms)
        coeffs = np.where(coeffs == 0, -1, 1)
        
        # normalize by sqrt(n_terms)
        coeffs = coeffs / np.sqrt(self.n_terms)

        translator = {
            0: "I",
            1: "X",
            2: "Y",
            3: "Z",
        }

        translate = lambda x: translator[x]

        for indx in range(self.n_terms):
            pauli_m = ops[indx]
            coeff = coeffs[indx]
            
            pauli_str = np.vectorize(translate)(pauli_m)
            
            pauli_list.append("".join(pauli_str))
            coeff_list.append(coeff)

        hamiltonian = [(p, c) for p, c in zip(pauli_list, coeff_list)]
        operator = SparsePauliOp.from_list(hamiltonian)

        evol = PauliEvolutionGate(operator, t)
        if self.method == "lt":
            trotter = LieTrotter(self.reps).synthesize(evol)
        elif self.method == "st":
            trotter = SuzukiTrotter(self.reps).synthesize(evol)
        else:
            raise ValueError(f"Unknown method {self.method}")

        pm = generate_preset_pass_manager(
            basis_gates=self.basis_set, optimization_level=0
        )
        compiled_circ = pm.run(trotter)

        circuit = init_circuit(nqubit, self.tool)

        for gate in compiled_circ:
            op = gate.operation
            wires = [compiled_circ.find_bit(qubit).index for qubit in gate.qubits]
            if op.name == "cz":
                circuit.cz(wires[0], wires[1])
            elif op.name == "cx":
                circuit.cnot(wires[0], wires[1])
            elif op.name == "rx":
                circuit.rx(float(op.params[0]), wires[0])
            elif op.name == "ry":
                circuit.ry(float(op.params[0]), wires[0])
            elif op.name == "rz":
                circuit.rz(float(op.params[0]), wires[0])
            else:
                raise ValueError(f"Unknown operation {op.name}")

        return circuit