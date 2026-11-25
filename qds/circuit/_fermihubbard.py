# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from qiskit.quantum_info import SparsePauliOp
import openfermion as of


def get_n_qubits(terms):
    max_qubit = 0
    for term in terms.keys():
        for qubit, _ in term:
            if qubit > max_qubit:
                max_qubit = qubit
    return max_qubit + 1


def convert_of_to_sparse_pauli_op(terms):
    n_qubits = get_n_qubits(terms)
    pauli_list = []
    coeff_list = []

    for term, coeff in terms.items():
        pauli_str = ["I"] * n_qubits
        for qubit, pauli in term:
            pauli_str[qubit] = pauli
        pauli_list.append("".join(pauli_str))
        coeff_list.append(coeff)

    hamiltonian = [(p, c) for p, c in zip(pauli_list, coeff_list)]
    operator = SparsePauliOp.from_list(hamiltonian)
    return operator


def fermihubbard_model(nsites=4, J=-1.0, U=2.0, periodic=False, encoding='jw'):
    """
    Encoding can be 'jw' or 'bk'
    """
    nqubits = 2 * nsites

    one_body_terms = [
        op + of.hermitian_conjugated(op) for op in (
            of.FermionOperator(((i, 1), (i + 2, 0)), coefficient=J) for i in range(nqubits - 2)
        )
    ]

    two_body_terms = [
        of.FermionOperator(((i, 1), (i, 0), (i + 1, 1), (i + 1, 0)), coefficient=U)
        for i in range(0, nqubits, 2)
    ]

    hopping_term = sum(one_body_terms)
    interaction_term = sum(two_body_terms)

    if encoding in ['jw', 'jordan_wigner']:
        hop = of.transforms.jordan_wigner(hopping_term)
        iop = of.transforms.jordan_wigner(interaction_term)
    elif encoding in ['bk', 'bravyi_kitaev']:
        hop = of.transforms.bravyi_kitaev(hopping_term)
        iop = of.transforms.bravyi_kitaev(interaction_term)
    else:
        raise ValueError(f'Unknown encoding: {encoding}')

    hop_op = convert_of_to_sparse_pauli_op(hop.terms)
    iop_op = convert_of_to_sparse_pauli_op(iop.terms)
    return hop_op + iop_op, hop_op, iop_op
