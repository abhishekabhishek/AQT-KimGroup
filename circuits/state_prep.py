"""Implementation of qiskit circuits to prepare W states for arbitrary N
Original source - https://github.com/vutuanhai237/UC-VQA

Modified to work with the data collection notebook
"""

import qiskit
import numpy as np

def w(qc: qiskit.QuantumCircuit, num_qubits: int, shift: int = 0):
    """The below codes is implemented from [this paper]
    (https://arxiv.org/abs/1606.09290)
    Args:
        - num_qubits (int): number of qubits
        - shift (int, optional): begin wire. Defaults to 0.
    Raises:
        - ValueError: When the number of qubits is not valid
    Returns:
        - qiskit.QuantumCircuit
    """
    if num_qubits < 2:
        raise ValueError('W state must has at least 2-qubit')
    if num_qubits == 2:
        # |W> state ~ |+> state
        qc.h(0)
        return qc
    if num_qubits == 3:
        # Return the base function
        qc.w3(shift)
        return qc
    else:
        # Theta value of F gate base on the circuit that it acts on
        theta = np.arccos(1 / np.sqrt(qc.num_qubits - shift))
        qc.cf(theta, shift, shift + 1)
        # Recursion until the number of qubits equal 3
        w(qc, num_qubits - 1, qc.num_qubits - (num_qubits - 1))
        for i in range(1, num_qubits):
            qc.cnot(i + shift, shift)
    return qc

def create_w_state(num_qubits):
    """Create n-qubit W state based on the its number of qubits
    Args:
        - qc (qiskit.QuantumCircuit): init circuit
    Returns:
        - qiskit.QuantumCircuit
    """
    qc = qiskit.QuantumCircuit(num_qubits)
    qc.x(0)
    qc = w(qc, qc.num_qubits)
    return qc


def cf(qc: qiskit.QuantumCircuit, theta: float, qubit1: int, qubit2: int):
    """Add Controlled-F gate to quantum circuit
    Args:
        - qc (qiskit.QuantumCircuit): ddded circuit
        - theta (float): arccos(1/sqrt(num_qubits), base on number of qubit
        - qubit1 (int): control qubit
        - qubit2 (int): target qubit
    Returns:
        - qiskit.QuantumCircuit: Added circuit
    """
    cf = qiskit.QuantumCircuit(2)
    u = np.array([[1, 0, 0, 0], [0, np.cos(theta), 0,
                                 np.sin(theta)], [0, 0, 1, 0],
                  [0, np.sin(theta), 0, -np.cos(theta)]])
    cf.unitary(u, [0, 1])
    cf_gate = cf.to_gate(label='CF')
    qc.append(cf_gate, [qubit1, qubit2])
    return qc


def w3(circuit: qiskit.QuantumCircuit, qubit: int):
    """Create W state for 3 qubits
    Args:
        - circuit (qiskit.QuantumCircuit): added circuit
        - qubit (int): the index that w3 circuit acts on
    Returns:
        - qiskit.QuantumCircuit: added circuit
    """
    qc = qiskit.QuantumCircuit(3)
    theta = np.arccos(1 / np.sqrt(3))
    qc.cf(theta, 0, 1)
    qc.cx(1, 0)
    qc.ch(1, 2)
    qc.cx(2, 1)
    w3 = qc.to_gate(label='w3')
    # Add the gate to your circuit which is passed as the first argument to cf
    # function:
    circuit.append(w3, [qubit, qubit + 1, qubit + 2])
    return circuit


qiskit.QuantumCircuit.w3 = w3
qiskit.QuantumCircuit.cf = cf
