import numpy as np
from qiskit.quantum_info import random_clifford
from qiskit import QuantumCircuit

def prepare_meas_circuits(circuit: QuantumCircuit, n_shadows):
    # append n_shadows random clifford unitaries to the circuit
    # returns collection of circuits along with collection of the unitaries used

    circuits = []
    unitaries = []
    for i in range(n_shadows):
        clifford = random_clifford(num_qubits=circuit.num_qubits, seed=None)
        cir = circuit.compose(clifford.to_circuit())
        cir.measure_all()
        circuits.append(cir)
        unitaries.append(clifford.to_matrix())

    return circuits, unitaries

def estimate_fidelity_from_shadow(unitaries, measured_bitstrings, target_state):
    # fidelity estimation with classical shadows implemented as per
    # Approximate complex amplitude encoding algorithm and its application to data classification problems.
    # Mitsuda et al. https://arxiv.org/abs/2211.13039 (page 2, equation (10))
    measured_bitstrings = [np.matrix(b) for b in measured_bitstrings]
    unitaries = [np.matrix(u) for u in unitaries]
    target_state = np.matrix(target_state)

    n_shots = len(measured_bitstrings)
    n_qubits = np.log2(target_state.shape[1])
    observable = target_state.T @ np.conj(target_state)

    fidelity = 0
    factor = 2**n_qubits + 1
    for i in range(n_shots):
        product = np.conj(measured_bitstrings[i]) @ (unitaries[i] @ observable @ unitaries[i].H) @ measured_bitstrings[i].T
        cur_fid = factor * product - 1
        #print(f"{cur_fid} -- {measured_bitstrings[i]}")
        fidelity += cur_fid

    fidelity = fidelity/n_shots#/(2**n_qubits)
    fidelity = np.abs(np.sum(fidelity))
    return fidelity
