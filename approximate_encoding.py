from qiskit import Aer, execute
from fidelity_estimation import prepare_meas_circuits, estimate_fidelity_from_shadow
from qiskit import QuantumCircuit
import numpy as np
from qiskit.quantum_info import Statevector
from utils.MinimizeWrapper import MinimizeWrapper

def approx_encode(ansatz: QuantumCircuit, target_statevector, backend=Aer.get_backend("aer_simulator"), n_shadows=100, initial_params=None, fixed_unitaries=False):
    params = ansatz.parameters
    initial_param_values = np.random.uniform(-np.pi, np.pi, size=len(params)) if initial_params is None else initial_params
    #initial_param_values = np.random.normal(0, np.pi/4, size=len(params)) if initial_params is None else initial_params
    meas_circuits, unitaries = prepare_meas_circuits(ansatz, n_shadows)

    def objective(x):
        nonlocal meas_circuits, unitaries
        if not fixed_unitaries:
            meas_circuits, unitaries = prepare_meas_circuits(ansatz, n_shadows)

        final_meas_circuits = [c.bind_parameters(x) for c in meas_circuits]
        job = execute(final_meas_circuits, backend=backend, shots=1)
        res = job.result()
        counts = [res.get_counts(i) for i,c in enumerate(final_meas_circuits)]
        bs = [Statevector.from_label(list(cnt.keys())[0]) for cnt in counts]
        obj_value = -estimate_fidelity_from_shadow(unitaries, bs, target_statevector)
        #print(f"current params = {list(x)}")
        print(obj_value)
        return obj_value

    def store_intermediate_result(xk):
        return#print(f"{xk}")

    # MinimizeWrapper stops estimating when fidelity is below threshold and stores optimization history to obtain the best parameterization observed during the optimization
    res = MinimizeWrapper(threshold=-0.95).minimize(fun=objective, x0=initial_param_values, method="COBYLA", callback=store_intermediate_result,
           options=({"rhobeg": np.pi/4, "maxiter": 50, "tol": np.pi/(2**6)}))

    return res
