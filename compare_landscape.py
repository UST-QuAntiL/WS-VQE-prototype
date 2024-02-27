import json
import numpy as np
from tqdm import tqdm
from matplotlib import cm
import matplotlib.pyplot as plt
from qiskit import Aer, execute
from qiskit.quantum_info import SparsePauliOp, Statevector, state_fidelity
from qiskit_algorithms.minimum_eigensolvers import VQE # quantum solver
from qiskit.circuit.library import TwoLocal, EfficientSU2
from qiskit_aer.primitives import Estimator as AerEstimator
from utils.results_storage import store_results, load_results
from fidelity_estimation import prepare_meas_circuits, estimate_fidelity_from_shadow

problem_size = 3 # = num_qubits, matrix dimensions will be 2**problem_size
circuit_ansatz = EfficientSU2(num_qubits=problem_size, reps=2)
num_parameters = len(circuit_ansatz.parameters)

n_shots = 200 # number of shots for VQE
n_shadows = 400 # number of classical snapshots for ACAE

# obtain a problem instance from the other experiments (just take the first one)
fname_in = "results/ws-vqe_comparison-n3-aiter3-iter100-dynOptComp-sparse.5-uniform-init.json"
fname_out = "results/compare_landscape.json"
results = load_results(fname_in)
problem_instance = results[0]["problem instance"]
target_statevector = results[0]["approx. eigenstate"]
ref_statevector = results[0]["ref eigenstate"]
pauli_operator = SparsePauliOp.from_operator(problem_instance)

# select a random pair of parameters for the landscape, set the others to random values
xy = np.random.choice(num_parameters, 2, replace=False)
param_values = np.random.uniform(-np.pi, np.pi, size=num_parameters-2)
variable_params = []
fixed_params = {}
for i, param in enumerate(circuit_ansatz.parameters):
    if i in xy:
        variable_params.append(param)
    else:
        fixed_params[param] = param_values[i-len(variable_params)]

circuit_2Dansatz = circuit_ansatz.assign_parameters(fixed_params)
print(circuit_2Dansatz)
print(fixed_params)
print(variable_params)

""" VQE objective function """
def objective_vqe(x, y):
    # get the evaluate energy function from qiskit#s VQE implementation
    noiseless_estimator = AerEstimator(run_options={"seed": 1, "shots": n_shots}, transpile_options={"seed_transpiler": 1},)
    vqe = VQE(noiseless_estimator, circuit_2Dansatz, optimizer=None, initial_point=[])
    qiskit_eval_energy = vqe._get_evaluate_energy(circuit_2Dansatz, pauli_operator)

    # evaluate for dummy parameter set to 0, i.e., the dummy parameter doesn't change anything
    result = qiskit_eval_energy([x, y])
    return result

""" fidelity estimation """
meas_circuits, unitaries = prepare_meas_circuits(circuit_2Dansatz, n_shadows)
def objective_fid_est(x, y):
    # todo
    final_meas_circuits = [c.assign_parameters([x, y]) for c in meas_circuits]
    job = execute(final_meas_circuits, backend=Aer.get_backend("aer_simulator"), shots=1)
    res = job.result()
    counts = [res.get_counts(i) for i,c in enumerate(final_meas_circuits)]
    bs = [Statevector.from_label(list(cnt.keys())[0]) for cnt in counts]
    obj_value = -estimate_fidelity_from_shadow(unitaries, bs, target_statevector)
    #print(f"current params = {list(x)}")
    return obj_value

""" actual fidelity """
def objective_fid(ref_state, x, y):
    statev = execute(circuit_2Dansatz.assign_parameters([x, y]), backend=Aer.get_backend("statevector_simulator")).result().get_statevector()
    return -state_fidelity(ref_state, statev)

# prepare equidistant grid
step = np.pi/20
gridx, gridy = np.arange(-np.pi, np.pi + step, step), np.arange(-np.pi, np.pi + step, step)
grid_x, grid_y = np.meshgrid(gridx, gridy)
mesh_shape = grid_x.shape
grid_x, grid_y = grid_x.flatten(), grid_y.flatten()

# obtain and store properties for each point in the grid
vqe_objective_values = []
estfid_objective_values = []
appxfid_objective_values = []
reffid_objective_values = []
for i, (x, y) in enumerate(tqdm(zip(grid_x, grid_y))):
    vqe_objective_value = objective_vqe(x, y)
    vqe_objective_values.append(vqe_objective_value)

    estfid_objective_value = objective_fid_est(x, y)
    estfid_objective_values.append(estfid_objective_value)

    appxfid_objective_value = objective_fid(target_statevector, x, y)
    appxfid_objective_values.append(appxfid_objective_value)

    reffid_objective_value = objective_fid(ref_statevector, x, y)
    reffid_objective_values.append(reffid_objective_value)

# dump to file
results_json = {
        "problem_instance": problem_instance.dumps().decode("ISO-8859-1"),
        "target_statevector": str(list(target_statevector)),
        "fixed_params": {str(k): v for k, v in fixed_params.items()},
        "variable_params": [str(param) for param in variable_params],
        "vqe_objective_values": [[x, y, vqe_objective_values[i]] for i, (x, y) in enumerate(zip(grid_x, grid_y))],
        "estfid_objective_values": [[x, y, estfid_objective_values[i]] for i, (x, y) in enumerate(zip(grid_x, grid_y))],
        "appxfid_objective_values": [[x, y, appxfid_objective_values[i]] for i, (x, y) in enumerate(zip(grid_x, grid_y))],
        "reffid_objective_values": [[x, y, reffid_objective_values[i]] for i, (x, y) in enumerate(zip(grid_x, grid_y))]
    }

with open(fname_out, "a") as outfile:
        json.dump(results_json, outfile)
        outfile.write("\n")

# plot
fig = plt.figure()
ax = fig.add_subplot(1,4,1)
grid_x, grid_y, vqe_objective_values = grid_x.reshape(mesh_shape), grid_y.reshape(mesh_shape), np.array(vqe_objective_values).reshape(mesh_shape)
img = ax.contourf(grid_x, grid_y, vqe_objective_values, cmap=cm.get_cmap('viridis', 256), antialiased=True)

ax = fig.add_subplot(1,4,2)
grid_x, grid_y, estfid_objective_values = grid_x.reshape(mesh_shape), grid_y.reshape(mesh_shape), np.array(estfid_objective_values).reshape(mesh_shape)
img = ax.contourf(grid_x, grid_y, estfid_objective_values, cmap=cm.get_cmap('viridis', 256), antialiased=True)

ax = fig.add_subplot(1,4,3)
grid_x, grid_y, appxfid_objective_values = grid_x.reshape(mesh_shape), grid_y.reshape(mesh_shape), np.array(appxfid_objective_values).reshape(mesh_shape)
img = ax.contourf(grid_x, grid_y, appxfid_objective_values, cmap=cm.get_cmap('viridis', 256), antialiased=True)

ax = fig.add_subplot(1,4,4)
grid_x, grid_y, reffid_objective_values = grid_x.reshape(mesh_shape), grid_y.reshape(mesh_shape), np.array(reffid_objective_values).reshape(mesh_shape)
img = ax.contourf(grid_x, grid_y, reffid_objective_values, cmap=cm.get_cmap('viridis', 256), antialiased=True)

plt.show()
