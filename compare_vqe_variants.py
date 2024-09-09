from qiskit.quantum_info import SparsePauliOp
import numpy as np
from utils.helper_functions import generate_hermitian, inverse_power_method, get_gershgorin_extrema
from utils.results_storage import store_results, load_results
from qiskit.algorithms.minimum_eigensolvers import NumPyMinimumEigensolver # classical solver
from qiskit.algorithms.minimum_eigensolvers import VQE # quantum solver
from qiskit.circuit.library import TwoLocal, EfficientSU2
from qiskit.algorithms.optimizers import SPSA, ADAM, COBYLA
from qiskit.utils import algorithm_globals
from qiskit_aer.primitives import Estimator as AerEstimator
from qiskit import Aer, execute, QuantumCircuit
from qiskit.circuit import Parameter
import matplotlib.pyplot as plt
import matplotlib as mpl
from approximate_encoding import approx_encode
import json
from matplotlib.gridspec import GridSpec

problem_size = 3 # = num_qubits, matrix dimensions will be 2**problem_size
sparsity = 0.5 # probability of matrix entry being zero

circuit_ansatz = EfficientSU2(num_qubits=problem_size, reps=2)
# ansatz_part1 = EfficientSU2(num_qubits=problem_size, reps=1, skip_final_rotation_layer=True, parameter_prefix="1")
# ansatz_part2 = EfficientSU2(num_qubits=problem_size, reps=1, entanglement="linear")
# circuit_ansatz = ansatz_part1.compose(ansatz_part2)

num_parameters = len(circuit_ansatz.parameters)
iterations = 100 # optimizer's iterations for VQE
n_shots = 200 # number of shots for VQE
n_shadows = 400 # number of classical snapshots for ACAE

a_seed = None # determines problem instance
b_seed = 124 # determines randomness in simulation
algorithm_globals.random_seed = b_seed


# store results to file/evaluate results from file
fname = "results/ws-vqe_comparison-n3-aiter3-iter100-dynOptComp-sparse.5-uniform-init.json"

#fname = "results/ws-vqe_comparison-n5-aiter12-aeiter100-iter200-newstepsize-16-24-64-ae-800shadows400shots-sparse.9-uniform-init.json"
#optimizer = ADAM(maxiter=iterations)

# default optimizer
optimizer = COBYLA(maxiter=iterations, rhobeg=3*np.pi/8)
# optimizer for biased initial state
optimizer1 = COBYLA(maxiter=iterations, rhobeg=3*np.pi/64)
# optimizer for ACAE pretrained parameters
optimizer2 = COBYLA(maxiter=iterations, rhobeg=3*np.pi/16)


def generate_problem_instance(seed=None, approx_iterations=3):
    if seed:
        np.random.seed(seed)

    # generate problem instance = a hermitian matrix
    while True:
        operator_matrix = generate_hermitian(n=problem_size, complex=True, sparsity=None)
        if np.any(operator_matrix):
            break
        else:
            print('\033[91m'+"All zeroes matrix, generating a new one."+'\033[0m')

    # get SparsePauliOp (qiskit scales coefficients by 1/2^n)
    pauli_operator = SparsePauliOp.from_operator(operator_matrix)

    # group by commuting paulis -> determines # of circuits that are actually executed
    pauli_lists = pauli_operator.paulis.group_commuting(qubit_wise=True)

    # calculate the actual # of shots consumed in each iteration of VQE for this problem instance
    total_shots = len(pauli_lists) * n_shots

    # obtain classical solution as a reference
    classical_result = NumPyMinimumEigensolver().compute_minimum_eigenvalue(operator=pauli_operator)
    ref_value = classical_result.eigenvalue.real

    # also get the corresponding eigenstate
    ref_eigenstate = classical_result.eigenstate

    # prepare warm-start from classical solution/approximation by inverse power method
    shift = get_gershgorin_extrema(operator_matrix)[0]
    try:
        approx_eigenstate = inverse_power_method(operator_matrix, shift=shift, num_iterations=approx_iterations)
    except:
        # avoid rare cases of non-invertible matrices by generating a new one
        print('\033[91m'+"Matrix not invertible, generating a new one."+'\033[0m')
        return generate_problem_instance(approx_iterations=approx_iterations)

    return operator_matrix, pauli_operator, pauli_lists, total_shots, ref_value, ref_eigenstate, approx_eigenstate

def run_comparison(a_seed=a_seed, plot=False):
    operator_matrix, pauli_operator, pauli_lists, total_shots, ref_value, ref_eigenstate, approx_eigenstate = generate_problem_instance(seed=a_seed)

    normalized = approx_eigenstate/np.sqrt(np.sum([np.abs(x)**2 for x in approx_eigenstate]))
    approx_eigenstate_amplitudes = normalized.T.tolist()[0]
    approx_ratio = eval_energy(approx_eigenstate_amplitudes, pauli_operator)/ref_value

    # print some values
    print("pauli decomposition", len(pauli_operator.paulis), pauli_operator.paulis)
    print("non-commuting paulis", len(pauli_lists), pauli_lists)
    print(f"Reference value: {ref_value:.5f}")
    print(f"ref_eigenstate: {ref_eigenstate}")
    print(f"Amplitudes: {approx_eigenstate_amplitudes}")
    print(f"Approximation ratio: {approx_ratio}")

    pretrain = True
    if pretrain:
        # ACAE pretraining
        pretrain_opt_res = approx_encode(circuit_ansatz, approx_eigenstate_amplitudes, backend=Aer.get_backend("statevector_simulator"), n_shadows=n_shadows, fixed_unitaries=True)
        print("Pre-Training completed", pretrain_opt_res.bestValue)
        est_fidelity = pretrain_opt_res.bestValue[1]
        pretrained_params = pretrain_opt_res.bestValue[0]
    else:
        pretrained_params, pretrain_opt_res = None, None

    rand_init_params = list(np.random.uniform(-np.pi, np.pi, num_parameters))

    # define different variants of the (WS-)VQE to be executed
    variants = [
                {"title": "VQE randomly initialized parameters", "active": True, "kwargs": {"initial_params": rand_init_params}},
                {"title": "VQE randomly initialized parameters", "active": True, "kwargs": {"initial_params": rand_init_params, "optimizer": optimizer2}},
                {"title": "VQE zero initialized parameters", "active": False, "kwargs": {}},
                {"title": "WS-VQE randomly initialized parameters", "active": False, "kwargs": {"initial_params": rand_init_params, "initial_point": approx_eigenstate_amplitudes}},
                {"title": "WS-VQE (biased initial state, approx. solution) zero initialized parameters", "active": False, "kwargs": {"initial_point": approx_eigenstate_amplitudes, "optimizer": optimizer1}},
                {"title": "WS-VQE (biased initial state, classic solution) zero initialized parameters", "active": False, "kwargs": {"initial_point": list(ref_eigenstate), "optimizer": optimizer1}},
                {"title": "WS-VQE pre-trained parameters, same step size", "active": pretrain, "pretrain_path": pretrain_opt_res.optimizationPath if pretrain_opt_res else None, "kwargs": {"initial_params": pretrained_params}},
                {"title": "WS-VQE pre-trained parameters, half step size", "active": pretrain, "pretrain_path": pretrain_opt_res.optimizationPath if pretrain_opt_res else None, "kwargs": {"initial_params": pretrained_params, "optimizer": optimizer2}},
                {"title": "WS-VQE pre-trained parameters, dynamic step size", "active": pretrain, "pretrain_path": pretrain_opt_res.optimizationPath if pretrain_opt_res else None, "kwargs": {"initial_params": pretrained_params, "optimizer": COBYLA(maxiter=iterations, rhobeg=-1/est_fidelity*3*np.pi/32)}}
            ]

    # (only variants with "active"-flag set to True are executed)
    variants = [var for var in variants if var["active"]]
    for variant in variants:
        # set default optimizer
        if not "optimizer" in variant["kwargs"].keys():
            variant["kwargs"]["optimizer"] = optimizer

        # execute variant and add results to dict
        print(f"Running variant '{variant['title']}'")
        (counts, values), result = run_vqe(circuit_ansatz, pauli_operator, **variant["kwargs"])

        # add offset to VQE iterations to account for ACAE pretraining
        if "pretrain_path" in variant.keys():
            offset = np.ceil( (len(variant["pretrain_path"]) -1) * n_shadows/(total_shots) )
            counts = list(np.array(counts) + offset)
        print(f"Result value: {values[-1]}")

        # compute values as approx. ratios and store as dict
        variant["values"] = {int(counter): value/ref_value for counter, value in zip(counts, values)}
        variant["result"] = result

    # plot
    if plot:
        fig = plt.figure(1, figsize=(20, 7))
        ax1 = fig.add_subplot(1, 1, 1)
        if pretrain:
            #x_values = np.arange(0, len(variant["pretrain_path"])*n_shadows/total_shots, n_shadows/total_shots)
            x_values = np.linspace(0, len(variant["pretrain_path"])*n_shadows/total_shots, len(variant["pretrain_path"]))
            y_values = -1*np.array(variant["pretrain_path"], dtype=object)[:,1]
            ax1.plot(x_values, y_values, label="Encoding pre-training (fidelity)", linestyle="dotted")
            ax1.axhline(y=approx_ratio, linestyle="-", label="approx. ratio")
        for variant in variants:
            if not variant["active"]:
                continue
            ax1.plot(variant["values"].keys(), variant["values"].values(), label=variant["title"])

        plt.subplots_adjust(hspace=0.23, wspace=0.13)
        plt.legend()
        plt.show()

    return operator_matrix, ref_value, ref_eigenstate, approx_eigenstate_amplitudes, approx_ratio, variants

def run_vqe(circuit_ansatz, pauli_operator, optimizer, initial_params=[0 for i in range(num_parameters)], initial_point=None, shots=n_shots):
    """Execute VQE according to the configuration provided (including initial state/parameter initialization for optional warm-starting)"""

    circuit = QuantumCircuit(problem_size)
    if initial_point is not None:
        circuit.initialize(initial_point, circuit.qubits)
    ansatz = circuit.compose(circuit_ansatz)

    counts = []
    values = []
    def store_intermediate_result(eval_count, parameters, mean, std):
        counts.append(eval_count)
        values.append(mean)

    noiseless_estimator = AerEstimator(run_options={"seed": b_seed, "shots": shots}, transpile_options={"seed_transpiler": b_seed},)
    vqe = VQE(noiseless_estimator, ansatz, optimizer=optimizer, callback=store_intermediate_result, initial_point=initial_params)
    result = vqe.compute_minimum_eigenvalue(operator=pauli_operator)

    print(f"VQE on Aer qasm simulator (no noise): {result.eigenvalue.real:.5f}")

    return (counts, values), result

def eval_energy(state, operator):
    """Uses qiskit's VQE implementation to evaluate the expectation value of an operator for a given state."""

    # first generate a (dummy) ansatz that just prepares the state
    ansatz = QuantumCircuit(int(np.log2(len(state))))
    ansatz.initialize(state)

    # ansatz needs to have at least one parameter
    parameter = Parameter("dummy")
    ansatz.ry(parameter, 0)

    # get the evaluate energy function from qiskit#s VQE implementation
    noiseless_estimator = AerEstimator(run_options={"seed": 1, "shots": 1024}, transpile_options={"seed_transpiler": 1},)
    vqe = VQE(noiseless_estimator, ansatz, optimizer=None, initial_point=[])
    qiskit_eval_energy = vqe._get_evaluate_energy(ansatz, operator)

    # evaluate for dummy parameter set to 0, i.e., the dummy parameter doesn't change anything
    result = qiskit_eval_energy([0])
    return result


def run_bulk_experiment(n_problems):
    """generate n_problems problem instances and run the comparison for each of them. Results are stored in file fname."""

    for i in range(n_problems):
        print(f"Iteration #{a_seed}")
        problem_instance, ref_eigenvalue, ref_eigenstate, approx_eigenstate, approx_ratio, _variants = \
            store_results(fname, *run_comparison(a_seed=a_seed, plot=False))


def evaluate_bulk_experiments():
    """evaluate results stored in file fname"""

    collect = np.array([20,40,60,80,100])-1
    results = load_results(fname)
    variants = None

    approx_ratios = [r["approx. ratio"] for r in results]
    print("approx. ratios", approx_ratios)
    pretrain_fidelities = []
    optimizer_rhobeg = []
    for r in results:
        found = False
        for v in r["results"]:
            if "pretrain_path" in v.keys():
                if not found:
                    pretrain_fidelities.append(min([v["pretrain_path"][i][1] for i in range(len(v["pretrain_path"]))]))
                    found = True
                optimizer_rhobeg.append(v["kwargs"]["optimizer"]["options"]["rhobeg"])
    print("pretrain_fidelities", pretrain_fidelities)
    n_rhobegs = int(len(optimizer_rhobeg)/len(pretrain_fidelities))
    [print("optimizer_rhobeg", optimizer_rhobeg[i::n_rhobegs]) for i in range(n_rhobegs)]

    for _variants in [tuple(r.values())[-1] for r in results]:
        variants = _variants if variants is None else variants
        for i in collect:
            for k, _variant in enumerate(_variants):
                if not _variant["active"]:
                    continue
                # collect values at position i, but if ith itereation was not reached, take the final value
                if not i in _variant["values"].keys() and i<max(_variant["values"].keys()):
                    continue
                value = _variant["values"][i] if i in _variant["values"].keys() else _variant["values"][max(_variant["values"].keys())]
                if "progress" in variants[k].keys():
                    if i in variants[k]["progress"].keys():
                        variants[k]["progress"][i].append(value)
                    else:
                        variants[k]["progress"][i] = [value]
                else:
                    variants[k]["progress"] = {i: [value]}

        for i in range(collect[0], iterations):
            for k, _variant in enumerate(_variants):
                if not _variant["active"]:
                    continue
                value = _variant["values"][i]
                if "medians" in variants[k].keys():
                    if i in variants[k]["medians"].keys():
                        variants[k]["medians"][i].append(value)
                    else:
                        variants[k]["medians"][i] = [value]
                else:
                    variants[k]["medians"] = {i: [value]}

    for variant in variants:
        for k in variant["medians"].keys():
            variant["medians"][k] = np.median(variant["medians"][k])
        print(variant["title"], "final median", list(variant["medians"].values())[-1])

    for i in collect:
        print(i)
        for variant in variants:
            print(f"{variant['title']}: {len(variant['progress'][i])}")

    min_v, max_v = 0, 0
    for variant in variants:
        if not variant["active"]:
            continue
        if np.min(sum(list(variant["progress"].values()), []))<min_v:
            min_v=np.min(list(variant["progress"].values()))
        if np.max(sum(list(variant["progress"].values()), []))>max_v:
            max_v=np.max(list(variant["progress"].values()))

    fig = plt.figure(figsize =(10, 10))
    fig.tight_layout()
    n_plots = np.sum([1 if variant["active"] else 0 for variant in variants])
    dimensions = int(np.ceil(np.sqrt(n_plots))) # arrange plots in dim x dim square
    shape = (dimensions+1, dimensions)
    gs = GridSpec(*shape, figure=fig)

    current_plot = 0
    ax_summary = fig.add_subplot(gs[dimensions, :])
    for variant in variants:
        if not variant["active"]:
            continue

        ax = fig.add_subplot(gs[current_plot//dimensions, current_plot % dimensions])
        ax.boxplot(list(variant["progress"].values()), positions=collect+1, widths=(collect[1]-collect[0])//2)
        ax.axhline(y=np.average(approx_ratios), linestyle="-", label="avg. classical approx. ratio")
        #ax.plot(list(variant["medians"].keys()), list(variant["medians"].values()))
        ax.set_title(variant["title"])
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Approximation Ratio")
        ax.set_ylim(min_v, max_v)
        ax_summary.plot(np.array(list(variant["medians"].keys()))+1, list(variant["medians"].values()), label=variant["title"])
        current_plot += 1

    ax_summary.legend()
    plt.legend()
    plt.subplots_adjust(hspace=0.33, wspace=0.23)
    #plt.show()

    if True: # plot summary separately
        ieee_fontsize = "8"
        mpl.rcParams['text.usetex'] = True
        mpl.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}' #for \text command
        mpl.rcParams["font.family"] = "Times New Roman"

        nice_titles = [r"VQE/$\varrho_\text{VQE}$",
                       r"VQE/$\varrho_\text{WS-VQE}^\text{static}$",
                       r"WS-VQE/$\varrho_\text{VQE}$",
                       r"WS-VQE/$\varrho_\text{WS-VQE}^\text{static}$",
                       r"WS-VQE/$\varrho_\text{WS-VQE}^\text{dynamic}$"]
        markers = ["o", "D", "^", "s", "."]
        linestyles = ["-", "--", "-.", "dotted", "dotted"]
        cmap = mpl.colormaps["Blues"]
        n_lines = len(variants)
        colors = cmap(np.linspace(0.25, 1, n_lines))

        cm = 1/2.54  # centimeters in inches
        fig = plt.figure(figsize =(8.89*cm, 7*cm))
        ax = fig.add_subplot()

        for variant, title, marker, linestyle, color in zip(variants, nice_titles, markers, linestyles, colors):
            ax.plot(np.array(list(variant["medians"].keys()))+1, list(variant["medians"].values()), label=title,
                    marker="None", linestyle="-", markerfacecolor='white', markersize=4, color=color)

        #ax.set_yscale("logit")
        ax.set_xlabel("Iteration", fontsize=ieee_fontsize)
        ax.set_ylabel("Median approx. ratio", fontsize=ieee_fontsize)

        # xticks = ax.get_xticks()
        # ax.set_xticks(collect+1, collect, rotation="vertical")
        yticks = np.round(list(np.arange(0.3,0.95,0.2)),1)
        ax.set_yticks(yticks, yticks, fontsize=ieee_fontsize)
        xticks = collect+1
        ax.set_xticks(xticks, xticks, fontsize=ieee_fontsize)

        handles, labels = plt.gca().get_legend_handles_labels()
        order = np.array(range(n_lines))[::-1]
        plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order], loc="best", fontsize=ieee_fontsize)

        plt.tight_layout()

        plt.show()

def run_single_experiment(a_seed=a_seed, store=False):
    """Run one comparison for one problem instance and plot the optimization progress"""

    problem_instance, ref_eigenvalue, ref_eigenstate, approx_eigenstate, approx_ratio, variants = \
        run_comparison(a_seed=a_seed, plot=True)
    if store:
        store_results(fname, problem_instance, ref_eigenvalue, ref_eigenstate, approx_eigenstate, approx_ratio, variants)


#run_single_experiment(store=False)
#run_bulk_experiment(n_problems=25)
evaluate_bulk_experiments()
