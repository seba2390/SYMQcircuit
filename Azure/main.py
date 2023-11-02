from typing import *
from itertools import combinations

import numpy as np
import scipy as sc

from src.Ansatz import CPQAOAansatz
from src.cp_QAOATools import qubo_to_ising


def ising_min_cost_partition(nr_qubits: int,
                             k: int,
                             J_mat: np.ndarray,
                             h_vector: np.ndarray,
                             offset: float) -> Tuple[float, float, np.ndarray]:
    def generate_ising_string_permutations(n: int, k: int) -> np.ndarray:

        def to_ising_state(qubo_state: np.ndarray) -> np.ndarray:
            return 2 * qubo_state - 1

        num_permutations = 2 ** n
        # Generate all combinations of k positions out of N
        for indices in combinations(range(n), k):
            # Create a numpy array of zeros of size N
            arr = np.zeros(n, dtype=int)
            # Set ones at the specified positions
            arr[list(indices)] = 1
            yield to_ising_state(qubo_state=arr)

    def cost(state: np.ndarray, _J_mat: np.ndarray, _h_vec: np.ndarray, offset: float) -> float:
        return np.dot(state, np.dot(_J_mat, state)) + np.dot(state, _h_vec) + offset

    max_cost, min_cost, min_perm = -np.inf, np.inf, np.empty(shape=(nr_qubits,))
    for perm in generate_ising_string_permutations(n=nr_qubits, k=k):
        perm_cost = cost(state=perm, _J_mat=J_mat, _h_vec=h_vector, offset=offset)
        if perm_cost < min_cost:
            min_cost, min_perm = perm_cost, perm
        if perm_cost > max_cost:
            max_cost = perm_cost

    binary = min_perm / 2 + 1
    return max_cost, min_cost, binary.astype(int)


def get_ising(mu: np.ndarray, sigma: np.ndarray, alpha: float) -> tuple[np.ndarray, np.ndarray, float]:
    if not len(mu) == sigma.shape[0] == sigma.shape[1]:
        raise ValueError(f'Dimensions of mu and sigma are note appropriate.')
    _Q_ = np.zeros_like(sigma)
    for i in range(sigma.shape[0]):
        for j in range(sigma.shape[1]):
            if i == j:
                _Q_[i, j] += -mu[j] + alpha * sigma[j, j]
            else:
                _Q_[i, j] += alpha * sigma[i, j]
    _Q_dict = {}
    for i in range(_Q_.shape[0]):
        for j in range(_Q_.shape[1]):
            _Q_dict[(i, j)] = _Q_[i, j]

    _h_dict, _J_dict, _offset_ = qubo_to_ising(_Q_dict)
    h, J = np.zeros_like(mu), np.zeros_like(sigma)
    for key in _h_dict.keys():
        h[key] = _h_dict[key]
    for key in _J_dict.keys():
        i, j = key
        J[i, j] = _J_dict[key]
    return J, h, _offset_


file_id = 0
alpha = 0.001
N_seeds = 10
max_iter = 10000
for N in range(3, 11):
    print(f'=== N === : {N} / 8')
    for k in range(1, N):
        result = {}
        for layers in range(1, 6):
            print(f'Layer: {layers}')
            normed_c_vals = []
            max_iter_reached = 0
            for seed in range(N_seeds):
                print(f'seed: {seed}')
                np.random.seed(seed)

                expected_returns = np.random.uniform(low=0, high=0.95, size=N)

                _temp_ = np.random.uniform(low=0, high=1, size=(N, N))
                covariances = np.dot(_temp_, _temp_.transpose())
                if not np.alltrue(covariances == covariances.T) or not np.alltrue(np.linalg.eigvals(covariances) >= 0):
                    raise ValueError('Covariance matrix is not PSD.')

                J, h, offset = get_ising(mu=expected_returns, sigma=covariances, alpha=alpha)
                max_cost, min_cost, min_state = ising_min_cost_partition(nr_qubits=N, k=k, J_mat=J, h_vector=h, offset=offset)

                # Size of circuit
                n_qubits = N

                # Defining instance of QAOA ansatz
                QAOA_objective = CPQAOAansatz(n_qubits=n_qubits,
                                              n_layers=layers,
                                              w_edges=None,
                                              cardinality=k,
                                              precision=64)
                QAOA_objective.set_ising_model(J=J, h=h, offset=offset)

                # Initial guess for parameters (gamma, beta) of circuit
                theta_min, theta_max = -np.pi, np.pi
                gamma_i = np.random.uniform(low=theta_min, high=theta_max, size=QAOA_objective.nr_cost_terms * layers).tolist()
                beta_i = np.random.uniform(low=theta_min, high=theta_max, size=(QAOA_objective.n_qubits - 1) * layers).tolist()
                theta_i = gamma_i + beta_i

                # ------ Optimizer run ------ #
                _available_methods_ = ['Nelder-Mead', 'Powell', 'COBYLA', 'trust-constr']
                res = sc.optimize.minimize(fun=QAOA_objective.evaluate_circuit, x0=theta_i, method=_available_methods_[2],
                                           options={'disp': False, 'maxiter': max_iter})

                # Final parameters (beta, gamma) for circuit
                theta_f = res.x.tolist()
                c = res.fun

                normed_c_vals.append(1 / (max_cost - min_cost) * c - 1 / (max_cost / min_cost - 1))
                if res['message'] == 'Maximum number of function evaluations has been exceeded.':
                    max_iter_reached += 1 / N_seeds

            result[layers] = (np.mean(normed_c_vals), np.std(normed_c_vals), max_iter_reached)


        def save_run(fname: str, description: str, results: dict[int, tuple[float, float, float]]) -> None:
            with open(file=fname + '.txt', mode='w') as output_file:
                output_file.write(' ## Description: \n ' + description + '\n\n')
                output_file.write(
                    '|| --- Num. layers --- || --- Avg. --- || --- Std. dev --- || --- Prc. max iter. reached --- || \n')
                for num_layers in list(results.keys()):
                    __str__ = '           ' + str(num_layers) + '        '
                    __str__ += '        ' + str(np.round(results[num_layers][0], 5)) + '      '
                    __str__ += '        ' + str(np.round(results[num_layers][1], 5)) + '          '
                    __str__ += '        ' + str(np.round(results[num_layers][2], 5)) + '        ' + '\n'
                    output_file.write(__str__)
            output_file.close()


        desc = (f'N={N}, k={k}, alpha={alpha}, avg. over {N_seeds} runs. Only Nearest Neighbor mixer w. no Z w. "k" first set, and COBYLA max iter set to {max_iter}')
        fname = f'run{file_id}'
        save_run(fname=fname, description=desc, results=result)
        file_id += 1
