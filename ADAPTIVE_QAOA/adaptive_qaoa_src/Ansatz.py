from itertools import product

import numpy as np
import qiskit

from src.SYMQCircuit import *
from src.Tools import _get_state_probabilities_
from ADAPTIVE_QAOA.adaptive_qaoa_src.ADAPTIVEQAOATools import *

import qiskit
from qiskit import QuantumCircuit, execute
from qiskit import BasicAer


class ADAPTIVEQAOAansatz:
    def __init__(self, n_qubits: int, w_edges: List[Tuple[int, int, float]], precision: int = 64,
                 backend: str = "SYMQ"):

        __backends__ = ["SYMQ", "QISKIT"]
        if backend not in __backends__:
            raise ValueError(f"Backend should be either one of: {__backends__}.")
        if not isinstance(n_qubits, int):
            raise ValueError(f"n_qubits should be integer.")
        if n_qubits <= 0:
            raise ValueError(f'n_qubits should be > 0, but is: {n_qubits}.')

        self.backend = backend
        self.n_qubits = n_qubits
        self.w_edges = w_edges
        self.precision = precision

        # --------- QUBO --------- #
        self.QUBO_mat = get_qubo(size=self.n_qubits, edges=self.w_edges)
        QUBO_dict = {}
        for row in range(self.QUBO_mat.shape[0]):
            for col in range(self.QUBO_mat.shape[1]):
                if self.QUBO_mat[row, col] != 0.0:
                    QUBO_dict[(row, col)] = self.QUBO_mat[row, col]

        # --------- Ising --------- #
        self.ISING_vec = np.zeros(shape=(self.n_qubits, 1), dtype=float)
        self.ISING_mat = np.zeros(shape=(self.n_qubits, self.n_qubits), dtype=float)
        h_dict, J_dict, offset = qubo_to_ising(Q=QUBO_dict)
        self.h_list = [(key, h_dict[key]) for key in h_dict.keys()]
        self.J_list = [(key[0], key[1], J_dict[key]) for key in J_dict.keys()]
        for i, val in self.h_list:
            self.ISING_vec[i] = val
        for i, j, val in self.J_list:
            self.ISING_mat[i, j] = val

        # ---------- Algorithm ----------- #
        self.current_circuit = SYMQCircuit(nr_qubits=self.n_qubits,precision=self.precision)
        self.cost_hamilton_matrix = self.get_cost_hamiltonian()
        self.mixer_gate_pool = self.get_gate_pool(kind='SINGLE')

    def set_circuit(self, theta: List[float]) -> None:
        """
        Construct a QAOA circuit with weighted terms.

        Parameters:
        -----------
        n_qubits : int
            Number of qubits in the circuit.
        edges : Tuple[List[Tuple[int, int, float]], List[Tuple[int, float]]]
            Tuple containing lists of edge information: (J, h).
            J: List of tuples (qubit_i, qubit_j, weight) representing weighted edges.
            h: List of tuples (qubit_i, weight) representing weighted terms.
        theta : List[float]
            List of optimization parameters.

        Returns:
        --------
        SYMQCircuit
            Constructed QAOA circuit with weighted terms.

        Notes:
        ------
        This function constructs a QAOA circuit with weighted terms for the optimization problem.
        The QAOA circuit includes alternating cost and mixer unitaries based on the given parameters.
        """
        # Number of alternating (Cost,Mixer) unitaries
        p = len(theta) // 2
        # print(theta)

        # Gamma opt param for cost unitaries as last p vals.
        gamma = theta[p:]

        # Beta opt param for mixing unitaries as first p vals.
        beta = theta[:p]

        # Initial_state: Hadamard gate on each qbit
        for qubit_index in range(self.n_qubits):
            self.current_circuit.add_h(target_qubit=qubit_index)

        # For each Cost,Mixer repetition
        for irep in range(0, p):

            # ------ Cost unitary: ------ #
            # Weighted RZZ gate for each edge
            for qubit_i, qubit_j, weight in self.J_list:
                angle = 2 * gamma[irep] * weight
                # print(f"Layer: {irep}, RZZ angle: {angle}, gamma: {gamma[irep]}, weight: {weight}")
                self.current_circuit.add_rzz(qubit_1=qubit_i, qubit_2=qubit_j, angle=angle)
            # Weighted RZ gate for each qubit
            for qubit_i, weight in self.h_list:
                angle = 2 * gamma[irep]
                # print(f"Layer: {irep}, RZ angle: {angle}, gamma: {gamma[irep]}, weight: {weight}")
                self.current_circuit.add_rz(target_qubit=qubit_i, angle=angle)

            # ------ Mixer unitary: ------ #
            # Get e^{-i*H_c*gamma}|psi_p>
            state_vector = self.current_circuit.get_state_vector().reshape((self.n_qubits**2, 1))
            # Initialize max expectation
            max_expectation, best_mixer = 0.0, None
            # Get gate pool
            for gates in self.mixer_gate_pool:
                mixer = np.eye(self.n_qubits**2, dtype={64: np.complex64, 128: np.complex128}[self.precision])
                for qubits, pauli_operators in gates.items():
                    # 2-qubit Pauli string
                    if isinstance(pauli_operators, tuple):
                        operator_1, operator_2 = pauli_operators
                        qubit_1, qubit_2 = qubits
                        mixer += self.get_mixer_hamiltonian(pauli_operator=operator_1, target_qubit=qubit_1)
                        mixer = mixer @ self.get_mixer_hamiltonian(pauli_operator=operator_2, target_qubit=qubit_2)
                    # 1-qubit Pauli string
                    else:
                        operator = pauli_operators
                        qubit = qubits
                        mixer += self.get_mixer_hamiltonian(pauli_operator=operator, target_qubit=qubit)
                # Calculate expectation
                commutator = self.get_commutator(A=self.cost_hamilton_matrix, B=mixer)
                expectation = np.abs(1j*(state_vector.T @ (commutator @ state_vector))[0, 0])
                # Compare expectation
                if expectation > max_expectation:
                    max_expectation = expectation
                    best_mixer = gates

            # Set mixer
            for qubits, pauli_operators in best_mixer.items():

                # N.B. works for both 1 and 2-qubit pauli strings
                pauli_string = ''.join(pauli_operators)
                angle = 2 * beta[irep]
                self.current_circuit.add_exp_of_pauli_string(pauli_string=pauli_string,theta=angle)


    @staticmethod
    def get_commutator(A: np.ndarray, B: np.ndarray):
        return np.matmul(A, B) - np.matmul(B, A)

    def set_QISKIT_circuit(self, theta: List[float]):

        # Number of alternating (Cost,Mixer) unitaries
        p = len(theta) // 2

        # Initializing Q circuit
        qc = QuantumCircuit(self.n_qubits)

        # Gamma opt param for cost unitaries as last p vals.
        gamma = theta[p:]

        # Beta opt param for mixing unitaries as first p vals.
        beta = theta[:p]

        # Initial_state: Hadamard gate on each qbit
        for qubit_index in range(self.n_qubits):
            qc.h(qubit=qubit_index)

        # For each Cost,Mixer repetition
        for irep in range(0, p):

            # ------ Cost unitary: ------ #
            # Weighted RZZ gate for each edge
            for qubit_i, qubit_j, weight in self.J_list:
                angle = 2 * gamma[irep] * weight
                qc.rzz(qubit1=qubit_i, qubit2=qubit_j, theta=angle)
            # Weighted RZ gate for each qubit
            for qubit_i, weight in self.h_list:
                angle = 2 * gamma[irep]
                qc.rz(qubit=qubit_i, phi=angle)

            # ------ Mixer unitary: ------ #
            # Mixer unitary: Weighted X rotation on each qubit
            for qubit_i, weight in self.h_list:
                angle = 2 * beta[irep]
                qc.rx(qubit=qubit_i, theta=angle)

        self.current_circuit = qc

    def get_gate_pool(self, kind: str = 'SINGLE') -> Union[
        List[Dict[int, str]], List[Dict[Tuple[int, int], Tuple[str, str]]]]:
        r"""
            Generate a gate pool for a quantum circuit, based on the specified 'kind'.

            This function generates a pool of quantum gates that can be used for constructing
            quantum circuits. The gate pool is tailored according to the specified 'kind' parameter.

            Parameters:
                kind (str, optional): The type of gate pool to generate. Can be one of the following:
                    - 'SINGLE': Single-qubit gate pool.
                    - 'MULTI': Multi-qubit gate pool.

            Returns:
                Union[List[Dict[int, str]], List[Dict[Tuple[int, int], Tuple[str, str]]]]:
                A list of dictionaries representing the gate pool based on the specified 'kind'.

            Raises:
                ValueError: If the provided 'kind' is not one of ['QAOA', 'SINGLE', 'MULTI'].

            Note:
                - 'QAOA' pool scales like O(1)
                - 'SINGLE' pool scales like O(n)
                - 'MULTI' pool scales like O(n^2) (specifically (3n)^2-2n+2).
            """

        if not isinstance(kind, str) or kind not in ['SINGLE', 'MULTI']:
            raise ValueError('Kind should be str, and one of: ', ['SINGLE', 'MULTI'])
        native_gates = {'X': 'X', 'Y': 'Y', 'Z': 'Z'}

        QAOA_pool = [{qubit: native_gates['X'] for qubit in range(self.n_qubits)}]
        SINGLE_qubit_pool = ([{qubit: native_gates['Y'] for qubit in range(self.n_qubits)}]
                             + [{qubit: native_gates['X']} for qubit in range(self.n_qubits)] + QAOA_pool)

        qubit_pairs = list(product([qubit for qubit in range(self.n_qubits)], repeat=2))
        gate_pairs = list(product(native_gates, repeat=2))

        MULTI_qubit_pool = []
        for gate_pair in gate_pairs:
            for qubit_1, qubit_2 in qubit_pairs:
                # N.B. X^2,Y^2,Z^2=1 (pauli operators are idempotent)
                if not (qubit_1 == qubit_2 and gate_pair[0] == gate_pair[1]):
                    MULTI_qubit_pool.append({(qubit_1, qubit_2): gate_pair})
        MULTI_qubit_pool += SINGLE_qubit_pool

        return {'SINGLE': SINGLE_qubit_pool, 'MULTI': MULTI_qubit_pool}[kind]

    def get_cost_hamiltonian(self):
        """ Getting the matrix representation of the cost hamiltonian H_c."""

        if self.backend == "SYMQ":

            initial_matrix = SYMQCircuit(nr_qubits=self.n_qubits, precision=self.precision).get_circuit_unitary()
            # ------ Cost unitary: ------ #
            # ZZ gate for each edge
            for qubit_i, qubit_j, weight in self.J_list:
                # Initializing Q circuit
                qcircuit = SYMQCircuit(nr_qubits=self.n_qubits, precision=self.precision)
                qcircuit.add_z(target_qubit=qubit_i)
                qcircuit.add_z(target_qubit=qubit_j)
                initial_matrix += qcircuit.get_circuit_unitary()
            # Z gate for each qubit
            for qubit_i, weight in self.h_list:
                qcircuit = SYMQCircuit(nr_qubits=self.n_qubits, precision=self.precision)
                qcircuit.add_z(target_qubit=qubit_i)
                initial_matrix += qcircuit.get_circuit_unitary()
            return initial_matrix

        else:
            # TODO: implement QISKIT backend
            raise ValueError("Cost Hamiltonian hasn't implemented QISKIT backend yet...")
            # Initializing Q circuit
            qcircuit = QuantumCircuit(self.n_qubits)

            # ------ Cost unitary: ------ #
            # ZZ gate for each edge
            for qubit_i, qubit_j, weight in self.J_list:
                qcircuit.z(qubit=qubit_i)
                qcircuit.z(qubit=qubit_j)
            # Z gate for each qubit
            for qubit_i, weight in self.h_list:
                qcircuit.z(qubit=qubit_i)
            return qiskit.quantum_info.Operator(qcircuit).data

    def get_mixer_hamiltonian(self, pauli_operator: str, target_qubit: int):
        """ Getting the matrix representation of the mixer hamiltonian H_c (as chosen from gate pool.)"""

        if not isinstance(pauli_operator, str):
            raise ValueError("Pauli operator should be string type.")
        if pauli_operator not in ['X', 'Y', 'Z']:
            raise ValueError("Pauli operator should be in: ", ['X', 'Y', 'Z'])

        if self.backend == "SYMQ":
            # Initializing Q circuit
            qcircuit = SYMQCircuit(nr_qubits=self.n_qubits, precision=self.precision)
            # Adding gate
            if pauli_operator == 'X':
                qcircuit.add_x(target_qubit=target_qubit)
            elif pauli_operator == 'Y':
                qcircuit.add_y(target_qubit=target_qubit)
            else:
                qcircuit.add_z(target_qubit=target_qubit)
            return qcircuit.get_circuit_unitary()

        else:
            # Initializing Q circuit
            qcircuit = QuantumCircuit(self.n_qubits)
            # Adding gate
            if pauli_operator == 'X':
                qcircuit.x(qubit=target_qubit)
            elif pauli_operator == 'Y':
                qcircuit.y(qubit=target_qubit)
            else:
                qcircuit.z(qubit=target_qubit)
            return qiskit.quantum_info.Operator(qcircuit).data

    def Ising_cost(self, state: np.ndarray) -> float:
        """
        Calculate the Ising cost for a given spin configuration.

        Parameters:
        state (np.ndarray): Array representing the spin configuration.

        Returns:
        float: The computed Ising cost.
        """
        return float(state.T @ (self.ISING_mat @ state) + state.T @ self.ISING_vec)

    def QUBO_cost(self, state: np.ndarray, offset: float = 0.0) -> float:
        """
        Calculate the Ising cost for a given spin configuration.

        Parameters:
        state (np.ndarray): Array representing the spin configuration.

        Returns:
        float: The computed Ising cost.
        """
        return float(state.T @ (self.QUBO_mat @ state)) + offset

    def compute_expectation(self, counts: dict) -> float:
        """
        Compute the expectation value based on measurement results.

        Parameters:
        counts (dict): A dictionary containing bitstring measurement outcomes as keys and their corresponding
        probabilities as values.

        Returns:
        float: The computed expectation value.
        """
        _state_ = np.zeros_like(self.ISING_vec)
        _result_ = 0.0
        for bitstring, probability in counts.items():
            _state_ = np.array(list(bitstring)).reshape((self.n_qubits, 1)).astype(int)
            # N.B. Q has been multiplied with -1, therefore we are doing += instead of -= here.
            _result_ += self.QUBO_cost(state=_state_) * probability
        return _result_

    def evaluate_circuit(self, theta: List[float]):
        """
        Execute a quantum circuit with the given parameters and compute the expectation value.

        Parameters:
        theta (List[float]): List of parameters for the quantum circuit.

        Returns:
        float: The computed expectation value.
        """

        if self.backend == "SYMQ":
            self.set_circuit(theta=theta)
            prob_distribution = self.current_circuit.get_state_probabilities(reverse_states=True)
        else:
            self.set_QISKIT_circuit(theta=theta)
            state_vector = execute(self.current_circuit,
                                   BasicAer.get_backend("statevector_simulator")).result().get_statevector()
            prob_distribution = _get_state_probabilities_(state_vector, reverse_states=True)
        return self.compute_expectation(counts=prob_distribution)
