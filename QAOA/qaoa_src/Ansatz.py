import numpy as np

from src.SYMQCircuit import *
from QAOA.qaoa_src.QAOATools import *

from qiskit import QuantumCircuit, execute, transpile
from qiskit.visualization import plot_histogram
from qiskit import Aer


class QAOAansatz:
    def __init__(self, n_qubits: int, w_edges: List[Tuple[int, int, float]], precision: int = 64, backend: str = "SYMQ"):

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
        self.h_vec = np.zeros(shape=(self.n_qubits, 1), dtype=float)
        self.J_mat = np.zeros(shape=(self.n_qubits, self.n_qubits), dtype=float)
        h_dict, J_dict, offset = qubo_to_ising(Q=QUBO_dict)
        self.h_list = [(key, h_dict[key]) for key in h_dict.keys()]
        self.J_list = [(key[0], key[1], J_dict[key]) for key in J_dict.keys()]
        for i, val in self.h_list:
            self.h_vec[i] = val
        for i, j, val in self.J_list:
            self.J_mat[i, j] = val

    def set_circuit(self, theta: List[float]) -> SYMQCircuit:
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

        # Initializing Q circuit
        qcircuit = SYMQCircuit(nr_qubits=self.n_qubits, precision=64)

        # Gamma opt param for cost unitaries as last p vals.
        gamma = theta[p:]

        # Beta opt param for mixing unitaries as first p vals.
        beta = theta[:p]

        # Initial_state: Hadamard gate on each qbit
        for qubit_index in range(self.n_qubits):
            qcircuit.add_h(target_qubit=qubit_index)

        # For each Cost,Mixer repetition
        for irep in range(0, p):

            # ------ Cost unitary: ------ #
            # Weighted RZZ gate for each edge
            for qubit_i, qubit_j, weight in self.J_list:
                angle = 2 * gamma[irep] * weight
                #print(f"Layer: {irep}, RZZ angle: {angle}, gamma: {gamma[irep]}, weight: {weight}")
                qcircuit.add_rzz(qubit_1=qubit_i, qubit_2=qubit_j, angle=angle)
            # Weighted RZ gate for each qubit
            for qubit_i, weight in self.h_list:
                angle = 2 * gamma[irep]
                #print(f"Layer: {irep}, RZ angle: {angle}, gamma: {gamma[irep]}, weight: {weight}")
                qcircuit.add_rz(target_qubit=qubit_i, angle=angle)

            # ------ Mixer unitary: ------ #
            # Mixer unitary: Weighted X rotation on each qubit
            for qubit_i, weight in self.h_list:
                angle = 2 * beta[irep]
                #print(f"Layer: {irep}, RX angle: {angle}, beta: {beta[irep]}, weight: {weight}")
                qcircuit.add_rx(target_qubit=qubit_i, angle=angle)

        return qcircuit

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

        return qc

    def Ising_cost(self, state: np.ndarray) -> float:
        """
        Calculate the Ising cost for a given spin configuration.

        Parameters:
        state (np.ndarray): Array representing the spin configuration.

        Returns:
        float: The computed Ising cost.
        """
        return float(state.T @ (self.J_mat @ state) + state.T @ self.h_vec)

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
        counts (dict): A dictionary containing bitstring measurement outcomes as keys and their corresponding probabilities as values.

        Returns:
        float: The computed expectation value.
        """
        _state_ = np.zeros(shape=self.h_vec.shape, dtype=float)
        _result_ = 0.0
        for bitstring, probability in counts.items():
            _state_ = np.array(list(bitstring)).reshape((self.n_qubits, 1)).astype(int)
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
        current_circuit = self.set_circuit(theta=theta)
        prob_distribution = current_circuit.get_state_probabilities()
        return self.compute_expectation(counts=prob_distribution)
