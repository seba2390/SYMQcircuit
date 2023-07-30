import numpy as np

from Tools import *
from SYMQState import *


class SYMQCircuit:
    def __init__(self, nr_qubits: int):

        if nr_qubits <= 0:
            raise ValueError(f"Circuit size: '{nr_qubits}' should be positive int.")

        self.circuit_size = nr_qubits
        self.__circuit_unitary__ = np.eye(2 ** nr_qubits, dtype=complex)
        self.__state_vector__ = np.array([1.0] + [0.0 for _ in range(2 ** nr_qubits - 1)], dtype=complex)

        self._identity_ = np.eye(2, dtype=complex)

        self._x_gate_ = np.array([[0, 1], [1, 0]], dtype=complex)
        self._y_gate_ = np.array([[0, -1j], [1j, 0]], dtype=complex)
        self._z_gate_ = np.array([[1, 0], [0, -1]], dtype=complex)

        self._h_gate_ = (1.0 / np.sqrt(2.0)) * np.array([[1, 1], [1, -1]], dtype=complex)

    def _update_circuit_unitary_(self, gate: np.ndarray):
        """
        Update the circuit's unitary representation by performing matrix multiplication with a gate.

        Args:
            gate (np.ndarray): The gate's unitary matrix representation.

        Returns:
            None: This function updates the internal circuit's unitary representation in place.
        """
        self.__circuit_unitary__ = np.matmul(gate, self.__circuit_unitary__)

    def _single_qubit_tensor_prod_matrix_rep_(self, target_qubit: int, gate_mat_rep: np.ndarray) -> np.ndarray:
        """
        Calculate the tensor product of a gate's matrix representation with the identity matrix
        for all qubits except the target qubit.

        Args:
            target_qubit (int): The index of the target qubit (0-based) in the quantum circuit.
            gate_mat_rep (np.ndarray): The matrix representation of the gate.

        Returns:
            np.ndarray: The tensor product of the gate's matrix representation with identity matrices.
        """
        if target_qubit == self.circuit_size - 1:
            _mat_rep_ = gate_mat_rep
            for _qubit_ in range(0, self.circuit_size - 1):
                _mat_rep_ = np.kron(_mat_rep_, self._identity_)
        else:
            _mat_rep_ = self._identity_
            for _qubit_ in range(0, self.circuit_size - 1):
                if self.circuit_size - _qubit_ - 2 == target_qubit:
                    _mat_rep_ = np.kron(_mat_rep_, gate_mat_rep)
                else:
                    _mat_rep_ = np.kron(_mat_rep_, self._identity_)
        return _mat_rep_

    def _validity_(self, target_qubit: int):
        """
        Check if the target qubit index is valid for the current quantum circuit.

        Args:
            target_qubit (int): The index of the target qubit (0-based) to be checked.

        Raises:
            ValueError: If the target qubit index is out of range for the circuit size.
        """
        if target_qubit >= self.circuit_size or target_qubit < 0:
            raise ValueError(f"Target qubit: '{target_qubit}', must be in 0 <= target qubit < circuit size.")

    ###############################################################
    ######################## 1 QUBIT GATES ########################
    ###############################################################

    def add_id(self, target_qubit: int) -> None:
        """
        Apply the identity gate to the target qubit.

        Args:
            target_qubit (int): The index of the target qubit (0-based) in the quantum circuit.

        Raises:
            ValueError: If the target qubit index is out of range for the circuit size.
        """
        self._validity_(target_qubit=target_qubit)
        _mat_rep_ = self._single_qubit_tensor_prod_matrix_rep_(target_qubit=target_qubit, gate_mat_rep=self._identity_)
        self._update_circuit_unitary_(_mat_rep_)

    def add_x(self, target_qubit: int) -> None:
        """
        Apply the Pauli-X gate (NOT gate) to the target qubit.

        Args:
            target_qubit (int): The index of the target qubit (0-based) in the quantum circuit.

        Raises:
            ValueError: If the target qubit index is out of range for the circuit size.
        """
        self._validity_(target_qubit=target_qubit)
        _mat_rep_ = self._single_qubit_tensor_prod_matrix_rep_(target_qubit=target_qubit, gate_mat_rep=self._x_gate_)
        self._update_circuit_unitary_(_mat_rep_)

    def add_y(self, target_qubit: int) -> None:
        """
        Apply the Pauli-Y gate to the target qubit.

        Args:
            target_qubit (int): The index of the target qubit (0-based) in the quantum circuit.

        Raises:
            ValueError: If the target qubit index is out of range for the circuit size.
        """
        self._validity_(target_qubit=target_qubit)
        _mat_rep_ = self._single_qubit_tensor_prod_matrix_rep_(target_qubit=target_qubit, gate_mat_rep=self._y_gate_)
        self._update_circuit_unitary_(_mat_rep_)

    def add_z(self, target_qubit: int) -> None:
        """
        Apply the Pauli-Z gate to the target qubit.

        Args:
            target_qubit (int): The index of the target qubit (0-based) in the quantum circuit.

        Raises:
            ValueError: If the target qubit index is out of range for the circuit size.
        """
        self._validity_(target_qubit=target_qubit)
        _mat_rep_ = self._single_qubit_tensor_prod_matrix_rep_(target_qubit=target_qubit, gate_mat_rep=self._z_gate_)
        self._update_circuit_unitary_(_mat_rep_)

    def add_h(self, target_qubit: int) -> None:
        """
        Apply the Hadamard gate to the target qubit.

        Args:
            target_qubit (int): The index of the target qubit (0-based) in the quantum circuit.

        Raises:
            ValueError: If the target qubit index is out of range for the circuit size.
        """
        self._validity_(target_qubit=target_qubit)
        _mat_rep_ = self._single_qubit_tensor_prod_matrix_rep_(target_qubit=target_qubit, gate_mat_rep=self._h_gate_)
        self._update_circuit_unitary_(_mat_rep_)

    def add_rx(self, target_qubit: int, angle: float) -> None:
        """
        Apply the rotation around the X-axis gate to the target qubit.

        Args:
            target_qubit (int): The index of the target qubit (0-based) in the quantum circuit.
            angle (float): The rotation angle in radians.

        Raises:
            ValueError: If the target qubit index is out of range for the circuit size.
        """
        self._validity_(target_qubit=target_qubit)
        _rx_gate_ = np.cos(angle / 2) * self._identity_ - 1j * np.sin(angle / 2) * self._x_gate_
        _mat_rep_ = self._single_qubit_tensor_prod_matrix_rep_(target_qubit=target_qubit, gate_mat_rep=_rx_gate_)
        self._update_circuit_unitary_(_mat_rep_)

    def add_ry(self, target_qubit: int, angle: float) -> None:
        """
        Apply the rotation around the Y-axis gate to the target qubit.

        Args:
            target_qubit (int): The index of the target qubit (0-based) in the quantum circuit.
            angle (float): The rotation angle in radians.

        Raises:
            ValueError: If the target qubit index is out of range for the circuit size.
        """
        self._validity_(target_qubit=target_qubit)
        _ry_gate_ = np.cos(angle / 2) * self._identity_ + 1j * np.sin(angle / 2) * self._y_gate_.T
        _mat_rep_ = self._single_qubit_tensor_prod_matrix_rep_(target_qubit=target_qubit, gate_mat_rep=_ry_gate_)
        self._update_circuit_unitary_(_mat_rep_)

    def add_rz(self, target_qubit: int, angle: float) -> None:
        """
        Apply the rotation around the Z-axis gate to the target qubit.

        Args:
            target_qubit (int): The index of the target qubit (0-based) in the quantum circuit.
            angle (float): The rotation angle in radians.

        Raises:
            ValueError: If the target qubit index is out of range for the circuit size.
        """
        self._validity_(target_qubit=target_qubit)
        _rz_gate_ = np.array([[np.exp(-1j * angle / 2), 0.0], [0.0, np.exp(1j * angle / 2)]], dtype=complex)
        _mat_rep_ = self._single_qubit_tensor_prod_matrix_rep_(target_qubit=target_qubit, gate_mat_rep=_rz_gate_)
        self._update_circuit_unitary_(_mat_rep_)

    ###############################################################
    ######################## 2 QUBIT GATES ########################
    ###############################################################
    def add_cnot(self, target_qubit: int, control_qubit: int, verbose: bool = False) -> None:
        _flip_ = {'0': '1', '1': '0'}

        _mat_rep_ = np.zeros_like(self.__circuit_unitary__)
        for basis_state in generate_bit_string_permutations(n=self.circuit_size):
            if verbose: print(f"#### Basis state: {basis_state} ####")
            # Reversing to match qiskit
            _reversed_state_ = basis_state[::-1]
            if _reversed_state_[control_qubit] == '1':
                _rs_basis_state_ = list(basis_state)
                # Note reverse indexing to match qiskit
                _rs_basis_state_[-(target_qubit + 1)] = _flip_[basis_state[-(target_qubit + 1)]]
                _rs_basis_state_ = ''.join(_rs_basis_state_)
                if verbose: print(f"flipping: {basis_state}, to: {_rs_basis_state_}")
                _rs_basis_state_vector_ = SYMQState(state=_rs_basis_state_).get_statevector()

                _ls_basis_state_vector_ = SYMQState(state=basis_state).get_statevector()
                _mat_rep_ += np.outer(_ls_basis_state_vector_, _rs_basis_state_vector_)
                if verbose: print(f"adding: \n",
                                  np.outer(_ls_basis_state_vector_, _rs_basis_state_vector_).real.astype(int))

            else:
                _basis_state_vector_ = SYMQState(state=basis_state).get_statevector()
                _mat_rep_ += np.outer(_basis_state_vector_, _basis_state_vector_)
                if verbose: print(f"adding: \n", np.outer(_basis_state_vector_, _basis_state_vector_).real.astype(int))
        self._update_circuit_unitary_(_mat_rep_)

    def _get_cnot_mat(self, target_qubit: int, control_qubit: int):
        _flip_ = {'0': '1', '1': '0'}

        _mat_rep_ = np.zeros_like(self.__circuit_unitary__)
        for basis_state in generate_bit_string_permutations(n=self.circuit_size):
            # Reversing to match qiskit
            _reversed_state_ = basis_state[::-1]
            if _reversed_state_[control_qubit] == '1':
                _rs_basis_state_ = list(basis_state)
                # Note reverse indexing to match qiskit
                _rs_basis_state_[-(target_qubit + 1)] = _flip_[basis_state[-(target_qubit + 1)]]
                _rs_basis_state_ = ''.join(_rs_basis_state_)
                _rs_basis_state_vector_ = SYMQState(state=_rs_basis_state_).get_statevector()

                _ls_basis_state_vector_ = SYMQState(state=basis_state).get_statevector()
                _mat_rep_ += np.outer(_ls_basis_state_vector_, _rs_basis_state_vector_)
            else:
                _basis_state_vector_ = SYMQState(state=basis_state).get_statevector()
                _mat_rep_ += np.outer(_basis_state_vector_, _basis_state_vector_)
        return _mat_rep_

    def add_cx(self, target_qubit: int, control_qubit: int) -> None:
        self.add_cnot(target_qubit=target_qubit, control_qubit=control_qubit)

    def add_cy(self, target_qubit: int, control_qubit: int) -> None:
        pass

    def add_cz(self, target_qubit: int, control_qubit: int) -> None:
        pass

    def add_swap(self, qubit_1: int, qubit_2: int) -> None:
        _cnot1_mat_rep_ = self._get_cnot_mat(target_qubit=qubit_1, control_qubit=qubit_2)
        _cnot2_mat_rep_ = self._get_cnot_mat(target_qubit=qubit_2, control_qubit=qubit_1)
        self._update_circuit_unitary_(_cnot1_mat_rep_)
        self._update_circuit_unitary_(_cnot2_mat_rep_)
        self._update_circuit_unitary_(_cnot1_mat_rep_)

    def add_rzz(self, qubit_1: int, qubit_2: int, angle: float) -> None:
        _rz_gate_ = np.array([[np.exp(-1j * angle / 2), 0.0], [0.0, np.exp(1j * angle / 2)]], dtype=complex)
        _rz_mat_rep_ = self._single_qubit_tensor_prod_matrix_rep_(target_qubit=qubit_2, gate_mat_rep=_rz_gate_)
        _cnot_mat_rep_ = self._get_cnot_mat(target_qubit=qubit_1, control_qubit=qubit_2)
        self._update_circuit_unitary_(_cnot_mat_rep_)
        self._update_circuit_unitary_(_rz_mat_rep_)
        self._update_circuit_unitary_(_cnot_mat_rep_)

    def add_rxx(self, qubit_1: int, qubit_2: int, angle: float) -> None:
        self.add_h(target_qubit=qubit_1)
        self.add_h(target_qubit=qubit_2)
        self.add_rzz(qubit_1=qubit_1, qubit_2=qubit_2, angle=angle)
        self.add_h(target_qubit=qubit_1)
        self.add_h(target_qubit=qubit_2)

    def add_ryy(self, qubit_1: int, qubit_2: int, angle: float) -> None:
        self.add_rx(target_qubit=qubit_1, angle=np.pi / 2)
        self.add_rx(target_qubit=qubit_2, angle=np.pi / 2)
        self.add_rzz(qubit_1=qubit_1, qubit_2=qubit_2, angle=angle)
        self.add_rx(target_qubit=qubit_1, angle=-np.pi / 2)
        self.add_rx(target_qubit=qubit_2, angle=-np.pi / 2)

    def get_circuit_unitary(self):
        """
        Get the unitary representation of the quantum circuit.

        Returns:
            np.ndarray: The unitary representation as a numpy array.
        """
        return self.__circuit_unitary__

    def get_state_vector(self):
        """
        Get the state vector of the quantum circuit.

        Returns:
            np.ndarray: The state vector as a numpy array.
        """
        __state_vector__ = np.array([1.0] + [0.0 for _ in range(2 ** self.circuit_size - 1)], dtype=complex)
        return self.__circuit_unitary__ @ __state_vector__

    def get_state_probabilities(self) -> dict:
        """
        Calculate the probabilities of each basis state in a quantum state.

        Returns:
            dict: A dictionary containing the basis state as keys and their respective probabilities as values.
        """
        _state_vector_ = self.get_state_vector()
        _probs_ = {}
        for n, c_n in enumerate(_state_vector_):
            _state_string_ = represent_integer_with_bits(number=n, nr_bits=int(np.log2(len(_state_vector_))))
            _probs_[_state_string_] = np.power(np.linalg.norm(c_n), 2)
        return _probs_
