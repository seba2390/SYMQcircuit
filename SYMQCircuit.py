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

    def _validity_(self, target_qubit: int, control_qubit=None) -> None:
        """
        Check if the target qubit index is valid for the current quantum circuit.

        Args:
            target_qubit (int): The index of the target qubit (0-based) to be checked.

        Raises:
            ValueError: If the target qubit index is out of range for the circuit size.
        """
        if target_qubit >= self.circuit_size or target_qubit < 0:
            raise ValueError(f"Target qubit: '{target_qubit}', must be in 0 <= target qubit < circuit size.")

        if control_qubit is not None:
            if control_qubit >= self.circuit_size or control_qubit < 0:
                raise ValueError(f"Control qubit: '{control_qubit}', must be in 0 <= control qubit < circuit size.")
            if control_qubit == target_qubit:
                raise ValueError(f"Control qubit should be different from target qubit.")

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

    def add_t(self, target_qubit: int) -> None:
        """
        Add a T gate to the circuit.

        The T gate is a single-qubit gate that introduces a phase of π/4 to the qubit state.

        Args:
            target_qubit (int): The index of the target qubit.

        Returns:
            None
        """
        # Ensure that the target qubit index is valid
        self._validity_(target_qubit=target_qubit)

        # Define the T gate matrix representation
        _t_gate_ = np.array([[1.0, 0.0], [0.0, np.exp(1j * np.pi / 4)]], dtype=complex)

        # Compute the tensor product matrix representation for the T gate on the target qubit
        _mat_rep_ = self._single_qubit_tensor_prod_matrix_rep_(target_qubit=target_qubit, gate_mat_rep=_t_gate_)

        # Update the circuit's unitary representation with the T gate's matrix representation
        self._update_circuit_unitary_(_mat_rep_)

    def add_s(self, target_qubit: int) -> None:
        """
        Add an S gate to the circuit.

        The S gate is a single-qubit gate that introduces a phase of π/2 to the qubit state.

        Args:
            target_qubit (int): The index of the target qubit.

        Returns:
            None
        """
        # Ensure that the target qubit index is valid
        self._validity_(target_qubit=target_qubit)

        # Define the S gate matrix representation
        _s_gate_ = np.array([[1.0, 0.0], [0.0, 1j]], dtype=complex)

        # Compute the tensor product matrix representation for the S gate on the target qubit
        _mat_rep_ = self._single_qubit_tensor_prod_matrix_rep_(target_qubit=target_qubit, gate_mat_rep=_s_gate_)

        # Update the circuit's unitary representation with the S gate's matrix representation
        self._update_circuit_unitary_(_mat_rep_)

    ###############################################################
    ######################## 2 QUBIT GATES ########################
    ###############################################################

    def add_cnot(self, target_qubit: int, control_qubit: int) -> None:
        """
        Add a controlled-NOT (CNOT) gate to the quantum circuit.

        Args:
            target_qubit (int): The index of the target qubit (the qubit whose state is flipped if the control qubit is in state 1).
            control_qubit (int): The index of the control qubit.

        Returns:
            None
        """
        self._validity_(target_qubit=target_qubit, control_qubit=control_qubit)

        _flip_ = {'0': '1', '1': '0'}

        # Create a matrix representation of the circuit unitary
        _mat_rep_ = np.zeros_like(self.__circuit_unitary__)

        # Iterate over all possible basis states (bit string permutations)
        for basis_state in generate_bit_string_permutations(n=self.circuit_size):
            # Reversing to match qiskit convention (least significant bit on the right)
            _reversed_state_ = basis_state[::-1]

            if _reversed_state_[control_qubit] == '1':
                # If the control qubit is in state 1, apply the CNOT operation to the basis state
                _rs_basis_state_ = list(basis_state)
                # Note reverse indexing to match qiskit convention
                _rs_basis_state_[-(target_qubit + 1)] = _flip_[basis_state[-(target_qubit + 1)]]
                _rs_basis_state_ = ''.join(_rs_basis_state_)
                _rs_basis_state_vector_ = SYMQState(state=_rs_basis_state_).get_statevector()

                _ls_basis_state_vector_ = SYMQState(state=basis_state).get_statevector()
                _mat_rep_ += np.outer(_ls_basis_state_vector_, _rs_basis_state_vector_)
            else:
                # If the control qubit is in state 0, apply an identity operation to the basis state
                _basis_state_vector_ = SYMQState(state=basis_state).get_statevector()
                _mat_rep_ += np.outer(_basis_state_vector_, _basis_state_vector_)

        # Update the circuit unitary with the CNOT operation
        self._update_circuit_unitary_(_mat_rep_)

    def _get_cnot_mat(self, target_qubit: int, control_qubit: int):
        """
        Get the matrix representation of the controlled-NOT (CNOT) gate for the given target and control qubits.

        Args:
            target_qubit (int): The index of the target qubit (the qubit whose state is flipped if the control qubit is in state 1).
            control_qubit (int): The index of the control qubit.

        Returns:
            np.ndarray: The matrix representation of the CNOT gate.
        """

        _flip_ = {'0': '1', '1': '0'}

        # Create a matrix representation of the CNOT gate
        _mat_rep_ = np.zeros_like(self.__circuit_unitary__)

        # Iterate over all possible basis states (bit string permutations)
        for basis_state in generate_bit_string_permutations(n=self.circuit_size):
            # Reversing to match qiskit convention (least significant bit on the right)
            _reversed_state_ = basis_state[::-1]

            if _reversed_state_[control_qubit] == '1':
                # If the control qubit is in state 1, apply the CNOT operation to the basis state
                _rs_basis_state_ = list(basis_state)
                # Note reverse indexing to match qiskit convention
                _rs_basis_state_[-(target_qubit + 1)] = _flip_[basis_state[-(target_qubit + 1)]]
                _rs_basis_state_ = ''.join(_rs_basis_state_)
                _rs_basis_state_vector_ = SYMQState(state=_rs_basis_state_).get_statevector()

                _ls_basis_state_vector_ = SYMQState(state=basis_state).get_statevector()
                _mat_rep_ += np.outer(_ls_basis_state_vector_, _rs_basis_state_vector_)
            else:
                # If the control qubit is in state 0, apply an identity operation to the basis state
                _basis_state_vector_ = SYMQState(state=basis_state).get_statevector()
                _mat_rep_ += np.outer(_basis_state_vector_, _basis_state_vector_)

        return _mat_rep_

    def add_cx(self, target_qubit: int, control_qubit: int) -> None:
        """
        Add a controlled-X (CNOT) gate to the quantum circuit.

        Args:
            target_qubit (int): The index of the target qubit (the qubit whose state is flipped if the control qubit is in state 1).
            control_qubit (int): The index of the control qubit.

        Returns:
            None
        """

        self.add_cnot(target_qubit=target_qubit, control_qubit=control_qubit)

    def add_cy(self, target_qubit: int, control_qubit: int) -> None:
        """
        Add a controlled-Y gate to the quantum circuit.

        Args:
            target_qubit (int): The index of the target qubit.
            control_qubit (int): The index of the control qubit.

        Returns:
            None
        """
        # Add an Rz gate with angle -pi/2 to implement the controlled-Y gate
        self.add_rz(target_qubit=target_qubit, angle=-np.pi / 2)
        # Add a CNOT gate to perform the controlled-X operation
        self.add_cx(target_qubit=target_qubit, control_qubit=control_qubit)
        # Add another Rz gate with angle pi/2 to complete the controlled-Y operation
        self.add_rz(target_qubit=target_qubit, angle=np.pi / 2)

    def add_cz(self, target_qubit: int, control_qubit: int) -> None:
        """
        Add a controlled-Z gate to the quantum circuit.

        Args:
            target_qubit (int): The index of the target qubit.
            control_qubit (int): The index of the control qubit.

        Returns:
            None
        """
        # Add an H gate (Hadamard) to the target qubit
        self.add_h(target_qubit=target_qubit)
        # Add a CNOT gate to perform the controlled-X operation
        self.add_cx(target_qubit=target_qubit, control_qubit=control_qubit)
        # Add another H gate to the target qubit
        self.add_h(target_qubit=target_qubit)

    def add_swap(self, qubit_1: int, qubit_2: int) -> None:
        """
        Add a SWAP gate to the quantum circuit, exchanging two qubits.

        Args:
            qubit_1 (int): The index of the first qubit to be exchanged.
            qubit_2 (int): The index of the second qubit to be exchanged.

        Returns:
            None
        """
        self._validity_(target_qubit=qubit_1, control_qubit=qubit_2)
        _cnot1_mat_rep_ = self._get_cnot_mat(target_qubit=qubit_1, control_qubit=qubit_2)
        _cnot2_mat_rep_ = self._get_cnot_mat(target_qubit=qubit_2, control_qubit=qubit_1)
        # Apply the CNOT gate with control qubit as qubit_2 and target qubit as qubit_1
        self._update_circuit_unitary_(_cnot1_mat_rep_)
        # Apply the CNOT gate with control qubit as qubit_1 and target qubit as qubit_2
        self._update_circuit_unitary_(_cnot2_mat_rep_)
        # Apply the CNOT gate again with control qubit as qubit_2 and target qubit as qubit_1
        self._update_circuit_unitary_(_cnot1_mat_rep_)

    def add_rzz(self, qubit_1: int, qubit_2: int, angle: float) -> None:
        """
        Add a controlled phase shift gate (Rzz) to the quantum circuit.

        Args:
            qubit_1 (int): The index of the first qubit (control qubit).
            qubit_2 (int): The index of the second qubit (target qubit).
            angle (float): The angle of rotation in radians.

        Returns:
            None
        """
        # Apply a CNOT gate with control qubit as qubit_1 and target qubit as qubit_2
        self.add_cnot(target_qubit=qubit_1, control_qubit=qubit_2)
        # Apply an Rz gate to the target qubit with the specified angle
        self.add_rz(target_qubit=qubit_1, angle=angle)
        # Apply another CNOT gate with control qubit as qubit_1 and target qubit as qubit_2
        self.add_cnot(target_qubit=qubit_1, control_qubit=qubit_2)

    def add_rxx(self, qubit_1: int, qubit_2: int, angle: float) -> None:
        """
        Add a controlled rotation around the XX-axis (Rxx) to the quantum circuit.

        Args:
            qubit_1 (int): The index of the first qubit (control qubit).
            qubit_2 (int): The index of the second qubit (target qubit).
            angle (float): The angle of rotation in radians.

        Returns:
            None
        """
        # Apply Hadamard gate (H) to qubit_1
        self.add_h(target_qubit=qubit_1)
        # Apply Hadamard gate (H) to qubit_2
        self.add_h(target_qubit=qubit_2)
        # Apply Rzz gate with the specified angle to qubit_1 and qubit_2
        self.add_rzz(qubit_1=qubit_1, qubit_2=qubit_2, angle=angle)
        # Apply Hadamard gate (H) to qubit_1
        self.add_h(target_qubit=qubit_1)
        # Apply Hadamard gate (H) to qubit_2
        self.add_h(target_qubit=qubit_2)

    def add_ryy(self, qubit_1: int, qubit_2: int, angle: float) -> None:
        """
        Add a controlled rotation around the YY-axis (Ryy) to the quantum circuit.

        Args:
            qubit_1 (int): The index of the first qubit (control qubit).
            qubit_2 (int): The index of the second qubit (target qubit).
            angle (float): The angle of rotation in radians.

        Returns:
            None
        """
        # Apply Rx gate with angle pi/2 to qubit_1
        self.add_rx(target_qubit=qubit_1, angle=np.pi / 2)
        # Apply Rx gate with angle pi/2 to qubit_2
        self.add_rx(target_qubit=qubit_2, angle=np.pi / 2)
        # Apply Rzz gate with the specified angle to qubit_1 and qubit_2
        self.add_rzz(qubit_1=qubit_1, qubit_2=qubit_2, angle=angle)
        # Apply Rx gate with angle -pi/2 to qubit_1
        self.add_rx(target_qubit=qubit_1, angle=-np.pi / 2)
        # Apply Rx gate with angle -pi/2 to qubit_2
        self.add_rx(target_qubit=qubit_2, angle=-np.pi / 2)

    def add_crz(self, target_qubit: int, control_qubit: int, angle: float) -> None:
        """
        Add a controlled-RZ gate to the circuit.

        Args:
            target_qubit (int): The target qubit index.
            control_qubit (int): The control qubit index.
            angle (float): The rotation angle in radians.

        Returns:
            None
        """
        # Apply RZ gate to the target qubit controlled by the control qubit
        self.add_rz(target_qubit=target_qubit, angle=angle / 2)

        # Apply CX (CNOT) gate with control_qubit controlling the target_qubit
        self.add_cx(target_qubit=target_qubit, control_qubit=control_qubit)

        # Apply RZ gate to the target qubit controlled by the control qubit
        self.add_rz(target_qubit=target_qubit, angle=-angle / 2)

        # Apply CX (CNOT) gate with control_qubit controlling the target_qubit again
        self.add_cx(target_qubit=target_qubit, control_qubit=control_qubit)

    def add_crx(self, target_qubit: int, control_qubit: int, angle: float) -> None:
        """
        Add a controlled-RX gate to the circuit.

        Args:
            target_qubit (int): The target qubit index.
            control_qubit (int): The control qubit index.
            angle (float): The rotation angle in radians.

        Returns:
            None
        """
        # Apply Hadamard gate to the target qubit
        self.add_h(target_qubit=target_qubit)

        # Apply controlled-RZ gate to the target qubit controlled by the control qubit
        self.add_crz(target_qubit=target_qubit, control_qubit=control_qubit, angle=angle)

        # Apply Hadamard gate to the target qubit again
        self.add_h(target_qubit=target_qubit)

    def add_cry(self, target_qubit: int, control_qubit: int, angle: float) -> None:
        """
        Add a controlled-RY gate to the circuit.

        Args:
            target_qubit (int): The target qubit index.
            control_qubit (int): The control qubit index.
            angle (float): The rotation angle in radians.

        Returns:
            None
        """
        # Apply RY gate to the target qubit controlled by the control qubit
        self.add_ry(target_qubit=target_qubit, angle=angle / 2)

        # Apply CX (CNOT) gate with control_qubit controlling the target_qubit
        self.add_cx(target_qubit=target_qubit, control_qubit=control_qubit)

        # Apply RY gate to the target qubit controlled by the control qubit
        self.add_ry(target_qubit=target_qubit, angle=-angle / 2)

        # Apply CX (CNOT) gate with control_qubit controlling the target_qubit again
        self.add_cx(target_qubit=target_qubit, control_qubit=control_qubit)

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
