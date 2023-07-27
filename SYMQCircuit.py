import numpy as np

from Tools import *


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

    def _update_circuit_unitary_(self, gate):
        self.__circuit_unitary__ = np.matmul(gate, self.__circuit_unitary__)

    def _tensor_prod_matrix_rep_(self, target_qubit: int, gate_mat_rep: np.ndarray) -> np.ndarray:
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
        if target_qubit >= self.circuit_size or target_qubit < 0:
            raise ValueError(f"Target qubit: '{target_qubit}', must be in 0 <= target qubit < circuit size.")

    def add_x(self, target_qubit: int):
        self._validity_(target_qubit=target_qubit)
        _mat_rep_ = self._tensor_prod_matrix_rep_(target_qubit=target_qubit, gate_mat_rep=self._x_gate_)
        self._update_circuit_unitary_(_mat_rep_)

    def add_y(self, target_qubit: int):
        self._validity_(target_qubit=target_qubit)
        _mat_rep_ = self._tensor_prod_matrix_rep_(target_qubit=target_qubit, gate_mat_rep=self._y_gate_)
        self._update_circuit_unitary_(_mat_rep_)

    def add_z(self, target_qubit: int):
        self._validity_(target_qubit=target_qubit)
        _mat_rep_ = self._tensor_prod_matrix_rep_(target_qubit=target_qubit, gate_mat_rep=self._z_gate_)
        self._update_circuit_unitary_(_mat_rep_)

    def add_h(self, target_qubit: int):
        self._validity_(target_qubit=target_qubit)
        _mat_rep_ = self._tensor_prod_matrix_rep_(target_qubit=target_qubit, gate_mat_rep=self._h_gate_)
        self._update_circuit_unitary_(_mat_rep_)

    def add_rx(self, target_qubit: int, angle: float):
        self._validity_(target_qubit=target_qubit)
        _rx_gate_ = np.cos(angle / 2) * self._identity_ - 1j * np.sin(angle / 2) * self._x_gate_
        _mat_rep_ = self._tensor_prod_matrix_rep_(target_qubit=target_qubit, gate_mat_rep=_rx_gate_)
        self._update_circuit_unitary_(_mat_rep_)

    def add_ry(self, target_qubit: int, angle: float):
        self._validity_(target_qubit=target_qubit)
        _ry_gate_ = np.cos(angle / 2) * self._identity_ + 1j * np.sin(angle / 2) * self._y_gate_.T
        _mat_rep_ = self._tensor_prod_matrix_rep_(target_qubit=target_qubit, gate_mat_rep=_ry_gate_)
        self._update_circuit_unitary_(_mat_rep_)

    def add_rz(self, target_qubit: int, angle: float):
        self._validity_(target_qubit=target_qubit)
        _rz_gate_ = np.array([[np.exp(-1j * angle / 2), 0.0], [0.0, np.exp(1j * angle / 2)]], dtype=complex)
        _mat_rep_ = self._tensor_prod_matrix_rep_(target_qubit=target_qubit, gate_mat_rep=_rz_gate_)
        self._update_circuit_unitary_(_mat_rep_)

    def get_circuit_unitary(self):
        return self.__circuit_unitary__

    def get_state_vector(self):
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
