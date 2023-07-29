import numpy as np


class SYMQState:
    def __init__(self, state: str) -> None:

        for single_qubit in state:
            if single_qubit not in ['0', '1']:
                raise ValueError(f"state: {state}, should be binary.")
        self._string_rep_ = state

    def get_statevector(self) -> np.ndarray:
        """
        Convert a binary string representation of a quantum state to its corresponding quantum state vector.

        Returns:
            np.ndarray: A complex-valued numpy array representing the quantum state vector.

        """

        # Define the relationship between binary digits '0' and '1' and their corresponding quantum state vectors.
        _relations_ = {'0': np.array([[1], [0]], dtype=complex),
                       '1': np.array([[0], [1]], dtype=complex)}

        # Initialize the quantum state vector using the first qubit's state.
        _state_ = _relations_[self._string_rep_[0]]

        # Iterate through the remaining qubits in the binary string and perform tensor product (kron) to get the full state.
        for single_qubit_state in range(1, len(self._string_rep_)):
            _state_ = np.kron(_state_, _relations_[self._string_rep_[single_qubit_state]])

        return _state_
