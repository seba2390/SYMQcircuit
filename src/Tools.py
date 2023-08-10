from typing import *
import numpy as np


def represent_integer_with_bits(number: int, nr_bits: int) -> str:
    """
    Represent an integer using a specific number of bits.

    Args:
        number (int): The integer to be represented.
        nr_bits (int): The number of bits to use for representation.

    Returns:
        str: A binary string representing the integer with leading zeros if required.
    """
    # Convert the integer to a binary string and remove the '0b' prefix
    binary_string = bin(number)[2:]
    # If the binary string is shorter than n, pad it with leading zeros
    binary_string = binary_string.zfill(nr_bits)
    return binary_string


def generate_bit_string_permutations(n: int) -> str:
    """
    A 'generator' type function that calculates all 2^n-1
    possible bitstring of a 'n-length' bitstring one at a time.
    (All permutations are not stored in memory simultaneously).

    :param n: length of bit-string
    :return: i'th permutation.
    """
    num_permutations = 2 ** n
    for i in range(num_permutations):
        _binary_string_ = bin(i)[2:].zfill(n)
        yield _binary_string_


def _get_state_probabilities_(state_vector_: np.ndarray) -> dict:
    """
    Calculate the probabilities of each basis state in a quantum state.

    Returns:
        dict: A dictionary containing the basis state as keys and their respective probabilities as values.
    """
    _state_vector_ = state_vector_
    _probs_ = {}
    for n, c_n in enumerate(_state_vector_):
        _state_string_ = represent_integer_with_bits(number=n, nr_bits=int(np.log2(len(_state_vector_))))
        _probs_[_state_string_] = np.power(np.linalg.norm(c_n), 2)
    return _probs_


def sparsity(matrix: np.ndarray) -> float:
    return 1.0 - np.sum(matrix != 0.0) / (matrix.shape[0] * matrix.shape[1])


