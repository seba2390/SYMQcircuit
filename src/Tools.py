from typing import *
import numpy as np
import matplotlib.pyplot as plt


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


def _get_state_probabilities_(state_vector_: np.ndarray, reverse_states: bool = False) -> dict:
    """
    Calculate the probabilities of each basis state in a quantum state.

    Returns:
        dict: A dictionary containing the basis state as keys and their respective probabilities as values.
    """
    _state_vector_ = state_vector_
    _probs_ = {}
    for n, c_n in enumerate(_state_vector_):
        _state_string_ = represent_integer_with_bits(number=n, nr_bits=int(np.log2(len(_state_vector_))))
        if reverse_states:
            _state_string_ = _state_string_[::-1]
        _probs_[_state_string_] = np.power(np.linalg.norm(c_n), 2)
    return _probs_


def sparsity(matrix: np.ndarray) -> float:
    return 1.0 - np.sum(matrix != 0.0) / (matrix.shape[0] * matrix.shape[1])


def plot_histogram(result_dict: dict[str, float]) -> None:
    fig, ax = plt.subplots(1,1, figsize=(5,3))

    x_labels = [r'|'+state+r'$\rangle$' for state in list(result_dict.keys())]

    x_positions = [0.3 * i for i in range(len(x_labels))]
    bars = ax.bar(x_positions, list(result_dict.values()), align='center', width=0.1)
    ax.hlines(0,ax.get_xlim()[0], ax.get_xlim()[1], ls='dashed')
    # Place the value of each bar above the respective bar
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, height + 0.01,
                '{:.3f}'.format(height), ha='center', va='bottom')
    ax.set_ylabel('Probability')
    ax.set_ylim(0-0.1*np.max(list(result_dict.values())), 1.2*np.max(list(result_dict.values())))
    ax.set_xticks(x_positions)
    ax.set_xticklabels(x_labels, rotation=75)

