import os
import sys

import pytest
import qiskit.quantum_info
from qiskit import *

import qiskit
from qiskit.quantum_info import Operator
from qiskit.extensions import HamiltonianGate

os.chdir('../')
sys.path.append(os.getcwd())
from src.SYMQCircuit import *

backend = Aer.get_backend('statevector_simulator')


######################################################################################################
#                                 TEST CASE GENERATOR FUNCTIONS                                      #
######################################################################################################

#################
# 1 QUBIT GATES #
#################

def generate_1_qubit_test_cases(nr_rng_trials: int = 3) -> List[Tuple[int, int, float]]:
    test_cases = []

    _CIRCUIT_SIZES_ = [1 + i for i in range(6)]
    for size in _CIRCUIT_SIZES_:
        for trial in range(nr_rng_trials):
            qubit_nr = np.random.randint(low=0, high=size)
            angle = np.random.uniform(low=0, high=2 * np.pi)
            test_cases.append((size, qubit_nr, angle))
    return test_cases


def generate_1_qubit_test_cases_2(nr_rng_trials: int = 3) -> List[Tuple[int, int, float, float, float]]:
    test_cases = []

    _CIRCUIT_SIZES_ = [1 + i for i in range(6)]
    for size in _CIRCUIT_SIZES_:
        for trial in range(nr_rng_trials):
            qubit_nr = np.random.randint(low=0, high=size)
            angles = np.random.uniform(low=0, high=2 * np.pi, size=3)
            test_cases.append((size, qubit_nr, angles[0], angles[1], angles[2]))
    return test_cases


def generate_1_qubit_test_cases_3(nr_rng_trials: int = 3) -> List[Tuple[int, List[int], List[float]]]:
    test_cases = []

    _N_GATES_ = 9
    _CIRCUIT_SIZES_ = [1 + i for i in range(6)]
    for size in _CIRCUIT_SIZES_:
        for trial in range(nr_rng_trials):
            qubit_nrs = np.random.randint(low=0, high=size, size=_N_GATES_).tolist()
            angles = np.random.uniform(low=0, high=2 * np.pi, size=_N_GATES_ - 2).tolist()
            test_cases.append((size, qubit_nrs, angles))
    return test_cases


#################
# 2 QUBIT GATES #
#################

def generate_2_qubit_test_cases(nr_rng_trials: int = 3) -> List[Tuple[int, int, int, float]]:
    test_cases = []

    _CIRCUIT_SIZES_ = [2 + i for i in range(6)]
    for size in _CIRCUIT_SIZES_:
        for trial in range(nr_rng_trials):
            qubit_nrs = np.random.choice(a=size, size=2, replace=False)
            angle = np.random.uniform(low=0, high=2 * np.pi)
            test_cases.append((size, qubit_nrs[0], qubit_nrs[1], angle))
    return test_cases


#################
#     OTHER     #
#################

def generate_pauli_str_exp_test_cases(nr_rng_trials: int = 3) -> List[Tuple[str, float]]:
    test_cases = []

    _ALPHABET_ = ['I', 'X', 'Y', 'Z']
    _STR_LENGTHS_ = [1 + i for i in range(6)]
    for length in _STR_LENGTHS_:
        for trial in range(nr_rng_trials):
            string = "".join(np.random.choice(a=_ALPHABET_, size=length, replace=True))
            angle = np.random.uniform(low=0.0, high=2 * np.pi, size=1)[0]
            test_cases.append((string, angle))
    return test_cases


def generate_add_gate_test_cases(nr_rng_trials: int = 3) -> List[Tuple[SYMQCircuit, SYMQCircuit]]:
    test_cases = []
    for trial in range(nr_rng_trials):
        tot_circuit = SYMQCircuit(nr_qubits=4)
        tot_circuit.add_h(0)
        tot_circuit.add_h(3)
        tot_circuit.add_x(1)
        angle_1 = np.random.uniform(-np.pi, np.pi)
        tot_circuit.add_rz(3, angle_1)
        angle_2 = np.random.uniform(-np.pi, np.pi)
        tot_circuit.add_rxx(0, 2, angle_2)

        added_circuit = SYMQCircuit(nr_qubits=tot_circuit.circuit_size)
        added_circuit.add_gate(matrix_representation=tot_circuit.get_circuit_unitary())
        test_cases.append((tot_circuit, added_circuit))
    return test_cases


######################################################################################################
#                                             TESTING                                               #
######################################################################################################

N_RNG_TRIALS = 2

#################
# 1 QUBIT GATES #
#################

one_qubit_test_cases = generate_1_qubit_test_cases(nr_rng_trials=N_RNG_TRIALS)


@pytest.mark.parametrize('size, qubit_nr, angle', one_qubit_test_cases, )
def test_1_qubit_x_gate(size: int,
                        qubit_nr: int,
                        angle: float):
    qiskit_circuit = QuantumCircuit(size)
    qiskit_circuit.x(qubit_nr)

    my_circuit = SYMQCircuit(size)
    my_circuit.add_x(qubit_nr)

    # Comparing
    assert np.allclose(qiskit.quantum_info.Operator(qiskit_circuit).data, my_circuit.get_circuit_unitary())


@pytest.mark.parametrize('size, qubit_nr, angle', one_qubit_test_cases, )
def test_1_qubit_y_gate(size: int,
                        qubit_nr: int,
                        angle: float):
    qiskit_circuit = QuantumCircuit(size)
    qiskit_circuit.y(qubit_nr)

    my_circuit = SYMQCircuit(size)
    my_circuit.add_y(qubit_nr)

    # Comparing
    assert np.allclose(qiskit.quantum_info.Operator(qiskit_circuit).data, my_circuit.get_circuit_unitary())


@pytest.mark.parametrize('size, qubit_nr, angle', one_qubit_test_cases, )
def test_1_qubit_z_gate(size: int,
                        qubit_nr: int,
                        angle: float):
    qiskit_circuit = QuantumCircuit(size)
    qiskit_circuit.z(qubit_nr)

    my_circuit = SYMQCircuit(size)
    my_circuit.add_z(qubit_nr)

    # Comparing
    assert np.allclose(qiskit.quantum_info.Operator(qiskit_circuit).data, my_circuit.get_circuit_unitary())


@pytest.mark.parametrize('size, qubit_nr, angle', one_qubit_test_cases, )
def test_1_qubit_rx_gate(size: int,
                         qubit_nr: int,
                         angle: float):
    qiskit_circuit = QuantumCircuit(size)
    qiskit_circuit.rx(qubit=qubit_nr, theta=angle)

    my_circuit = SYMQCircuit(size)
    my_circuit.add_rx(target_qubit=qubit_nr, angle=angle)

    # Comparing
    assert np.allclose(qiskit.quantum_info.Operator(qiskit_circuit).data, my_circuit.get_circuit_unitary())


@pytest.mark.parametrize('size, qubit_nr, angle', one_qubit_test_cases, )
def test_1_qubit_ry_gate(size: int,
                         qubit_nr: int,
                         angle: float):
    qiskit_circuit = QuantumCircuit(size)
    qiskit_circuit.ry(qubit=qubit_nr, theta=angle)

    my_circuit = SYMQCircuit(size)
    my_circuit.add_ry(target_qubit=qubit_nr, angle=angle)

    # Comparing
    assert np.allclose(qiskit.quantum_info.Operator(qiskit_circuit).data, my_circuit.get_circuit_unitary())


@pytest.mark.parametrize('size, qubit_nr, angle', one_qubit_test_cases, )
def test_1_qubit_rz_gate(size: int,
                         qubit_nr: int,
                         angle: float):
    qiskit_circuit = QuantumCircuit(size)
    qiskit_circuit.rz(qubit=qubit_nr, phi=angle)

    my_circuit = SYMQCircuit(size)
    my_circuit.add_rz(target_qubit=qubit_nr, angle=angle)

    # Comparing
    assert np.allclose(qiskit.quantum_info.Operator(qiskit_circuit).data, my_circuit.get_circuit_unitary())


@pytest.mark.parametrize('size, qubit_nr, angle', one_qubit_test_cases, )
def test_1_qubit_s_gate(size: int,
                        qubit_nr: int,
                        angle: float):
    qiskit_circuit = QuantumCircuit(size)
    qiskit_circuit.s(qubit_nr)

    my_circuit = SYMQCircuit(size)
    my_circuit.add_s(qubit_nr)

    # Comparing
    assert np.allclose(qiskit.quantum_info.Operator(qiskit_circuit).data, my_circuit.get_circuit_unitary())


@pytest.mark.parametrize('size, qubit_nr, angle', one_qubit_test_cases, )
def test_1_qubit_t_gate(size: int,
                        qubit_nr: int,
                        angle: float):
    qiskit_circuit = QuantumCircuit(size)
    qiskit_circuit.t(qubit_nr)

    my_circuit = SYMQCircuit(size)
    my_circuit.add_t(qubit_nr)

    # Comparing
    assert np.allclose(qiskit.quantum_info.Operator(qiskit_circuit).data, my_circuit.get_circuit_unitary())


one_qubit_test_cases_2 = generate_1_qubit_test_cases_2(nr_rng_trials=N_RNG_TRIALS)


@pytest.mark.parametrize('size, qubit_nr, angle1, angle2, angle3', one_qubit_test_cases_2, )
def test_1_qubit_u_gate(size: int,
                        qubit_nr: int,
                        angle1: float,
                        angle2: float,
                        angle3: float):
    qiskit_circuit = QuantumCircuit(size)
    qiskit_circuit.u(theta=angle1, phi=angle2, lam=angle3, qubit=qubit_nr)

    my_circuit = SYMQCircuit(size)
    my_circuit.add_u(target_qubit=qubit_nr, angle_1=angle1, angle_2=angle2, angle_3=angle3)

    # Comparing
    assert np.allclose(qiskit.quantum_info.Operator(qiskit_circuit).data, my_circuit.get_circuit_unitary())


@pytest.mark.parametrize('size, qubit_nr, angle', one_qubit_test_cases, )
def test_1_qubit_h_gate(size: int,
                        qubit_nr: int,
                        angle: float):
    qiskit_circuit = QuantumCircuit(size)
    qiskit_circuit.h(qubit_nr)

    my_circuit = SYMQCircuit(size)
    my_circuit.add_h(qubit_nr)

    # Comparing
    assert np.allclose(qiskit.quantum_info.Operator(qiskit_circuit).data, my_circuit.get_circuit_unitary())


one_qubit_test_cases_3 = generate_1_qubit_test_cases_3(nr_rng_trials=N_RNG_TRIALS)


@pytest.mark.parametrize('size, qubit_nrs, angles', one_qubit_test_cases_3, )
def test_1_qubit_gate_sequence(size: int,
                               qubit_nrs: List[int],
                               angles: List[float]):
    qiskit_circuit = QuantumCircuit(size)
    qiskit_circuit.rz(angles[0], qubit_nrs[0])
    qiskit_circuit.ry(angles[1], qubit_nrs[1])
    qiskit_circuit.ry(angles[2], qubit_nrs[2])
    qiskit_circuit.x(qubit_nrs[3])
    qiskit_circuit.u(theta=angles[3], phi=angles[4], lam=angles[5], qubit=qubit_nrs[4])
    qiskit_circuit.h(qubit_nrs[5])
    qiskit_circuit.s(qubit_nrs[6])
    qiskit_circuit.rx(angles[6], qubit_nrs[7])
    qiskit_circuit.t(qubit_nrs[8])

    my_circuit = SYMQCircuit(size)
    my_circuit.add_rz(qubit_nrs[0], angles[0])
    my_circuit.add_ry(qubit_nrs[1], angles[1])
    my_circuit.add_ry(qubit_nrs[2], angles[2])
    my_circuit.add_x(qubit_nrs[3])
    my_circuit.add_u(angle_1=angles[3], angle_2=angles[4], angle_3=angles[5], target_qubit=qubit_nrs[4])
    my_circuit.add_h(qubit_nrs[5])
    my_circuit.add_s(qubit_nrs[6])
    my_circuit.add_rx(qubit_nrs[7], angles[6])
    my_circuit.add_t(qubit_nrs[8])

    assert np.allclose(qiskit.quantum_info.Operator(qiskit_circuit).data, my_circuit.get_circuit_unitary())


#################
# 2 QUBIT GATES #
#################

two_qubit_test_cases = generate_2_qubit_test_cases(nr_rng_trials=N_RNG_TRIALS)


@pytest.mark.parametrize('size, qubit_1, qubit_2, angle', two_qubit_test_cases, )
def test_2_qubit_cnot_gate(size: int,
                           qubit_1: int,
                           qubit_2: int,
                           angle: float):
    qiskit_circuit = QuantumCircuit(size)
    qiskit_circuit.cnot(control_qubit=qubit_1, target_qubit=qubit_2)

    my_circuit = SYMQCircuit(size)
    my_circuit.add_cnot(control_qubit=qubit_1, target_qubit=qubit_2)

    # Comparing
    assert np.allclose(qiskit.quantum_info.Operator(qiskit_circuit).data, my_circuit.get_circuit_unitary())


@pytest.mark.parametrize('size, qubit_1, qubit_2, angle', two_qubit_test_cases, )
def test_2_qubit_cx_gate(size: int,
                         qubit_1: int,
                         qubit_2: int,
                         angle: float):
    qiskit_circuit = QuantumCircuit(size)
    qiskit_circuit.cx(control_qubit=qubit_1, target_qubit=qubit_2)

    my_circuit = SYMQCircuit(size)
    my_circuit.add_cx(control_qubit=qubit_1, target_qubit=qubit_2)

    # Comparing
    assert np.allclose(qiskit.quantum_info.Operator(qiskit_circuit).data, my_circuit.get_circuit_unitary())


@pytest.mark.parametrize('size, qubit_1, qubit_2, angle', two_qubit_test_cases, )
def test_2_qubit_cy_gate(size: int,
                         qubit_1: int,
                         qubit_2: int,
                         angle: float):
    qiskit_circuit = QuantumCircuit(size)
    qiskit_circuit.cy(control_qubit=qubit_1, target_qubit=qubit_2)

    my_circuit = SYMQCircuit(size)
    my_circuit.add_cy(control_qubit=qubit_1, target_qubit=qubit_2)

    # Comparing
    assert np.allclose(qiskit.quantum_info.Operator(qiskit_circuit).data, my_circuit.get_circuit_unitary())


@pytest.mark.parametrize('size, qubit_1, qubit_2, angle', two_qubit_test_cases, )
def test_2_qubit_cz_gate(size: int,
                         qubit_1: int,
                         qubit_2: int,
                         angle: float):
    qiskit_circuit = QuantumCircuit(size)
    qiskit_circuit.cz(control_qubit=qubit_1, target_qubit=qubit_2)

    my_circuit = SYMQCircuit(size)
    my_circuit.add_cz(control_qubit=qubit_1, target_qubit=qubit_2)

    # Comparing
    assert np.allclose(qiskit.quantum_info.Operator(qiskit_circuit).data, my_circuit.get_circuit_unitary())


@pytest.mark.parametrize('size, qubit_1, qubit_2, angle', two_qubit_test_cases, )
def test_2_qubit_swap_gate(size: int,
                           qubit_1: int,
                           qubit_2: int,
                           angle: float):
    qiskit_circuit = QuantumCircuit(size)
    qiskit_circuit.swap(qubit1=qubit_1, qubit2=qubit_2)

    my_circuit = SYMQCircuit(size)
    my_circuit.add_swap(qubit_1=qubit_1, qubit_2=qubit_2)

    # Comparing
    assert np.allclose(qiskit.quantum_info.Operator(qiskit_circuit).data, my_circuit.get_circuit_unitary())


@pytest.mark.parametrize('size, qubit_1, qubit_2, angle', two_qubit_test_cases, )
def test_2_qubit_rxx_gate(size: int,
                          qubit_1: int,
                          qubit_2: int,
                          angle: float):
    qiskit_circuit = QuantumCircuit(size)
    qiskit_circuit.rxx(theta=angle, qubit1=qubit_1, qubit2=qubit_2)

    my_circuit = SYMQCircuit(size)
    my_circuit.add_rxx(angle=angle, qubit_1=qubit_1, qubit_2=qubit_2)

    # Comparing
    assert np.allclose(qiskit.quantum_info.Operator(qiskit_circuit).data, my_circuit.get_circuit_unitary())


@pytest.mark.parametrize('size, qubit_1, qubit_2, angle', two_qubit_test_cases, )
def test_2_qubit_ryy_gate(size: int,
                          qubit_1: int,
                          qubit_2: int,
                          angle: float):
    qiskit_circuit = QuantumCircuit(size)
    qiskit_circuit.ryy(theta=angle, qubit1=qubit_1, qubit2=qubit_2)

    my_circuit = SYMQCircuit(size)
    my_circuit.add_ryy(angle=angle, qubit_1=qubit_1, qubit_2=qubit_2)

    # Comparing
    assert np.allclose(qiskit.quantum_info.Operator(qiskit_circuit).data, my_circuit.get_circuit_unitary())


@pytest.mark.parametrize('size, qubit_1, qubit_2, angle', two_qubit_test_cases, )
def test_2_qubit_rzz_gate(size: int,
                          qubit_1: int,
                          qubit_2: int,
                          angle: float):
    qiskit_circuit = QuantumCircuit(size)
    qiskit_circuit.rzz(theta=angle, qubit1=qubit_1, qubit2=qubit_2)

    my_circuit = SYMQCircuit(size)
    my_circuit.add_rzz(angle=angle, qubit_1=qubit_1, qubit_2=qubit_2)

    # Comparing
    assert np.allclose(qiskit.quantum_info.Operator(qiskit_circuit).data, my_circuit.get_circuit_unitary())


@pytest.mark.parametrize('size, qubit_1, qubit_2, angle', two_qubit_test_cases, )
def test_2_qubit_crx_gate(size: int,
                          qubit_1: int,
                          qubit_2: int,
                          angle: float):
    qiskit_circuit = QuantumCircuit(size)
    qiskit_circuit.crx(control_qubit=qubit_1, target_qubit=qubit_2, theta=angle)

    my_circuit = SYMQCircuit(size)
    my_circuit.add_crx(control_qubit=qubit_1, target_qubit=qubit_2, angle=angle)

    # Comparing
    assert np.allclose(qiskit.quantum_info.Operator(qiskit_circuit).data, my_circuit.get_circuit_unitary())


@pytest.mark.parametrize('size, qubit_1, qubit_2, angle', two_qubit_test_cases, )
def test_2_qubit_cry_gate(size: int,
                          qubit_1: int,
                          qubit_2: int,
                          angle: float):
    qiskit_circuit = QuantumCircuit(size)
    qiskit_circuit.cry(control_qubit=qubit_1, target_qubit=qubit_2, theta=angle)

    my_circuit = SYMQCircuit(size)
    my_circuit.add_cry(control_qubit=qubit_1, target_qubit=qubit_2, angle=angle)

    # Comparing
    assert np.allclose(qiskit.quantum_info.Operator(qiskit_circuit).data, my_circuit.get_circuit_unitary())


@pytest.mark.parametrize('size, qubit_1, qubit_2, angle', two_qubit_test_cases, )
def test_2_qubit_crz_gate(size: int,
                          qubit_1: int,
                          qubit_2: int,
                          angle: float):
    qiskit_circuit = QuantumCircuit(size)
    qiskit_circuit.crz(control_qubit=qubit_1, target_qubit=qubit_2, theta=angle)

    my_circuit = SYMQCircuit(size)
    my_circuit.add_crz(control_qubit=qubit_1, target_qubit=qubit_2, angle=angle)

    # Comparing
    assert np.allclose(qiskit.quantum_info.Operator(qiskit_circuit).data, my_circuit.get_circuit_unitary())


#################
#     OTHER     #
#################

pauli_str_test_cases = generate_pauli_str_exp_test_cases(nr_rng_trials=N_RNG_TRIALS)


@pytest.mark.parametrize('string, angle', pauli_str_test_cases, )
def test_pauli_str_exp(string: str,
                       angle: float):
    flipped_string = string[::-1]
    circuit = SYMQCircuit(nr_qubits=len(string), precision=64)
    circuit.add_exp_of_pauli_string(pauli_string=flipped_string, theta=angle)

    # Comparing
    assert np.allclose(circuit.get_circuit_unitary(),
                       Operator(HamiltonianGate(Operator.from_label(string), angle)).data)


add_gate_test_cases = generate_add_gate_test_cases(nr_rng_trials=N_RNG_TRIALS)


@pytest.mark.parametrize('circuit_1, circuit_2', add_gate_test_cases, )
def test_add_gate(circuit_1: SYMQCircuit,
                  circuit_2: SYMQCircuit):
    # Comparing
    assert np.allclose(circuit_1.get_circuit_unitary(), circuit_2.get_circuit_unitary())
