# SYMQcircuit
An implementation of a gate based quantum state vector simulator. 
Utilizes Scipy sparse matrix structure[^1] to improve memory consumption and computation time. 

## Currently Implemented Gates: ##
#### 1 qubit gates: ####
- Pauli X, Pauli Y, Pauli Z
- RX, RY, RX
- T, S, P-phase & Hadamard
- U (Generic single-qubit rotation gate with 3 Euler angles).
#### 2 qubit gates: ####
- CX (CNOT), CY, CZ (Controlled Pauli gates)
- SWAP
- RXX, RYY, RZZ
- CRX, CRY, CRZ (Controlled rotations)

[^1]: [Sparse scipy matrices (scipy.sparse)](https://docs.scipy.org/doc/scipy/reference/sparse.html)

## Example: ##

```python
from qiskit.visualization import plot_histogram
from SYMQCircuit import *

# Defining number of qubits in circuit
_N_QUBITS_ = 3

# Creating instance of circuit
my_circuit = SYMQCircuit(nr_qubits=_N_QUBITS_)

#Adding H to all qubits in circuit
for q in range(_N_QUBITS_):
    my_circuit.add_h(target_qubit=q)
    
# Adding miscellaneous gates to circuit 
my_circuit.add_cnot(target_qubit=2, control_qubit=0)
my_circuit.add_rz(target_qubit=1, angle=np.pi/2)
my_circuit.add_cry(target_qubit=2, control_qubit=1, angle=np.pi / 7)

# Retrieving state vector
state_vector = my_circuit.get_state_vector()

# Or just get probability distribution 
probs = my_circuit.get_state_probabilities()

# Plotting the probability distribution
plot_histogram(probs)
```
<p align="center">
  <img src="https://github.com/seba2390/SYMQcircuit/blob/main/gallery/output.png" width="50%" />
</p>

N.B. the '+' operator is overloaded for the class, so that
circuit1 + circuit2 corresponds to simply extending circuit1 by circuit2:

#### Overloaded '+': ###

```python
_N_QUBITS_ = 10

## Creates circuits individually 
circuit1 = SYMQCircuit(nr_qubits=_N_QUBITS_,precision=64)
circuit1.add_cnot(target_qubit=2,control_qubit=4)
circuit1.add_rx(target_qubit=9,angle=9.32)

circuit2 = SYMQCircuit(nr_qubits=_N_QUBITS_,precision=64)
circuit2.add_x(1)
circuit2.add_h(2)

## Composes circuits by operator overloading + 
circuit3 = circuit1+circuit2

## The circuit that corresponds to the above 
circuit4 = SYMQCircuit(nr_qubits=_N_QUBITS_,precision=64)
circuit4.add_cnot(target_qubit=2,control_qubit=4)
circuit4.add_rx(target_qubit=9,angle=9.32)
circuit4.add_x(1)
circuit4.add_h(2)
```



