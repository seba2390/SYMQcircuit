# SYMQcircuit
An implementation of a state vector circuit w. gates. 
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

## Example: ##

```python
import numpy as np
from SYMQCircuit import *

# Defining number of qubits in circuit
_N_QUBITS_ = 16

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
```




