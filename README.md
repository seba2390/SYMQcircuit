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
require 'redcarpet'
markdown = Redcarpet.new("Hello World!")
puts markdown.to_html
```




