import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import UnitaryGate
from qiskit.quantum_info import Statevector

# Define the H_YZ matrix: 1/sqrt(2) * [[1, -i], [i, -1]]
H_yz_matrix = 1/np.sqrt(2) * np.array([[1, -1j],
                                         [1j, -1]])

# Create a UnitaryGate with the H_YZ matrix.
H_yz_gate = UnitaryGate(H_yz_matrix, label='H_YZ')

# Create a quantum circuit with 5 qubits and 5 classical bits.
circuit = QuantumCircuit(5, 5)

# Encoding circuit (mapping: q[0]=q_0, q[1]=q_1, q[2]=q_2, q[3]=q_3, q[4]=q_4):
circuit.cx(0, 3)       # CNOT from q_0 to q_3
circuit.h(4)           # Hadamard on q_4

# Use the custom H_YZ gate on q_1.
circuit.append(H_yz_gate, [1])
circuit.h(1)           # Standard Hadamard on q_1

circuit.cx(1, 3)       # CNOT from q_1 to q_3
circuit.cx(1, 2)       # CNOT from q_1 to q_2
circuit.cx(4, 1)       # CNOT from q_4 to q_1
circuit.cx(4, 0)       # CNOT from q_4 to q_0

# Apply the custom H_YZ gate on q_0.
circuit.append(H_yz_gate, [0])
circuit.cx(0, 2)       # CNOT from q_0 to q_2

# Optional: Measure all qubits (not used for statevector simulation).
circuit.measure(range(5), range(5))

# Print the circuit for verification
print(circuit.draw())

# Compute the final statevector of the circuit before measurement
# Note: The measurement operations collapse the state, so we remove them for simulation.
# Here we create a copy of the circuit without measurements.
circuit_no_meas = circuit.remove_final_measurements(inplace=False)
final_state = Statevector.from_instruction(circuit_no_meas)

# Print the final statevector
print("\nFinal statevector:")
print(final_state)
