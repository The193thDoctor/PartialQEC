import stim

class TrackedCircuit:
    """
    A wrapper around stim.Circuit that tracks measurement indices.
    """
    def __init__(self):
        self.circuit = stim.Circuit()
        self.measurement_count = 0
    
    def append(self, instruction, targets=None, arg=None):
        """
        Append an instruction to the circuit and track measurement indices.
        
        Args:
            instruction: The instruction name (e.g., "H", "CNOT", "M", "MPP")
            targets: The targets of the instruction
            arg: Optional argument for the instruction (e.g., error rate)
        
        Returns:
            The current measurement count if the instruction is a measurement,
            otherwise None.
        """
        result = None
        is_measurement = instruction in ["M", "MPP", "MX", "MY", "MZ", "MRX", "MRY", "MRZ"]
        
        if is_measurement:
            result = self.measurement_count
            # Count how many measurements are being added
            if instruction == "MPP":
                # For MPP, we add one measurement
                self.measurement_count += 1
            else:
                # For other measurement instructions, count the number of targets
                if isinstance(targets, list):
                    self.measurement_count += len(targets)
                else:
                    self.measurement_count += 1
        
        # Add the instruction to the underlying circuit
        if arg is not None:
            self.circuit.append(instruction, targets, arg)
        else:
            self.circuit.append(instruction, targets)
        
        return result

def stabilizer_to_mpp_targets(stab, block_qubits):
    """
    Convert a stabilizer string (e.g. "XZZXI") into a list of MPP targets for stim.
    Only non-identity (non-'I') operators are included, and a target_combiner is inserted
    between successive non-identity targets.
    """
    mpp_targets = []
    first_target = True
    for i, pauli in enumerate(stab):
        if pauli == 'I':
            continue
        if not first_target:
            mpp_targets.append(stim.target_combiner())
        first_target = False
        qubit = block_qubits[i]
        if pauli == 'X':
            mpp_targets.append(stim.target_x(qubit))
        elif pauli == 'Y':
            mpp_targets.append(stim.target_y(qubit))
        elif pauli == 'Z':
            mpp_targets.append(stim.target_z(qubit))
        else:
            raise ValueError(f"Unexpected Pauli letter: {pauli}")
    return mpp_targets