import stim
from utils import stabilizer_to_mpp_targets

class ECBlock:
    """
    Represents a block of 5 qubits for the [[5,1,3]] code.
    """
    def __init__(self, circuit, start_index):
        """
        Initialize a block of qubits.
        
        Args:
            circuit: The stim circuit to which operations will be added
            start_index: The starting index of this block in the circuit
        """
        self.circuit = circuit
        self.start_index = start_index
        self.qubits = list(range(start_index, start_index + 5))
        
        # Define the stabilizer generators for the [[5,1,3]] code
        self.stabilizers = [
            "XZZXI",
            "IXZZX",
            "XIXZZ",
            "ZXIXZ",
        ]
        
        # Track the measurement record indices for each stabilizer
        self.last_measurement_indices = {stab: None for stab in self.stabilizers}

        # Logical Z
        self.logical_Z = "ZZZZZ"
    
    def encode_logical(self):
        """
        Encode the physical state in the first qubit of block_qubits into a logical state in the full block of qubits

        The ideal state is:

            |0_L> = 1/4 (|00000> + |10010> + |01001> + |10100> +
                          |01010> + |00101> - |11011> - |01110> -
                          |10111> - |11000> - |00110> - |11101> -
                          |01100> - |10001> - |00011> + |11110>)

        which is stabilized by:
          g1 = XZZXI, g2 = IXZZX, g3 = XIXZZ, g4 = ZXIXZ.
        """

        q0, q1, q2, q3, q4 = self.qubits

        # Methods from https://arxiv.org/abs/quant-ph/0410004. Looks like this doesn't work and there might be some issues
        #   differences
        #
        # circuit.append("CNOT", [q0, q3])
        # circuit.append("H_YZ", [q1])
        # circuit.append("H", [q1])
        # circuit.append("CNOT", [q1, q3])
        # circuit.append("CNOT", [q1, q2])
        # circuit.append("H", [q4])
        # circuit.append("CNOT", [q4, q1])
        # circuit.append("CNOT", [q4, q0])
        # circuit.append("H_YZ", [q0])
        # circuit.append("CNOT", [q0, q2])

        # Methods from https://doi.org/10.1201%2F9781420012293. There is some permutation of qubits we need to do to make
        #   this work.
        #
        # circuit.append("Z", [q0])
        # circuit.append("H", [q1])
        # circuit.append("CX", [q1, q0])
        # circuit.append("CZ", [q1, q2])
        # circuit.append("CZ", [q1, q4])
        #
        # circuit.append("H", [q4])
        # circuit.append("CX", [q4, q0])
        # circuit.append("CZ", [q4, q1])
        # circuit.append("CZ", [q4, q3])
        #
        # circuit.append("H", [q3])
        # circuit.append("CZ", [q3, q0])
        # circuit.append("CZ", [q3, q2])
        # circuit.append("CX", [q3, q4])
        #
        # circuit.append("H",[q2])
        # circuit.append("CZ", [q2, q1])
        # circuit.append("CX", [q2, q3])
        # circuit.append("CZ", [q2, q4])

        # Current encoding method
        self.circuit.append("H", [q1, q2, q3, q4])
        self.circuit.append("CX", [q1, q0, q2, q0, q3, q0, q4, q0])
        self.circuit.append("CZ", [q1, q2, q2, q3, q3, q4, q4, q0, q0, q1])
    
    def apply_noise(self, error_rate):
        """
        Apply depolarizing noise to all qubits in the block.
        
        Args:
            error_rate: The error rate for depolarizing noise
        """
        for q in self.qubits:
            self.circuit.append("DEPOLARIZE1", [q], error_rate)
    
    def measure_stabilizers(self):
        """
        Measure all stabilizers and create detectors.
        
        For each stabilizer, we:
        1. Measure the stabilizer using MPP (Multi-Pauli Product)
        2. Create a detector that compares the current measurement with the previous one
           (if available) to detect errors that occurred between measurements
        3. Store the current measurement index for future comparisons
        
        This enables the circuit to detect errors that flip stabilizer values between
        consecutive measurement rounds, which is essential for error correction.
        """
        for stab in self.stabilizers:
            mpp_targets = stabilizer_to_mpp_targets(stab, self.qubits)
            # Get the current measurement index before appending the MPP instruction
            current_index = self.circuit.append("MPP", mpp_targets)
            
            # Create detector targets based on current and previous measurement
            detector_targets = [stim.target_rec(-1)]
            if self.last_measurement_indices[stab] is not None:
                previous_index = self.last_measurement_indices[stab]
                # Calculate the record index relative to the current measurement
                relative_index = -(current_index - previous_index + 1)
                detector_targets.append(stim.target_rec(relative_index))
            
            # Add the detector instruction
            self.circuit.append("DETECTOR", detector_targets)
            
            # Update the last measurement index for this stabilizer
            self.last_measurement_indices[stab] = current_index
    
    def measure_logical(self, observable_index):
        # Make logical Measurement
        mpp_targets = stabilizer_to_mpp_targets(self.logical_Z, self.qubits)
        self.circuit.append("MPP", mpp_targets)

        # Append to observable record
        self.circuit.append("OBSERVABLE_INCLUDE", stim.target_rec(-1), observable_index)