import stim
import numpy as np
import matplotlib.pyplot as plt
import sinter
from typing import List, Dict, Optional, Any

# Change this constant to modify the number of logical qubits (blocks)
NUM_BLOCKS = 1

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

        q0, q1, q2, q3, q4 = self.qubits

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


def create_circuit(error_rate, num_rounds=3):
    """
    Create a stim circuit for the specified error rate.

    Qubit assignment:
      Qubit 0: physical control (uncorrected) qubit.
      Qubits 1 to 5*NUM_BLOCKS: Each block of 5 qubits represents an encoded logical qubit using the [[5,1,3]] code.
      
    Args:
        error_rate: The error rate for depolarizing noise
        num_rounds: Number of rounds of stabilizer measurements
    """
    total_qubits = 1 + NUM_BLOCKS * 5
    tracked_circuit = TrackedCircuit()

    # Initialize all qubits to |0>
    for q in range(total_qubits):
        tracked_circuit.append("R", q)

    # (Optional) Prepare the control qubit in |1> if desired.
    tracked_circuit.append("X", 0)
    
    # Apply noise to control qubit
    # tracked_circuit.append("DEPOLARIZE1", [0], error_rate)

    # Create and process each qubit block
    blocks = []
    for block_idx in range(NUM_BLOCKS):
        block_start = 1 + block_idx * 5
        block = ECBlock(tracked_circuit, block_start)
        blocks.append(block)
        
        # Encode the logical state
        block.encode_logical()

        # Measure stabilizers for error detection in all blocks
        block.measure_stabilizers()
    
    # Apply noise and measure stabilizers for multiple rounds
    for round_idx in range(num_rounds):
        # Apply noise to all blocks
        for block in blocks:
            block.apply_noise(error_rate)

        # Measure stabilizers for error detection in all blocks
        for block in blocks:
            block.measure_stabilizers()

    # Final measurements in the Z basis
    tracked_circuit.append("M", 0)  # Measure control qubit
    for index in range(NUM_BLOCKS):
        blocks[index].measure_logical(index)

    print(tracked_circuit.circuit)
    print("-------------")

    return tracked_circuit.circuit

def main():
    """
    Run the simulation using sinter for sampling and decoding.
    """
    tasks = [
        sinter.Task(
            circuit=create_circuit(noise),
            json_metadata={'blocks': NUM_BLOCKS, 'p': noise},
            collection_options=sinter.CollectionOptions(max_shots=20000)
        )
        for noise in np.linspace(0.0, 0.01, 20)
    ]

    collected_stats: List[sinter.TaskStats] = sinter.collect(
        num_workers=4,
        tasks=tasks,
        decoders=['pymatching'],
        max_shots=50000,
        max_errors=500,
    )

    fig, ax = plt.subplots(1, 1)
    sinter.plot_error_rate(
        ax=ax,
        stats=collected_stats,
        x_func=lambda stats: stats.json_metadata['p'],
        failure_values_func=lambda _ : NUM_BLOCKS,
    )
    ax.set_ylim(0, 0.05)
    ax.set_xlim(0, 0.01)
    ax.set_title("[[5,1,3]] Code Error Rates (Depolarizing Noise)")
    ax.set_xlabel("Physical Error Rate")
    ax.set_ylabel("Logical Error Rate per Shot")
    ax.grid(which='major')
    ax.grid(which='minor')
    ax.legend()
    fig.set_dpi(120)
    plt.savefig('logical_vs_physical_error_rate.png', dpi=300)
    plt.show()

    print("Plot saved as 'logical_vs_physical_error_rate.png'")

if __name__ == "__main__":
    main()