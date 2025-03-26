import stim
import numpy as np
import matplotlib.pyplot as plt
import sinter
from typing import List

# Change this constant to modify the number of logical qubits (blocks)
NUM_BLOCKS = 3

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
        """
        for stab in self.stabilizers:
            mpp_targets = stabilizer_to_mpp_targets(stab, self.qubits)
            self.circuit.append("MPP", mpp_targets)
            self.circuit.append("DETECTOR", [stim.target_rec(-1)])
    
    def measure_all(self):
        """
        Measure all qubits in the Z basis.
        """
        for q in self.qubits:
            self.circuit.append("M", q)

def create_circuit(error_rate):
    """
    Create a stim circuit for the specified error rate.

    Qubit assignment:
      Qubit 0: physical control (uncorrected) qubit.
      Qubits 1 to 5*NUM_BLOCKS: Each block of 5 qubits represents an encoded logical qubit using the [[5,1,3]] code.
    """
    total_qubits = 1 + NUM_BLOCKS * 5
    circuit = stim.Circuit()

    # Initialize all qubits to |0>
    for q in range(total_qubits):
        circuit.append("R", q)

    # (Optional) Prepare the control qubit in |1> if desired.
    circuit.append("X", 0)
    
    # Apply noise to control qubit
    circuit.append("DEPOLARIZE1", [0], error_rate)

    # Create and process each qubit block
    blocks = []
    for block_idx in range(NUM_BLOCKS):
        block_start = 1 + block_idx * 5
        block = ECBlock(circuit, block_start)
        blocks.append(block)
        
        # Encode the block into the logical state
        block.encode_logical()
        
        # Apply noise to the block
        block.apply_noise(error_rate)
        
        # Measure stabilizers for error detection
        block.measure_stabilizers()

    # Final measurements in the Z basis
    circuit.append("M", 0)  # Measure control qubit
    for block in blocks:
        block.measure_all()

    return circuit

def main():
    """
    Run the simulation using sinter for sampling and decoding.
    """
    tasks = [
        sinter.Task(
            circuit=create_circuit(noise),
            json_metadata={'blocks': NUM_BLOCKS, 'p': noise},
        )
        for noise in [0.001, 0.005, 0.01, 0.02, 0.05, 0.1]
    ]

    collected_stats: List[sinter.TaskStats] = sinter.collect(
        num_workers=4,
        tasks=tasks,
        decoders=['pymatching'],
        max_shots=10000,
        max_errors=500,
    )
    print(collected_stats)

    fig, ax = plt.subplots(1, 1)
    sinter.plot_error_rate(
        ax=ax,
        stats=collected_stats,
        x_func=lambda stats: stats.json_metadata['p'],
    )
    ax.set_ylim(0, 1.0)
    ax.set_xlim(0, 0.11)
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