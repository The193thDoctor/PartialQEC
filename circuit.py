from utils import TrackedCircuit
from ec_blocks import ECBlock

# Change this constant to modify the number of logical qubits (blocks)
NUM_BLOCKS = 1

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
    tracked_circuit.append("DEPOLARIZE1", [0], error_rate)

    # Create and process each qubit block
    blocks = []
    for block_idx in range(NUM_BLOCKS):
        block_start = 1 + block_idx * 5
        block = ECBlock(tracked_circuit, block_start)
        blocks.append(block)
        
        # Encode the logical state
        block.encode_logical()
    
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

    print(repr(tracked_circuit.circuit.detector_error_model()))

    return tracked_circuit.circuit