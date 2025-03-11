import stim
import numpy as np
import matplotlib.pyplot as plt
import sinter
from typing import List


def stabilizer_to_mpp_targets(stab, block_qubits):
    """
    Convert a stabilizer string (e.g. "XZZXI") into a list of MPP targets for stim.
    Only non-identity (non-'I') operators are included, and a target_combiner is inserted
    between successive non-identity targets.

    Parameters:
        stab (str): The stabilizer string (e.g. "XZZXI").
        block_qubits (List[int]): The list of qubit indices corresponding to the positions in the stabilizer.

    Returns:
        List[stim.GateTarget]: The list of targets suitable for an MPP gate.
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


def create_circuit(error_rate):
    """
    Create a stim circuit for the specified error rate.
    
    Parameters:
        error_rate (float): The physical error rate to use.
        
    Returns:
        stim.Circuit: The compiled circuit with detectors and observables.
    """
    # --------------------------
    # Build the circuit.
    # Total qubit count: 1 physical control qubit + 3 blocks * 5 qubits per block = 16 qubits.
    # Qubit assignment:
    #   Qubit 0: physical control qubit.
    #   Qubits 1-5: block 1 (encoded logical qubit using the [[5,1,3]] code).
    #   Qubits 6-10: block 2.
    #   Qubits 11-15: block 3.

    circuit = stim.Circuit()

    # Initialize all qubits to |0>
    for q in range(16):
        circuit.append("R", q)
    
    # Apply X to the control qubit to prepare |1> state
    circuit.append("X", 0)

    # Define the four stabilizer generators for the [[5,1,3]] code.
    stabilizers = [
        "XZZXI",
        "IXZZX",
        "XIXZZ",
        "ZXIXZ",
    ]
    
    # For each encoded block, perform:
    # 1. Transversal CNOT from the physical control (qubit 0) to each qubit in the block.
    # 2. Insert noise after the CNOTs.
    # 3. For each stabilizer, measure the corresponding multi-qubit Pauli observable using the MPP gate.
    num_blocks = 3
    for block in range(num_blocks):
        block_start = 1 + block * 5
        block_qubits = list(range(block_start, block_start + 5))

        # Transversal CNOT: apply CNOT from the physical control (qubit 0) to every qubit in the block.
        for tq in block_qubits:
            circuit.append("CNOT", [0, tq])

        # Insert noise after the transversal CNOT.
        circuit.append("DEPOLARIZE1", [0], error_rate)
        for q in block_qubits:
            circuit.append("DEPOLARIZE1", [q], error_rate)

        # For each stabilizer generator, convert the string to MPP targets using our function.
        for stab in stabilizers:
            mpp_targets = stabilizer_to_mpp_targets(stab, block_qubits)
            circuit.append("MPP", mpp_targets)
        circuit.append("DETECTOR", [stim.target_rec(-i) for i in range(1,len(stabilizers)+1)])

    # Final measurements:
    # Measure all qubits in the Z basis
    for qubit in range(16):
        circuit.append("M", qubit)

    return circuit


def main():
    """
    Main function to run the simulation using sinter for sampling and decoding.
    """
    # Create tasks for different code distances and noise rates
    tasks = [
        sinter.Task(
            circuit=create_circuit(noise),
            json_metadata={'blocks': 3, 'p': noise},
        )
        for noise in [0.001, 0.005, 0.01, 0.02, 0.05, 0.1]
    ]

    # Collect statistics using sinter with pymatching decoder
    collected_stats: List[sinter.TaskStats] = sinter.collect(
        num_workers=4,
        tasks=tasks,
        decoders=['pymatching'],
        max_shots=10_000,
        max_errors=500,
    )
    
    # Plot the results
    fig, ax = plt.subplots(1, 1)
    sinter.plot_error_rate(
        ax=ax,
        stats=collected_stats,
        x_func=lambda stats: stats.json_metadata['p'],
    )
    ax.set_ylim(1e-4, 1e-0)
    ax.set_xlim(1e-3, 1e-1)
    ax.loglog()
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