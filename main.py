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


def encode_logical(circuit, block_qubits):
    """
    Encode the physical state in the first qubit of block_qubits into a logical state in the full block of qubits

    The ideal state is:

        |0_L> = 1/4 (|00000> + |10010> + |01001> + |10100> +
                      |01010> + |00101> - |11011> - |01110> -
                      |10111> - |11000> - |00110> - |11101> -
                      |01100> - |10001> - |00011> + |11110>)

    which is stabilized by:
      g1 = XZZXI, g2 = IXZZX, g3 = XIXZZ, g4 = ZXIXZ.

    The following is one example encoding circuit (there are several alternatives).
    Adjust the gate sequence as needed to suit your simulation.
    """
    q0, q1, q2, q3, q4 = block_qubits

    circuit.append("CNOT",[q0, q3])
    circuit.append("H_YZ", [q1])
    circuit.append("H",[q1])
    circuit.append("CNOT", [q1,q3])
    circuit.append("CNOT",[q1,q2])
    circuit.append("H",[q4])
    circuit.append("CNOT",[q4,q1])
    circuit.append("CNOT",[q4,q0])
    circuit.append("H_YZ",[q0])
    circuit.append("CNOT",[q0,q2])

def create_circuit(error_rate):
    """
    Create a stim circuit for the specified error rate.

    Qubit assignment:
      Qubit 0: physical control (uncorrected) qubit.
      Qubits 1-5: Block 1 (encoded logical qubit using the [[5,1,3]] code).
      Qubits 6-10: Block 2.
      Qubits 11-15: Block 3.
    """
    circuit = stim.Circuit()

    # Initialize all qubits to |0>
    for q in range(16):
        circuit.append("R", q)

    # (Optional) Prepare the control qubit in |1> if desired.
    circuit.append("X", 0)

    # Define the four stabilizer generators for the [[5,1,3]] code.
    stabilizers = [
        "XZZXI",
        "IXZZX",
        "XIXZZ",
        "ZXIXZ",
    ]

    num_blocks = 3
    for block in range(num_blocks):
        block_start = 1 + block * 5
        block_qubits = list(range(block_start, block_start + 5))

        # Instead of using a transversal CNOT from qubit 0, encode the block into |0_L>.
        encode_logical(circuit, block_qubits)

        # Insert depolarizing noise after the encoding.
        circuit.append("DEPOLARIZE1", [0], error_rate)
        for q in block_qubits:
            circuit.append("DEPOLARIZE1", [q], error_rate)

        # For each stabilizer, measure the corresponding multi-qubit Pauli observable.
        for stab in stabilizers:
            mpp_targets = stabilizer_to_mpp_targets(stab, block_qubits)
            circuit.append("MPP", mpp_targets)
            circuit.append("DETECTOR", [stim.target_rec(-1)])

    # Final measurements in the Z basis.
    for qubit in range(16):
        circuit.append("M", qubit)

    print(circuit)
    print("CIRCUIT ABOVE")
    return circuit


def main():
    """
    Run the simulation using sinter for sampling and decoding.
    """
    tasks = [
        sinter.Task(
            circuit=create_circuit(noise),
            json_metadata={'blocks': 3, 'p': noise},
        )
        for noise in [0.001, 0.005, 0.01, 0.02, 0.05, 0.1]
    ]

    collected_stats: List[sinter.TaskStats] = sinter.collect(
        num_workers=4,
        tasks=tasks,
        decoders=['pymatching'],
        max_shots=10_000,
        max_errors=500,
    )

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
