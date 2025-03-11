import stim
import numpy as np
import matplotlib.pyplot as plt


def stabilizer_to_mpp_targets(stab, block_qubits):
    """Convert a stabilizer string to MPP targets."""
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
    """Create a stim circuit for the specified error rate."""
    circuit = stim.Circuit()

    # Initialize all qubits to |0>
    for q in range(16):
        circuit.append("R", q)
    
    # Flip control qubit to |1>
    circuit.append("X", 0)

    # Define the four stabilizer generators for the [[5,1,3]] code.
    stabilizers = ["XZZXI", "IXZZX", "XIXZZ", "ZXIXZ"]
    
    # For each block, apply transversal CNOT, add noise, measure stabilizers
    num_blocks = 3
    for block in range(num_blocks):
        block_start = 1 + block * 5
        block_qubits = list(range(block_start, block_start + 5))

        # Apply CNOTs from control to block
        for tq in block_qubits:
            circuit.append("CNOT", [0, tq])

        # Add noise
        circuit.append("DEPOLARIZE1", [0], error_rate)
        for q in block_qubits:
            circuit.append("DEPOLARIZE1", [q], error_rate)

        # Measure stabilizers
        for stab in stabilizers:
            mpp_targets = stabilizer_to_mpp_targets(stab, block_qubits)
            circuit.append("MPP", mpp_targets)

    # Measure all qubits in Z basis
    for qubit in range(16):
        circuit.append("M", qubit)

    return circuit


def simulate_and_get_error_rates(error_rate, num_shots=5000):
    """Simulate the circuit and return error rates."""
    circuit = create_circuit(error_rate)
    
    sampler = circuit.compile_sampler()
    measurement_samples = sampler.sample(num_shots)
    
    # Control qubit error rate (should be 1, errors flip it to 0)
    control_measurements = measurement_samples[:, 0]
    control_error_rate = np.sum(control_measurements == 0) / num_shots
    control_std_err = np.sqrt(control_error_rate * (1 - control_error_rate) / num_shots)
    
    # Calculate majority vote for each block and overall logical value
    block_votes = np.zeros((num_shots, 3))
    for shot in range(num_shots):
        for block in range(3):
            block_start = 1 + block * 5
            block_qubits = list(range(block_start, block_start + 5))
            block_measurements = measurement_samples[shot, block_qubits]
            # If majority are 1, the block vote is 1
            if np.sum(block_measurements) > len(block_qubits) / 2:
                block_votes[shot, block] = 1
    
    # Logical qubit error rate (should be 1, errors result in 0)
    logical_outcomes = np.zeros(num_shots)
    for shot in range(num_shots):
        if np.sum(block_votes[shot]) > 1.5:  # More than half of blocks vote 1
            logical_outcomes[shot] = 1
    
    logical_error_rate = np.sum(logical_outcomes == 0) / num_shots
    logical_std_err = np.sqrt(logical_error_rate * (1 - logical_error_rate) / num_shots)
    
    return control_error_rate, control_std_err, logical_error_rate, logical_std_err


def main():
    # Define just a few error rates for a quick run
    physical_error_rates = np.logspace(-3, -1, 5)
    
    # Storage for results
    control_error_rates = []
    control_error_std_errs = []
    logical_error_rates = []
    logical_error_std_errs = []
    
    # Simulate with a small number of shots
    num_shots = 5000
    
    print("Running simulations...")
    for p in physical_error_rates:
        print(f"Simulating with physical error rate p = {p:.6f}")
        control_err, control_std, logical_err, logical_std = simulate_and_get_error_rates(p, num_shots)
        
        control_error_rates.append(control_err)
        control_error_std_errs.append(control_std)
        logical_error_rates.append(logical_err)
        logical_error_std_errs.append(logical_std)
        
        print(f"  Physical error: {control_err:.6f} ± {control_std:.6f}, Logical error: {logical_err:.6f} ± {logical_std:.6f}")
    
    # Plot the results
    plt.figure(figsize=(10, 6))
    
    # Plot both error rates
    plt.errorbar(
        physical_error_rates, control_error_rates, yerr=control_error_std_errs,
        fmt='o--', capsize=5, linewidth=2, markersize=8, label='Physical Control Qubit'
    )
    
    plt.errorbar(
        physical_error_rates, logical_error_rates, yerr=logical_error_std_errs,
        fmt='o-', capsize=5, linewidth=2, markersize=8, label='Logical Qubit ([[5,1,3]] Code)'
    )
    
    # Add plot styling
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Physical Error Rate', fontsize=14)
    plt.ylabel('Error Rate', fontsize=14)
    plt.title('Error Rates vs Physical Error Rate', fontsize=16)
    plt.grid(True, which="both", ls="--", alpha=0.7)
    plt.legend(fontsize=12)
    plt.tight_layout()
    
    # Save and display plot
    plt.savefig('error_rates_comparison.png', dpi=300)
    plt.show()
    
    print("Plot saved as 'error_rates_comparison.png'")


if __name__ == "__main__":
    main()