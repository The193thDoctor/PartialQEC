import numpy as np
import matplotlib.pyplot as plt
import sinter
from typing import List
from circuit import create_circuit, NUM_BLOCKS
from decoder import LookUpTableDecoder

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
        for noise in np.linspace(0, 0.01, 2)
    ]

    collected_stats: List[sinter.TaskStats] = sinter.collect(
        num_workers=1,
        tasks=tasks,
        decoders=['lookup_table'],
        custom_decoders={'lookup_table': LookUpTableDecoder()},
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