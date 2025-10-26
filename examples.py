"""This script gives examples of aliasing with different signals."""

import matplotlib.pyplot as plt
from signals import Sinusoid, Square, Triangle, Sawtooth, Chirp

# CONFIGURATION: Change this to select which example to run
SIGNAL_TYPE = "sinusoid"  # Options: "sinusoid", "square", "triangle", "sawtooth", "chirp"

PRESETS = {
    "sinusoid": {
        "signal_class": Sinusoid,
        "signal_params": {
            "freq": 1000,
            "num_periods": 10,
            "sampling_freq": 10000,
        },
        "decimation_factor": 6,
    },

    "square": {
        "signal_class": Square,
        "signal_params": {
            "freq": 500,
            "num_periods": 8,
            "sampling_freq": 50000,
        },
        "decimation_factor": 12,
    },

    "triangle": {
        "signal_class": Triangle,
        "signal_params": {
            "freq": 800,
            "num_periods": 10,
            "sampling_freq": 80000,
        },
        "decimation_factor": 15,
    },

    "sawtooth": {
        "signal_class": Sawtooth,
        "signal_params": {
            "freq": 600,
            "num_periods": 10,
            "sampling_freq": 60000,
        },
        "decimation_factor": 10,
    },

    "chirp": {
        "signal_class": Chirp,
        "signal_params": {
            "f0": 100,
            "f1": 3000,
            "duration": 1.0,
            "sampling_freq": 30000,
        },
        "decimation_factor": 8,
    }
}


def run_example(signal_type):
    """
    Run an aliasing example for the specified signal type.

    Parameters:
    -----------
    signal_type : str
        One of: "sinusoid", "square", "triangle", "sawtooth", "chirp"
    """

    if signal_type not in PRESETS:
        print(f"Error: Unknown signal type '{signal_type}'")
        print(f"Available options: {', '.join(PRESETS.keys())}")
        return

    preset = PRESETS[signal_type]

    # Create and display the signal
    signal_class = preset["signal_class"]
    signal = signal_class(**preset["signal_params"])
    signal.decimate_and_compare(preset["decimation_factor"])


if __name__ == "__main__":
    run_example(SIGNAL_TYPE)
    plt.show()
