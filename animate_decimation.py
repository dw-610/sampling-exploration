"""
Animated decimation visualization.

Shows how a signal changes as decimation factor increases from 1 to a maximum value.
Creates an animation of the decimate_and_compare view with varying decimation factors.
"""

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from signals import Sinusoid, Square, Triangle, Sawtooth, Chirp


class DecimationAnimator:
    """Animates signal decimation over a range of decimation factors."""

    def __init__(self, signal_class, signal_params, max_decimation=20,
                 show_samples=True, freq_magnitude_db=False, figsize=(12, 8)):
        """
        Initialize the decimation animator.

        Args:
            signal_class: Signal class to instantiate (e.g., Sinusoid, Square)
            signal_params: Dictionary of parameters to pass to signal constructor
            max_decimation: Maximum decimation factor to animate to
            show_samples: Whether to show sample markers in time domain
            freq_magnitude_db: Whether to show frequency magnitude in dB
            figsize: Figure size tuple (width, height)
        """
        self.signal_class = signal_class
        self.signal_params = signal_params
        self.max_decimation = max_decimation
        self.show_samples = show_samples
        self.freq_magnitude_db = freq_magnitude_db
        self.figsize = figsize

        # Current decimation factor
        self.current_factor = 1

        # Create the initial figure using decimate_and_compare
        self.fig = None
        self._init_figure()

    def _init_figure(self):
        """Create initial figure using decimate_and_compare."""
        # Create a fresh signal and use decimate_and_compare
        signal = self.signal_class(**self.signal_params)
        self.fig = signal.decimate_and_compare(
            decimation_factor=self.current_factor,
            show_samples=self.show_samples,
            freq_magnitude_db=self.freq_magnitude_db,
            figsize=self.figsize
        )

    def _update(self, frame):
        """
        Update function for animation.

        Args:
            frame: Frame number (maps to decimation factor)
        """
        # Decimation factor cycles from 1 to max_decimation
        self.current_factor = frame + 1

        # Create a fresh signal and reuse the figure
        signal = self.signal_class(**self.signal_params)
        self.fig = signal.decimate_and_compare(
            decimation_factor=self.current_factor,
            show_samples=self.show_samples,
            freq_magnitude_db=self.freq_magnitude_db,
            fig=self.fig  # Reuse existing figure
        )

        return self.fig.get_axes()

    def animate(self, interval=200, repeat=True):
        """
        Create and display the animation.

        Args:
            interval: Time between frames in milliseconds
            repeat: Whether to loop the animation

        Returns:
            Animation object
        """
        anim = animation.FuncAnimation(
            self.fig,
            self._update,
            frames=self.max_decimation,
            interval=interval,
            repeat=repeat,
            blit=False
        )

        return anim

    def save(self, filename, interval=200, fps=5):
        """
        Save the animation to a file.

        Args:
            filename: Output filename (e.g., 'decimation.gif' or 'decimation.mp4')
            interval: Time between frames in milliseconds
            fps: Frames per second for video output
        """
        anim = self.animate(interval=interval, repeat=True)
        anim.save(filename, writer='pillow' if filename.endswith('.gif') else 'ffmpeg', fps=fps)
        print(f"Animation saved to {filename}")


def main():
    """Example usage of the decimation animator."""

    # =========================================================================
    # CONFIGURATION - Adjust these parameters to customize the animation
    # =========================================================================

    # Signal selection: Choose one (Sinusoid, Square, Triangle, Sawtooth, Chirp)
    SIGNAL_TYPE = Sinusoid

    # Signal parameters
    FREQUENCY = 1e6              # Signal frequency in Hz
    NUM_PERIODS = 25             # Number of periods to show
    SAMPLING_FREQ = 20e6          # Sampling frequency in Hz

    # For Chirp signals, use these instead:
    CHIRP_F0 = 1e6               # Starting frequency
    CHIRP_F1 = 10e6              # Ending frequency
    CHIRP_DURATION = 1e-5        # Duration in seconds
    CHIRP_SAMPLING_FREQ = 100e6  # Sampling frequency for Chirp

    # Animation parameters
    MAX_DECIMATION = 30          # Maximum decimation factor to animate to
    ANIMATION_INTERVAL = 200     # Time between frames in milliseconds
    SHOW_SAMPLES = True          # Show individual sample points
    FREQ_MAGNITUDE_DB = False    # Show frequency magnitude in dB scale

    # Output options
    SAVE_ANIMATION = False       # Set to True to save animation to file
    OUTPUT_FILENAME = 'decimation.gif'  # Output file (.gif or .mp4)
    OUTPUT_FPS = 5               # Frames per second for saved animation

    # =========================================================================
    # Create and run animation
    # =========================================================================

    # Build signal parameters dictionary
    if SIGNAL_TYPE == Chirp:
        signal_params = {
            'f0': CHIRP_F0,
            'f1': CHIRP_F1,
            'duration': CHIRP_DURATION,
            'sampling_freq': CHIRP_SAMPLING_FREQ
        }
    else:
        signal_params = {
            'freq': FREQUENCY,
            'num_periods': NUM_PERIODS,
            'sampling_freq': SAMPLING_FREQ
        }

    # Create animator
    animator = DecimationAnimator(
        signal_class=SIGNAL_TYPE,
        signal_params=signal_params,
        max_decimation=MAX_DECIMATION,
        show_samples=SHOW_SAMPLES,
        freq_magnitude_db=FREQ_MAGNITUDE_DB
    )

    # Run or save animation
    if SAVE_ANIMATION:
        animator.save(OUTPUT_FILENAME, interval=ANIMATION_INTERVAL, fps=OUTPUT_FPS)
    else:
        anim = animator.animate(interval=ANIMATION_INTERVAL, repeat=True)
        plt.show()


if __name__ == '__main__':
    main()
