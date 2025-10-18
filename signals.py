"""
Signal generation and manipulation.

This module provides classes for generating and working with various types of
continuous-time signals (implemented as high-resolution discrete signals).
"""

# -----------------------------------------------------------------------------
# imports

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional
from plotting import plot_time_domain, setup_time_domain_axes, add_period_markers

# -----------------------------------------------------------------------------
# Base Signal Class


class Signal:
    """
    Base class for all signal types.

    Attributes:
        t: Time array (numpy array).
        signal: Signal values (numpy array).
        fs: Sampling frequency in Hz.
        duration: Total signal duration in seconds.
    """

    def __init__(self):
        self.t = None
        self.signal = None
        self.fs = None
        self.duration = None

    def plot(self, ax: Optional[plt.Axes] = None, show_samples: bool = False,
             show_grid: bool = True, **kwargs) -> plt.Axes:
        """
        Plot the signal in time domain.

        Args:
            ax: Matplotlib axes object. If None, creates new axes.
            show_samples: If True, show individual sample points.
            show_grid: If True, show grid lines.
            **kwargs: Additional keyword arguments passed to plotting functions.

        Returns:
            Matplotlib axes object containing the plot.

        Raises:
            ValueError: If signal has not been initialized.
        """
        if self.t is None or self.signal is None:
            raise ValueError("Signal not initialized")

        ax = plot_time_domain(self.t, self.signal, ax=ax,
                             show_samples=show_samples, **kwargs)

        if not show_grid:
            ax.grid(False)

        return ax

    def __len__(self):
        """
        Return number of samples in signal.

        Returns:
            Number of samples, or 0 if signal not initialized.
        """
        return len(self.signal) if self.signal is not None else 0

    def __repr__(self):
        """
        Return string representation of the signal.

        Returns:
            String containing signal class name, sampling frequency, duration,
            and number of samples.
        """
        return (f"{self.__class__.__name__}(fs={self.fs}, "
                f"duration={self.duration:.6f}s, samples={len(self)})")


# -----------------------------------------------------------------------------
# Signal Types


class Sinusoid(Signal):
    """
    Sinusoidal signal (cosine wave).

    Generates a cosine wave at a specified frequency.

    Attributes:
        freq: Signal frequency in Hz.
        period: Signal period in seconds (1/freq).
        num_periods: Number of periods in the signal.
        All attributes from Signal base class.
    """

    def __init__(self, freq: float, num_periods: float,
                 sampling_freq: Optional[int] = None):
        """
        Initialize a Sinusoid object (cosine wave).

        Args:
            freq: Signal frequency in Hz.
            num_periods: Number of periods to generate (sets signal length).
            sampling_freq: Sampling frequency in Hz. If None, defaults to 100*freq.
        """
        super().__init__()

        self.freq = freq
        self.period = 1/freq
        self.num_periods = num_periods
        self.fs = sampling_freq if sampling_freq else 100 * self.freq

        signal_length = num_periods * self.period
        self.duration = signal_length
        num_samples = int(signal_length * self.fs)

        self.t = np.linspace(0, signal_length, num_samples, endpoint=False)
        self.signal = np.cos(2 * np.pi * self.freq * self.t)

    def plot(self, ax: Optional[plt.Axes] = None, show_samples: bool = False,
             show_periods: bool = False, show_grid: bool = True,
             **kwargs) -> plt.Axes:
        """
        Plot the sinusoid signal in time domain.

        Args:
            ax: Matplotlib axes object. If None, creates new axes.
            show_samples: If True, show individual sample points.
            show_periods: If True, show vertical lines marking each period.
            show_grid: If True, show grid lines.
            **kwargs: Additional keyword arguments passed to plotting functions.

        Returns:
            Matplotlib axes object containing the plot.
        """
        # Call parent plot method
        ax = super().plot(ax=ax, show_samples=show_samples,
                         show_grid=show_grid, **kwargs)

        # Add period markers if requested
        if show_periods:
            add_period_markers(ax, self.period, self.num_periods)

        ax.set_title(f"Sinusoid: {self.freq:.2e} Hz, {self.num_periods} periods, "
                    f"fs = {self.fs:.2e} Hz")

        return ax


# -----------------------------------------------------------------------------


if __name__ == "__main__":

    frequency = 1e6
    periods = 5
    sampling_frequency = 25 * frequency

    s = Sinusoid(frequency, periods, sampling_frequency)

    # Test the new plot method
    s.plot(show_samples=True, show_periods=True)
    plt.show()


# -----------------------------------------------------------------------------