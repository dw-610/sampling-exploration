"""
Signal generation and manipulation.

This module provides classes for generating and working with various types of
continuous-time signals (implemented as high-resolution discrete signals).
"""

# -----------------------------------------------------------------------------
# imports

from typing import Optional
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from plotting import (plot_time_domain, add_period_markers,
                      plot_frequency_domain)

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
        self.t_axis = None
        self.signal = None
        self.fs = None
        self.duration = None
        self.transform = None
        self.f_axis = None

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
        if self.t_axis is None or self.signal is None:
            raise ValueError("Signal not initialized")

        ax = plot_time_domain(self.t_axis, self.signal, ax=ax,
                             show_samples=show_samples, **kwargs)

        if not show_grid:
            ax.grid(False)

        return ax
    
    def _get_fft(self, shift = True):
        """
        Computes the discrete Fourier transform of the signal.

        Args:
            shift: uses fftshift to center the transform if True.

        Raises:
            ValueError: If signal has not been initialized.
        """
        if self.t_axis is None or self.signal is None:
            raise ValueError("Signal not initialized")

        # Compute FFT and normalize by signal length
        S = np.fft.fft(self.signal) / len(self.signal)
        self.transform = np.fft.fftshift(S) if shift else S

        w = np.linspace(-np.pi, np.pi, len(self.transform), endpoint=False)
        self.f_axis = w * self.fs / 2 / np.pi

    def plot_fft(self, ax: Optional[plt.Axes] = None, shift: bool = True,
                 show_magnitude: bool = True, show_grid: bool = True,
                 magnitude_db: bool = False, **kwargs) -> plt.Axes:
        """
        Plot the signal in frequency domain (FFT).

        Args:
            ax: Matplotlib axes object. If None, creates new axes.
            shift: If True, use fftshift to center the transform.
            show_magnitude: If True, plot magnitude; otherwise plot real part.
            show_grid: If True, show grid lines.
            magnitude_db: If True, plot magnitude in dB (20*log10 scale).
            **kwargs: Additional keyword arguments passed to plotting functions.

        Returns:
            Matplotlib axes object containing the plot.

        Raises:
            ValueError: If signal has not been initialized.
        """
        if self.t_axis is None or self.signal is None:
            raise ValueError("Signal not initialized")

        # Compute FFT
        self._get_fft(shift=shift)

        ax = plot_frequency_domain(self.f_axis, self.transform, ax=ax,
                                   show_magnitude=show_magnitude,
                                   magnitude_db=magnitude_db, **kwargs)

        if not show_grid:
            ax.grid(False)

        return ax
    
    def decimate(self, factor: int = 1):
        """
        Reduces the sampling rate of the signal by factor.

        Args:
            factor: Decimation factor.

        Raises:
            ValueError: If signal has not been initiated.
        """
        if self.t_axis is None or self.signal is None:
            raise ValueError("Signal not initialized")

        self.signal = self.signal[::factor]
        self.t_axis = self.t_axis[::factor]
        self.fs = self.fs / factor

        # Transform is now invalid, clear it
        self.f_axis = None
        self.transform = None

    def decimate_and_compare(self, decimation_factor: int,
                          show_samples: bool = True,
                          freq_magnitude_db: bool = False,
                          match_freq_xlim: bool = False,
                          figsize: tuple = (12, 8),
                          fig: Optional[plt.Figure] = None) -> plt.Figure:
        """
        Plot before/after decimation comparison in a 2x2 grid, then decimate.

        Creates a 2x2 plot showing:
        - Top left: Original time domain
        - Top right: Original frequency domain
        - Bottom left: Decimated time domain
        - Bottom right: Decimated frequency domain

        Note: This method modifies the signal state (decimates it).

        Args:
            decimation_factor: Decimation factor to apply.
            show_samples: If True, show individual sample points.
            freq_magnitude_db: If True, plot frequency domain magnitude in dB.
            match_freq_xlim: If True, set top freq plot x-axis to match bottom (decimated) range.
            figsize: Figure size tuple (width, height).
            fig: Optional existing figure to reuse. If None, creates new figure.

        Returns:
            Matplotlib figure object.
        """
        # Create subplot grid or reuse existing figure
        if fig is None:
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=figsize)
        else:
            # Reuse existing figure - get and clear axes
            axes = fig.get_axes()
            if len(axes) != 4:
                raise ValueError("Provided figure must have exactly 4 axes (2x2 grid)")
            ax1, ax2, ax3, ax4 = axes
            for ax in axes:
                ax.clear()

        # Plot original (blue - default color)
        self.plot(ax=ax1, show_samples=show_samples)
        ax1.set_title(f"Original Time Domain (fs={self.fs:.2e} Hz)")
        ax1.set_ylabel("Amplitude")

        self.plot_fft(ax=ax2, magnitude_db=freq_magnitude_db)
        ax2.set_title("Original Frequency Domain")

        # Store the original frequency domain y-limits for consistency
        original_freq_ylim = ax2.get_ylim()

        # Apply decimation
        self.decimate(decimation_factor)

        # Plot decimated (red)
        self.plot(ax=ax3, show_samples=show_samples, color='red')
        ax3.set_title(f"Decimated Time Domain (factor={decimation_factor}, fs={self.fs:.2e} Hz)")
        ax3.set_xlabel("Time (s)")
        ax3.set_ylabel("Amplitude")

        self.plot_fft(ax=ax4, magnitude_db=freq_magnitude_db, color='red')
        ax4.set_title(f"Decimated Frequency Domain (factor={decimation_factor})")
        ax4.set_xlabel("Frequency (Hz)")

        # Format frequency axes with MHz/kHz suffixes
        def freq_formatter(x, pos):
            """Format frequency labels with appropriate units."""
            if abs(x) >= 1e6:
                return f'{x/1e6:.1f}M'
            elif abs(x) >= 1e3:
                return f'{x/1e3:.1f}k'
            else:
                return f'{x:.1f}'

        ax2.xaxis.set_major_formatter(FuncFormatter(freq_formatter))
        ax4.xaxis.set_major_formatter(FuncFormatter(freq_formatter))

        # Set y-axis limits: use original limits, but expand if decimated exceeds them
        # If reusing figure, preserve the maximum y-limit seen so far
        decimated_freq_ylim = ax4.get_ylim()

        # Check if ax4 has a stored maximum (from previous calls when reusing figure)
        if hasattr(ax4, '_freq_ylim_max'):
            current_max = ax4._freq_ylim_max
        else:
            current_max = original_freq_ylim[1]

        # Update maximum if needed
        new_max = max(current_max, decimated_freq_ylim[1])
        ax4._freq_ylim_max = new_max

        final_ylim = (original_freq_ylim[0], new_max)
        ax4.set_ylim(final_ylim)

        # Match frequency x-axis limits if requested, or show span markers
        if match_freq_xlim:
            ax2.set_xlim(ax4.get_xlim())
        else:
            # Add vertical lines to show decimated frequency span
            xlim_decimated = ax4.get_xlim()
            ax2.axvline(xlim_decimated[0], color='red', linestyle='-',
                       linewidth=1, alpha=0.7)
            ax2.axvline(xlim_decimated[1], color='red', linestyle='-',
                       linewidth=1, alpha=0.7)

        plt.tight_layout()
        return fig

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

        self.t_axis = np.linspace(0, signal_length, num_samples, endpoint=False)
        self.signal = np.cos(2 * np.pi * self.freq * self.t_axis)

    def plot(self, ax: Optional[plt.Axes] = None, show_samples: bool = False,
             show_grid: bool = True, show_periods: bool = False,
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
    

class Square(Signal):
    """
    Square wave signal.

    Generates a square wave (50% duty cycle) at a specified frequency.

    Attributes:
        freq: Signal frequency in Hz.
        period: Signal period in seconds (1/freq).
        num_periods: Number of periods in the signal.
        All attributes from Signal base class.
    """

    def __init__(self, freq: float, num_periods: float,
                 sampling_freq: Optional[int] = None):
        """
        Initialize a Square object (square wave).

        Args:
            freq: Signal frequency in Hz.
            num_periods: Number of periods to generate (sets signal length).
            sampling_freq: Sampling frequency in Hz. If None, defaults to 200*freq.
        """
        super().__init__()

        self.freq = freq
        self.period = 1/freq
        self.num_periods = num_periods
        self.fs = sampling_freq if sampling_freq else 200 * self.freq

        signal_length = num_periods * self.period
        self.duration = signal_length
        num_samples = int(signal_length * self.fs)

        self.t_axis = np.linspace(0, signal_length, num_samples, endpoint=False)
        self.signal = np.ones((len(self.t_axis),))
        p_samps = int(self.period * self.fs)
        for i in range(num_periods):
            self.signal[int((i + 1/2) * p_samps):(i + 1) * p_samps] = -1
            

    def plot(self, ax: Optional[plt.Axes] = None, show_samples: bool = False,
             show_grid: bool = True, show_periods: bool = False,
             **kwargs) -> plt.Axes:
        """
        Plot the square wave in time domain.

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

        ax.set_title(f"Square Wave: {self.freq:.2e} Hz, {self.num_periods} periods, "
                    f"fs = {self.fs:.2e} Hz")

        return ax


class Chirp(Signal):
    """
    Linear chirp signal.

    Generates a signal with linearly varying frequency from f0 to f1.

    Attributes:
        f0: Starting frequency in Hz.
        f1: Ending frequency in Hz.
        duration: Signal duration in seconds.
        All attributes from Signal base class.
    """

    def __init__(self, f0: float, f1: float, duration: float,
                 sampling_freq: Optional[int] = None):
        """
        Initialize a Chirp object (linear frequency sweep).

        Args:
            f0: Starting frequency in Hz.
            f1: Ending frequency in Hz.
            duration: Signal duration in seconds.
            sampling_freq: Sampling frequency in Hz. If None, defaults to 100*max(f0, f1).
        """
        super().__init__()

        self.f0 = f0
        self.f1 = f1
        self.duration = duration
        self.fs = sampling_freq if sampling_freq else 100 * max(f0, f1)

        num_samples = int(duration * self.fs)

        self.t_axis = np.linspace(0, duration, num_samples, endpoint=False)

        # Linear chirp formula: s(t) = cos(2π * (f0*t + ((f1-f0)/(2*T)) * t²))
        chirp_rate = (f1 - f0) / (2 * duration)
        phase = 2 * np.pi * (f0 * self.t_axis + chirp_rate * self.t_axis**2)
        self.signal = np.cos(phase)

    def plot(self, ax: Optional[plt.Axes] = None, show_samples: bool = False,
             show_grid: bool = True, **kwargs) -> plt.Axes:
        """
        Plot the chirp signal in time domain.

        Args:
            ax: Matplotlib axes object. If None, creates new axes.
            show_samples: If True, show individual sample points.
            show_grid: If True, show grid lines.
            **kwargs: Additional keyword arguments passed to plotting functions.

        Returns:
            Matplotlib axes object containing the plot.
        """
        # Call parent plot method
        ax = super().plot(ax=ax, show_samples=show_samples,
                         show_grid=show_grid, **kwargs)

        ax.set_title(f"Chirp: {self.f0:.2e} Hz → {self.f1:.2e} Hz, "
                    f"duration={self.duration:.6f}s, fs={self.fs:.2e} Hz")

        return ax


class Triangle(Signal):
    """
    Triangle wave signal.

    Generates a triangle wave at a specified frequency.

    Attributes:
        freq: Signal frequency in Hz.
        period: Signal period in seconds (1/freq).
        num_periods: Number of periods in the signal.
        All attributes from Signal base class.
    """

    def __init__(self, freq: float, num_periods: float,
                 sampling_freq: Optional[int] = None):
        """
        Initialize a Triangle object (triangle wave).

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

        self.t_axis = np.linspace(0, signal_length, num_samples, endpoint=False)
        self.signal = (
            (4 / self.period) * 
            np.abs((self.t_axis % self.period) - self.period / 2) - 1
        )


class Sawtooth(Signal):
    """
    Sawtooth wave signal.

    Generates a sawtooth wave at a specified frequency.

    Attributes:
        freq: Signal frequency in Hz.
        period: Signal period in seconds (1/freq).
        num_periods: Number of periods in the signal.
        All attributes from Signal base class.
    """

    def __init__(self, freq: float, num_periods: float,
                 sampling_freq: Optional[int] = None):
        """
        Initialize a Sawtooth object (sawtooth wave).

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

        self.t_axis = np.linspace(0, signal_length, num_samples, endpoint=False)
        self.signal = (self.t_axis % self.period) / self.period * 2 - 1


# -----------------------------------------------------------------------------


if __name__ == "__main__":

    frequency = 1e6
    num_periods = 25
    sampling_freq = 100 * frequency
    decimation_factor = 15

    # Create a triangle wave at 5 Hz with 3 periods
    saw = Square(frequency, num_periods, sampling_freq)

    # Plot the decimation effect
    saw.decimate_and_compare(decimation_factor, show_samples=False)
    plt.show()


# -----------------------------------------------------------------------------