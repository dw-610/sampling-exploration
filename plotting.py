"""
Plotting utilities for signal visualization.

This module provides common plotting functions for time-domain and frequency-domain
signal visualization, designed to be used across different signal types.
"""

# -----------------------------------------------------------------------------
# imports

from typing import Optional
import matplotlib.pyplot as plt
import numpy as np

# -----------------------------------------------------------------------------
# Time Domain Plotting


def setup_time_domain_axes(ax: Optional[plt.Axes] = None,
                           title: str = "Time Domain Signal",
                           xlabel: str = "Time (s)",
                           ylabel: str = "Amplitude") -> plt.Axes:
    """
    Configure axes for time-domain plotting with consistent styling.

    Args:
        ax: Matplotlib axes object. If None, creates a new figure and axes.
        title: Plot title.
        xlabel: Label for x-axis.
        ylabel: Label for y-axis.

    Returns:
        Configured matplotlib axes object.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 4))

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    return ax


def plot_time_domain(t, signal, ax: Optional[plt.Axes] = None,
                     show_samples: bool = False,
                     **kwargs) -> plt.Axes:
    """
    Plot a time-domain signal.

    Args:
        t: Time array (numpy array or list).
        signal: Signal values (numpy array or list).
        ax: Matplotlib axes object. If None, creates new axes.
        show_samples: If True, show individual sample points as markers.
        **kwargs: Additional keyword arguments passed to plt.plot().

    Returns:
        Matplotlib axes object containing the plot.
    """
    if ax is None:
        ax = setup_time_domain_axes()

    if show_samples:
        ax.plot(t, signal, 'o-', markersize=3, **kwargs)
    else:
        ax.plot(t, signal, **kwargs)

    return ax


def add_period_markers(ax: plt.Axes, period: float, num_periods: int,
                       signal_start: float = 0.0) -> None:
    """
    Add vertical lines marking signal periods.

    Args:
        ax: Matplotlib axes object.
        period: Period duration in seconds.
        num_periods: Number of periods to mark.
        signal_start: Time offset where signal starts (default 0.0).
    """
    for i in range(num_periods + 1):
        t_marker = signal_start + i * period
        ax.axvline(t_marker, color='gray', linestyle='--',
                   alpha=0.5, linewidth=0.8)


# -----------------------------------------------------------------------------
# Frequency Domain Plotting


def setup_frequency_domain_axes(ax: Optional[plt.Axes] = None,
                                title: str = "Frequency Domain",
                                xlabel: str = "Frequency (Hz)",
                                ylabel: str = "Magnitude") -> plt.Axes:
    """
    Configure axes for frequency-domain plotting with consistent styling.

    Args:
        ax: Matplotlib axes object. If None, creates a new figure and axes.
        title: Plot title.
        xlabel: Label for x-axis.
        ylabel: Label for y-axis.

    Returns:
        Configured matplotlib axes object.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 4))

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    return ax


def plot_frequency_domain(frequencies, spectrum, ax: Optional[plt.Axes] = None,
                         show_magnitude: bool = True,
                         magnitude_db: bool = False,
                         **kwargs) -> plt.Axes:
    """
    Plot a frequency-domain representation of a signal.

    Args:
        frequencies: Frequency array (numpy array or list).
        spectrum: Complex spectrum values (numpy array or list).
        ax: Matplotlib axes object. If None, creates new axes.
        show_magnitude: If True, plot magnitude; otherwise plot real part.
        magnitude_db: If True, plot magnitude in dB (20*log10). Only applies when show_magnitude=True.
        **kwargs: Additional keyword arguments passed to plt.plot().

    Returns:
        Matplotlib axes object containing the plot.
    """
    if ax is None:
        if show_magnitude and magnitude_db:
            ylabel = "Magnitude (dB)"
        elif show_magnitude:
            ylabel = "Magnitude"
        else:
            ylabel = "Real Part"
        ax = setup_frequency_domain_axes(ylabel=ylabel)

    if show_magnitude:
        if magnitude_db:
            # Convert to dB scale, adding small epsilon to avoid log(0)
            magnitude = np.abs(spectrum)
            magnitude_db_values = 20 * np.log10(magnitude + 1e-10)
            # Clip to reasonable minimum to avoid extremely low values
            magnitude_db_values = np.clip(magnitude_db_values, -80, None)
            ax.plot(frequencies, magnitude_db_values, **kwargs)
        else:
            ax.plot(frequencies, np.abs(spectrum), **kwargs)
    else:
        ax.plot(frequencies, np.real(spectrum), **kwargs)

    return ax


# -----------------------------------------------------------------------------


def main():
    pass


# -----------------------------------------------------------------------------


if __name__ == "__main__":
    main()


# -----------------------------------------------------------------------------