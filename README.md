# sampling-exploration

A Python toolkit for exploring signal sampling, decimation, and frequency-domain analysis. Visualize how aliasing affects various waveforms when sampling rates are reduced.

## Features

- **Signal Generation**: Sinusoid, Square, Triangle, Sawtooth, Chirp
- **Time & Frequency Visualization**: Plot signals in both domains with FFT analysis
- **Decimation**: Downsample signals and observe aliasing effects
- **Before/After Comparison**: 2×2 plots showing decimation impact
- **Animation**: Animate progressive decimation from factor 1 to N

## Installation

```bash
pip install -r requirements.txt
```

Requirements: `numpy`, `matplotlib`

## Quick Start

```python
from signals import Square
import matplotlib.pyplot as plt

# Create and visualize a signal
sig = Square(freq=1e6, num_periods=25, sampling_freq=20e6)
sig.plot(show_samples=True)

# Compare before/after decimation
sig.decimate_and_compare(decimation_factor=15)
plt.show()
```

## Signal Types

All signals use `(freq, num_periods, sampling_freq)` except Chirp:

```python
from signals import Sinusoid, Square, Triangle, Sawtooth, Chirp

# Periodic signals
Sinusoid(freq=1e6, num_periods=10, sampling_freq=100e6)
Square(freq=5, num_periods=3, sampling_freq=1000)

# Chirp: frequency sweep
Chirp(f0=1e6, f1=10e6, duration=1e-5, sampling_freq=100e6)
```

Default sampling: 100× frequency (200× for Square waves)

## Decimation Animation

Animate decimation effects over a range of factors:

```python
from animate_decimation import DecimationAnimator
from signals import Sinusoid

animator = DecimationAnimator(
    signal_class=Sinusoid,
    signal_params={'freq': 1e6, 'num_periods': 25, 'sampling_freq': 20e6},
    max_decimation=30,
    show_samples=True
)

anim = animator.animate(interval=200)  # 200ms per frame
plt.show()

# Or save to file
animator.save('decimation.gif', fps=5)
```

Configure animation parameters in `animate_decimation.py` at the top of `main()`.

## Core Methods

**Signal Class:**
- `plot(show_samples=False)` - Time-domain visualization
- `plot_fft(magnitude_db=False)` - Frequency-domain (FFT) visualization
- `decimate(factor)` - Reduce sampling rate by integer factor
- `decimate_and_compare(decimation_factor)` - 2×2 grid showing original vs decimated

**DecimationAnimator:**
- `animate(interval=200, repeat=True)` - Create matplotlib animation
- `save(filename, fps=5)` - Export to GIF or MP4

## Key Concepts

**Aliasing**: This toolkit uses *naive downsampling* (no anti-aliasing filter) to intentionally demonstrate aliasing artifacts. When decimation reduces the Nyquist frequency below signal bandwidth, high-frequency components fold into the baseband.

**FFT Normalization**: FFT is normalized by signal length, making magnitudes comparable across different signal durations.

**Visualization**: Decimated signals appear in red, originals in blue. Frequency axes auto-format with MHz/kHz suffixes for readability.

## Example Workflow

```python
from signals import Square
import matplotlib.pyplot as plt

# Generate square wave at 1 MHz, 20 MHz sampling
sig = Square(freq=1e6, num_periods=25, sampling_freq=20e6)

# View time domain
fig, ax = plt.subplots()
sig.plot(ax=ax, show_samples=True)
plt.show()

# View frequency domain (linear scale)
fig, ax = plt.subplots()
sig.plot_fft(ax=ax, magnitude_db=False)
plt.show()

# Compare decimation by factor of 15
# Creates 2×2 grid: original time/freq, decimated time/freq
fig = sig.decimate_and_compare(decimation_factor=15, show_samples=True)
plt.show()
```

## Project Structure

```
signals.py              # Signal classes and core methods
plotting.py             # Visualization utilities
animate_decimation.py   # Animation framework
requirements.txt        # Dependencies
```

## Notes

- Decimation modifies signal state (use fresh instance for multiple experiments)
- No windowing applied to FFT (rectangular window implicit)
- Square waves have fixed 50% duty cycle
- Chirp uses linear frequency sweep only
