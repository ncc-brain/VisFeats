from typing import Dict, Tuple

import numpy as np
from PIL import Image
from scipy import fft


def _pad_and_resize_image(
    image: Image.Image,
    size: int = 1024,
) -> np.ndarray:
    """Convert to grayscale, pad to a square, then resize to ``size``."""
    grayscale = np.asarray(image.convert("L"), dtype=np.float64)
    height, width = grayscale.shape

    # Redies et al. pad using a uniform border at the image mean gray value.
    side = max(height, width)
    padded = np.full((side, side), grayscale.mean(), dtype=np.float64)

    y0 = (side - height) // 2
    x0 = (side - width) // 2
    padded[y0 : y0 + height, x0 : x0 + width] = grayscale

    # Pillow's F mode preserves floating-point values during bicubic resize.
    resized = Image.fromarray(padded.astype(np.float64), mode="F").resize(
        (size, size),
        resample=Image.BICUBIC,
    )

    return np.asarray(resized, dtype=np.float64)


def _radially_averaged_power_spectrum(
    grayscale: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return radial spatial frequencies and mean power for a square image."""
    height, width = grayscale.shape
    if height != width:
        raise ValueError("Expected a square image.")

    size = height

    spectrum = fft.fftshift(fft.fft2(grayscale))
    power = np.abs(spectrum) ** 2

    # Frequencies expressed in cycles per image, rather than cycles/pixel.
    # For a 1024-pixel image, values span roughly -512 to +511 cycles/image.
    frequencies = fft.fftshift(fft.fftfreq(size)) * size
    fy, fx = np.meshgrid(frequencies, frequencies, indexing="ij")
    radius = np.hypot(fx, fy)

    # Annular radial bins: 0-1, 1-2, 2-3 ... cycles/image.
    radial_index = np.floor(radius).astype(np.intp)
    max_radius = radial_index.max()

    power_sum = np.bincount(
        radial_index.ravel(),
        weights=power.ravel(),
        minlength=max_radius + 1,
    )
    sample_count = np.bincount(
        radial_index.ravel(),
        minlength=max_radius + 1,
    )

    radial_power = np.divide(
        power_sum,
        sample_count,
        out=np.zeros_like(power_sum, dtype=np.float64),
        where=sample_count > 0,
    )

    radial_frequency = np.arange(max_radius + 1, dtype=np.float64)

    return radial_frequency, radial_power


def fourier_slope_and_sigma(
    image: Image.Image,
    size: int = 1024,
    min_frequency: float = 10.0,
    max_frequency: float = 256.0,
    log_bins: int = 17,
) -> Dict[str, float]:
    """
    Calculate Fourier slope and Fourier sigma of an image.

    Parameters
    ----------
    image : PIL.Image.Image
        Input image.
    size : int, default=1024
        Square analysis resolution.
    min_frequency : float, default=10.0
        Lowest fitted radial spatial frequency, in cycles/image.
    max_frequency : float, default=256.0
        Highest fitted radial spatial frequency, in cycles/image.
    log_bins : int, default=17
        Number of equally spaced bins in log-frequency space before fitting.
        Redies et al. [2]_ use 17 bins.

    Returns
    -------
    dict
        ``fourier_slope``: slope of the fitted line in log(power) versus
        log(frequency); natural scenes and many artworks are near -2.
        ``fourier_sigma``: mean squared residual of the log-binned data
        around the fitted line (sum of squared residuals divided by the
        number of points, not square-rooted).

    Notes
    -----
    The feature is computed in the following steps:

    1. Convert the image to grayscale, pad it to a square using the mean gray
       value, and resize it to ``size`` by bilinear interpolation according to [1]_.
    2. Compute the two-dimensional Fourier power spectrum and average power
       within annular radial-frequency bins.
    3. Keep frequencies from ``min_frequency`` to ``max_frequency`` cycles per
       image, then average log power and log frequency in ``log_bins``
       equally spaced log-frequency bins. The default of 17 follows [2]_.
    4. Fit a least-squares line to the binned log power versus log frequency.
       ``fourier_slope`` is its slope; ``fourier_sigma`` is the mean squared
       residual around the fitted line.

    .. [1] Redies, C., Grebenkina, M., Mohseni, M., Kaduhm, A., & Dobel, C. (2020).
       Global Image Properties Predict Ratings of Affective Pictures. Frontiers in Psychology, 11.
       https://doi.org/10.3389/fpsyg.2020.00953
    .. [2] Redies, C., Hasenstein, J., & Denzler, J. (2008). 
       Fractal-like image statistics in visual art: Similarity to natural scenes. Spatial Vision, 21(1–2), 137–148. 
       https://doi.org/10.1163/156856807782753921

    """
    grayscale = _pad_and_resize_image(image, size=size)
    frequency, power = _radially_averaged_power_spectrum(grayscale)

    # Exclude DC (0 cycles/image) and restrict to the published fit range.
    valid = (
        (frequency >= min_frequency)
        & (frequency <= max_frequency)
        & (power > 0)
        & np.isfinite(power)
    )

    frequency = frequency[valid]
    power = power[valid]

    if frequency.size < 2:
        raise ValueError("Not enough valid Fourier frequencies to fit a line.")

    log_frequency = np.log10(frequency)
    log_power = np.log10(power)

    # Equally spaced bins in log-frequency space.
    edges = np.linspace(
        np.log10(min_frequency),
        np.log10(max_frequency),
        log_bins + 1,
    )
    bin_index = np.digitize(log_frequency, edges, right=False) - 1
    bin_index = np.clip(bin_index, 0, log_bins - 1)

    counts = np.bincount(bin_index, minlength=log_bins)
    x_sum = np.bincount(
        bin_index,
        weights=log_frequency,
        minlength=log_bins,
    )
    y_sum = np.bincount(
        bin_index,
        weights=log_power,
        minlength=log_bins,
    )

    occupied = counts > 0
    x = x_sum[occupied] / counts[occupied]
    y = y_sum[occupied] / counts[occupied]

    if x.size < 2:
        raise ValueError("Not enough occupied log-frequency bins to fit a line.")

    # y = intercept + slope * x
    slope, intercept = np.polyfit(x, y, deg=1)
    fitted_y = intercept + slope * x

    # Redies et al. (2007): sum of squared deviations / number of points.
    fourier_sigma = np.mean((y - fitted_y) ** 2)

    return {
        "fourier_slope": float(slope),
        "fourier_sigma": float(fourier_sigma),
    }
