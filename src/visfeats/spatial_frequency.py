from typing import Dict, Literal, Tuple, overload

import numpy as np
from scipy import ndimage
from numpy.typing import NDArray
from PIL import Image
from scipy import fft


def _pad_and_resize_image(
    image: Image.Image,
    size: int = 1024,
    pad: bool = True,
) -> np.ndarray:
    """Convert to grayscale, optionally pad to a square, then resize to ``size``."""
    grayscale = np.asarray(image.convert("L"), dtype=np.float64)

    if pad:
        height, width = grayscale.shape

        # Redies et al. pad using a uniform border at the image mean gray value.
        side = max(height, width)
        padded = np.full((side, side), grayscale.mean(), dtype=np.float64)

        y0 = (side - height) // 2
        x0 = (side - width) // 2
        padded[y0 : y0 + height, x0 : x0 + width] = grayscale
        grayscale = padded

    # Pillow's F mode is float32; it preserves fractional values during bicubic
    # resize (passing float64 with mode="F" would misread the buffer).
    resized = Image.fromarray(grayscale.astype(np.float32), mode="F").resize(
        (size, size),
        resample=Image.Resampling.BICUBIC,
    )

    return np.asarray(resized, dtype=np.float64)


def _centered_power_spectrum(
    grayscale: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the centered 2D power spectrum and its (fy, fx) frequency grids."""
    height, width = grayscale.shape
    if height != width:
        raise ValueError("Expected a square image.")

    spectrum = fft.fftshift(fft.fft2(grayscale))
    power = np.abs(spectrum) ** 2

    # Frequencies expressed in cycles per image, rather than cycles/pixel.
    # For a 1024-pixel image, values span roughly -512 to +511 cycles/image.
    frequencies = fft.fftshift(fft.fftfreq(height)) * height
    fy, fx = np.meshgrid(frequencies, frequencies, indexing="ij")

    return power, fy, fx


def _radially_averaged_spectrum(
    grayscale: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return mean radial frequency, mean power and mean magnitude per annulus.

    Annuli are 1 cycle/image wide, up to the Nyquist frequency.
    """
    power, fy, fx = _centered_power_spectrum(grayscale)

    # Radial spatial frequency, in cycles/image
    radius = np.hypot(fx, fy)
    # Consider only frequencies up to the Nyquist limit (half the image size).
    valid = radius <= min(grayscale.shape) / 2

    radius = radius[valid]
    power = power[valid]
    radial_index = np.floor(radius).astype(np.intp)
    index = np.arange(radial_index.max() + 1)

    def annular_mean(values: np.ndarray) -> np.ndarray:
        return np.asarray(ndimage.mean(values, labels=radial_index, index=index))

    return annular_mean(radius), annular_mean(power), annular_mean(np.sqrt(power))


def _spectral_centroid_and_variance(grayscale: np.ndarray) -> Tuple[float, float]:
    """Return the power-weighted mean frequency and orientation circular variance."""
    power, fy, fx = _centered_power_spectrum(grayscale)

    # Exclude DC: it carries the mean luminance and has no defined orientation.
    nonzero = np.hypot(fx, fy) > 0
    power = power[nonzero]
    radius = np.hypot(fx, fy)[nonzero]
    angle = np.arctan2(fy, fx)[nonzero]

    total_power = power.sum()
    if total_power == 0:
        raise ValueError("Total power of the Fourier spectrum is zero.")

    centroid = (radius * power).sum() / total_power

    # Orientation is axial (theta and theta + pi are equivalent in a
    # real-valued image's spectrum), so angles are doubled before averaging.
    resultant = np.abs((power * np.exp(2j * angle)).sum()) / total_power

    return float(centroid), float(1.0 - resultant)


def _dimensionality(radial_magnitude: np.ndarray) -> float:
    """Slope of log magnitude versus log rank of the radially averaged spectrum."""
    # Skip the DC bin; rank all remaining components by magnitude.
    magnitude = radial_magnitude[1:]
    magnitude = magnitude[magnitude > 0]
    if magnitude.size < 2:
        raise ValueError("At least two non-zero Fourier components are required.")

    ranked = np.sort(magnitude)[::-1]
    ranks = np.arange(1, ranked.size + 1)
    return float(np.polyfit(np.log10(ranks), np.log10(ranked), deg=1)[0])


@overload
def spatial_frequency_features(
    image: Image.Image,
    size: int = ...,
    min_frequency: float = ...,
    max_frequency: float = ...,
    log_bins: int = ...,
    pad: bool = ...,
    return_spectrum: Literal[False] = ...,
) -> Dict[str, float]: ...


@overload
def spatial_frequency_features(
    image: Image.Image,
    size: int = ...,
    min_frequency: float = ...,
    max_frequency: float = ...,
    log_bins: int = ...,
    pad: bool = ...,
    *,
    return_spectrum: Literal[True],
) -> Tuple[Dict[str, float], NDArray[np.float64]]: ...


def spatial_frequency_features(
    image: Image.Image,
    size: int = 1024,
    min_frequency: float = 10.0,
    max_frequency: float = 256.0,
    log_bins: int = 17,
    pad: bool = True,
    return_spectrum: bool = False,
) -> Dict[str, float] | Tuple[Dict[str, float], NDArray[np.float64]]:
    """
    Calculate features related to the spatial frequency content of an image.

    Parameters
    ----------
    image : Image
        Input image.
    size : int, default=1024
        Square analysis resolution.
    min_frequency : float, default=10.0
        Lowest radial spatial frequency to analyze, in cycles/image.
    max_frequency : float, default=256.0
        Highest radial spatial frequency to analyze, in cycles/image.
    log_bins : int, default=17
        Number of log-spaced frequency bins used before fitting. Redies et
        al. [2]_ used 17 bins (Figure 3a).
    pad : bool, default=True
        Whether to pad non-square images to a square with the mean gray value
        before resizing, as in Redies et al. [1]_. If False, the image is
        resized directly to ``size`` x ``size``, as in the original method
        [2]_, which distorts the aspect ratio of non-square images.
    return_spectrum : bool, default=False
        If True, also return the radially averaged power spectrum
        (see Returns).

    Returns
    -------
    features : dict of {str: float}
        A mapping of feature names to values:

        .. list-table::
           :header-rows: 1
           :widths: 30 70

           * - Key
             - Description
           * - ``fourier_slope``
             - Slope of log power versus log frequency (about -2 for natural
               scenes and many artworks).
           * - ``fourier_deviation``
             - Mean squared residual (log10 units) of the binned data around
               the fitted line.
           * - ``dimensionality``
             - Slope of log magnitude versus log rank of the ranked spectrum
               (see Notes).
           * - ``spectral_centroid``
             - Power-weighted mean spatial frequency, in cycles/image.
           * - ``spectral_variance``
             - Circular variance of the power spectrum over orientation, in
               [0, 1] (see Notes).

    spectrum : ndarray, shape (n_frequencies, 2)
        Only returned when ``return_spectrum=True``, as the second element of
        a ``(features, spectrum)`` tuple. This is the radially averaged
        spectrum before log-binning, restricted to frequencies between
        ``min_frequency`` and ``max_frequency`` with non-zero power. Axes are:

        - axis 0: radial frequency bin, in ascending order (one bin per
          cycle/image);
        - axis 1: column 0 = mean radial frequency of the Fourier components
          in the bin (cycles/image), column 1 = mean power in the bin
          (squared amplitude, arbitrary units).

        Both columns are linear (not log-transformed). For amplitude instead
        of power, use ``np.sqrt(spectrum[:, 1])``.

    Notes
    -----
    The function follows these steps:

    1. Convert image to grayscale, optionally (``pad``) pad to a square with the
       mean gray value, and resize to a ``size`` x ``size`` square
       by bicubic interpolation [1,2]_.
    2. Compute the 2D Fourier transform and average it in annular bins of
       1 cycle/image up to the Nyquist frequency, giving the mean radial
       frequency, mean power and mean magnitude of each annulus.
    3. ``fourier_slope`` and ``fourier_deviation``: keep bins between
       ``min_frequency`` and ``max_frequency``, average log power and log
       frequency within ``log_bins`` log-spaced bins, and fit a least-squares
       line. The slope is ``fourier_slope``; the mean squared residual is
       ``fourier_deviation`` [2]_.
    4. ``dimensionality``: using the radially averaged magnitude of all bins
       except DC (no frequency restriction), rank the components from
       largest to smallest and fit a line to log magnitude versus log rank
       [3]_.
    5. ``spectral_centroid``: using the full 2D power spectrum without DC,
       the mean radial frequency weighted by power (center of mass).
    6. ``spectral_variance``: circular variance, ``1 - R``, of the spectral
       orientation weighted by power, where ``R`` is the length of the
       power-weighted mean resultant vector of the doubled orientation angles
       (the spectrum is point-symmetric). 0 means all power lies along one
       orientation; 1 means power is spread evenly across orientations.

    .. [1] Redies, C., Grebenkina, M., Mohseni, M., Kaduhm, A., & Dobel, C. (2020).
       Global Image Properties Predict Ratings of Affective Pictures. Frontiers in Psychology, 11.
       https://doi.org/10.3389/fpsyg.2020.00953
    .. [2] Redies, C., Hasenstein, J., & Denzler, J. (2007).
       Fractal-like image statistics in visual art: Similarity to natural scenes. Spatial Vision, 21(1–2), 137–148.
       https://doi.org/10.1163/156856807782753921
    .. [3] Uran, C., Peter, A., Lazar, A., Barnes, W., Klon-Lipok, J., Shapcott, K. A., Roese, R.,
       Fries, P., Singer, W., & Vinck, M. (2022).
       Predictive coding of natural images by V1 firing rates and rhythmic synchronization. Neuron, 110(7),
       1240-1257.e8. https://doi.org/10.1016/j.neuron.2022.01.002
    """
    # Validate input parameters
    if size < 2:
        raise ValueError("size must be at least 2.")
    if min_frequency <= 0:
        raise ValueError("min_frequency must be positive.")
    if max_frequency <= min_frequency:
        raise ValueError("max_frequency must exceed min_frequency.")
    if log_bins < 2:
        raise ValueError("log_bins must be at least 2.")
    if max_frequency > size / 2:
        raise ValueError(
            f"max_frequency must not exceed Nyquist frequency (size / 2 = {size / 2})."
        )

    # Preprocess the image
    grayscale = _pad_and_resize_image(image, size=size, pad=pad)
    grayscale -= grayscale.mean()

    frequency, power, magnitude = _radially_averaged_spectrum(grayscale)
    dimensionality = _dimensionality(magnitude)
    centroid, variance = _spectral_centroid_and_variance(grayscale)
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
    deviation = np.mean((y - fitted_y) ** 2)

    features = {
        "fourier_slope": float(slope),
        "fourier_deviation": float(deviation),
        "dimensionality": dimensionality,
        "spectral_centroid": centroid,
        "spectral_variance": variance,
    }
    if return_spectrum:
        return features, np.column_stack([frequency, power])
    return features
