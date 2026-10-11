from typing import Literal, overload

from PIL import Image
from skimage import color
import numpy as np
from numpy.typing import NDArray
from scipy.stats import circmean, entropy

HSV_RANGE = (0.0, 1.0)


@overload
def hsv_features(
    image: Image.Image, bins: int = ..., return_hsv: Literal[False] = ...
) -> dict[str, float]: ...


@overload
def hsv_features(
    image: Image.Image, bins: int = ..., *, return_hsv: Literal[True]
) -> tuple[dict[str, float], NDArray[np.float64]]: ...


def hsv_features(
    image: Image.Image, bins: int = 100, return_hsv: bool = False
) -> dict[str, float] | tuple[dict[str, float], NDArray[np.float64]]:
    """Compute the mean and entropy of the HSV channels of an image.

    Parameters
    ----------
    image : Image
        Input image.
    bins : int, default=100
        Number of bins for the histograms used to compute entropy.
    return_hsv : bool, default=False
        If True, also return the per-pixel HSV array (see Returns).

    Returns
    -------
    features : dict of {str: float}
        A mapping of feature names to values:

        .. list-table::
           :header-rows: 1
           :widths: 30 70

           * - Key
             - Description
           * - ``hue_mean``
             - Circular mean hue in [0, 1).
           * - ``saturation_mean``
             - Mean saturation in [0, 1].
           * - ``value_mean``
             - Mean value (brightness) in [0, 1].
           * - ``hue_entropy``
             - Entropy of the hue histogram.
           * - ``saturation_entropy``
             - Entropy of the saturation histogram.
           * - ``value_entropy``
             - Entropy of the value histogram.

    hsv : ndarray, shape (height, width, 3)
        Raw per-pixel HSV values.
        Only returned when ``return_hsv=True``.

    Notes
    -----
    The image is converted to RGB, then to HSV using
    :func:`skimage.color.rgb2hsv`.
    """
    rgb = image.convert("RGB")
    array = np.array(rgb)
    hsv = color.rgb2hsv(array)

    hue = hsv[:, :, 0]
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]

    hue_pk = np.histogram(hue, bins=bins, range=HSV_RANGE, density=True)
    saturation_pk = np.histogram(saturation, bins=bins, range=HSV_RANGE, density=True)
    value_pk = np.histogram(value, bins=bins, range=HSV_RANGE, density=True)

    features = {
        "hue_mean": float(circmean(hue, high=1.0, low=0.0)),  # circular mean of hue
        "saturation_mean": float(saturation.mean()),
        "value_mean": float(value.mean()),
        "hue_entropy": float(entropy(hue_pk[0])),
        "saturation_entropy": float(entropy(saturation_pk[0])),
        "value_entropy": float(entropy(value_pk[0])),
    }
    return (features, hsv) if return_hsv else features
