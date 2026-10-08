from PIL import Image
from skimage import color
import numpy as np
from scipy.stats import circmean, entropy

HSV_RANGE = (0.0, 1.0)


def hsv_features(image: Image.Image, bins: int = 100) -> dict:
    """Compute the mean and entropy of the HSV channels of an image.

    Parameters
    ----------
    image : PIL.Image.Image
        Input image.
    bins : int, default=100
        Number of bins for the histograms used to compute entropy.

    Returns
    -------
    dict of {str: float}
        A mapping of feature names to values:

        ==================  ======================================
        Key                 Description
        ==================  ======================================
        hue_mean            Circular mean hue in [0, 1).
        saturation_mean     Mean saturation in [0, 1].
        value_mean          Mean value (brightness) in [0, 1].
        hue_entropy         Entropy of the hue histogram.
        saturation_entropy  Entropy of the saturation histogram.
        value_entropy       Entropy of the value histogram.
        ==================  ======================================

    Notes
    -----
    The image is converted to RGB, then to HSV using
    ``skimage.color.rgb2hsv``. The mean and histogram entropy are computed
    for each channel.
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

    return {
        "hue_mean": float(circmean(hue, high=1.0, low=0.0)),  # circular mean of hue
        "saturation_mean": float(saturation.mean()),
        "value_mean": float(value.mean()),
        "hue_entropy": float(entropy(hue_pk[0])),
        "saturation_entropy": float(entropy(saturation_pk[0])),
        "value_entropy": float(entropy(value_pk[0])),
    }
