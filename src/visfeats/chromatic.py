from PIL import Image
from skimage import color
import numpy as np
from scipy.stats import circmean, entropy

HSV_RANGE = (0.0, 1.0)


def hsv_features(image: Image.Image, bins: int = 100) -> dict:
    """Compute the mean and entropy of the HSV channels of an image.

    The image is internally converted to RGB, then to HSV with
    `skimage.color.rgb2hsv`.

    Parameters
    ----------
    image : PIL.Image.Image
        Input image.
    bins : int
        Number of bins for the histograms used to compute entropy.
        Defaults to 100.

    Returns
    -------
    dict of {str: float}
        Dictionary with the keys:

        - ``"hue_mean"``: circular mean hue in [0, 1).
        - ``"saturation_mean"``: mean saturation in [0, 1].
        - ``"value_mean"``: mean value (brightness) in [0, 1].
        - ``"hue_entropy"``: entropy of the hue histogram.
        - ``"saturation_entropy"``: entropy of the saturation histogram.
        - ``"value_entropy"``: entropy of the value histogram.
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
