from PIL import Image
from skimage import color
import numpy as np
from scipy.stats import circmean


def mean_hsv(image: Image.Image) -> dict:
    """Compute the mean hue, saturation, and value of an image.

    The image is converted to RGB, then to HSV with
    `skimage.color.rgb2hsv` (all channels in [0, 1]). Saturation and value
    are the arithmetic means over all pixels. Hue is an angle, so it is
    averaged with a circular mean.

    Parameters
    ----------
    image : PIL.Image.Image
        Input image.

    Returns
    -------
    dict of {str: float}
        Dictionary with the keys:

        - ``"hue"``: circular mean hue in [0, 1).
        - ``"saturation"``: mean saturation in [0, 1].
        - ``"value"``: mean value (brightness) in [0, 1].
    """
    rgb = image.convert("RGB")
    array = np.array(rgb)
    hsv = color.rgb2hsv(array)

    mean_hue = float(circmean(hsv[:, :, 0], high=1.0, low=0.0))
    mean_saturation = float(hsv[:, :, 1].mean())
    mean_value = float(hsv[:, :, 2].mean())
    return {"hue": mean_hue, "saturation": mean_saturation, "value": mean_value}


def rms(image: Image.Image) -> float:
    """Compute the root-mean-square (RMS) contrast of an image.

    The image is converted to 8-bit grayscale and the RMS contrast is the
    population standard deviation of the pixel intensities (values in
    [0, 255]).

    Parameters
    ----------
    image : PIL.Image.Image
        Input image in any Pillow mode; converted to grayscale internally.

    Returns
    -------
    float
        RMS contrast. 0 for a uniform image; 127.5 is the maximum.
    """
    grayscale = image.convert("L")
    return float(np.array(grayscale).std())
