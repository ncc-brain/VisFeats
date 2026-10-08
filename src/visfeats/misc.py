import numpy as np
from PIL import Image


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
