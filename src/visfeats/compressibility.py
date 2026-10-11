from io import BytesIO

from PIL import Image


def compressibility_pillow(
    image: Image.Image, *, fmt: str = "PNG", **save_kwargs
) -> float:
    """Estimate the compressibility of an image from its encoded size.

    The image is converted to RGB and encoded in memory with Pillow (no file
    is written). Compressibility is the negative of the encoded size in bits
    per pixel, so more compressible (simpler) images get higher values.

    Parameters
    ----------
    image : Image
        Input image in any Pillow mode; converted to RGB internally.
    fmt : str, default="PNG"
        Pillow format used for encoding. Use a lossless format (such as
        ``"PNG"``) so the size reflects image redundancy rather than a
        quality setting.
    **save_kwargs
        Additional keyword arguments passed to ``Image.save``, for example
        ``compress_level`` for PNG.

    Returns
    -------
    float
        Negative encoded size in bits per pixel. Always negative; values
        closer to 0 indicate a more compressible image. The value depends on
        ``fmt`` and ``save_kwargs``, so only compare images analysed with the
        same settings.
    """
    # Choose a consistent pixel representation for comparison.
    image = image.convert("RGB")

    buffer = BytesIO()
    image.save(buffer, format=fmt, **save_kwargs)

    width, height = image.size
    compressed_bits = 8 * buffer.tell()  # number of bytes written * 8

    bpp = compressed_bits / (width * height)
    return float(-bpp)
