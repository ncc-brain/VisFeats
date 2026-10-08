import math

from PIL import Image

from visfeats.misc import rms


def test_rms_on_scikit_images(skimage_images):
    for image in skimage_images:
        value = rms(image)

        assert type(value) is float
        assert math.isfinite(value)
        assert value >= 0.0


def test_rms_for_uniform_image_is_zero():
    image = Image.new("RGB", (4, 4), color=(100, 150, 200))

    assert rms(image) == 0.0


def test_rms_for_two_level_grayscale_image():
    image = Image.new("L", (2, 1))
    image.putdata([0, 255])

    assert rms(image) == 127.5
