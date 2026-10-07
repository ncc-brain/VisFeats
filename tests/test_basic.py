import math

from PIL import Image

from visfeats.basic import mean_hsv, rms


def test_mean_hsv_on_scikit_images(skimage_images):
    expected_keys = {"hue", "saturation", "value"}

    for image in skimage_images:
        features = mean_hsv(image)

        assert set(features) == expected_keys
        assert all(type(value) is float for value in features.values())
        assert all(math.isfinite(value) for value in features.values())
        assert all(0.0 <= value <= 1.0 for value in features.values())


def test_rms_on_scikit_images(skimage_images):
    for image in skimage_images:
        value = rms(image)

        assert type(value) is float
        assert math.isfinite(value)
        assert value >= 0.0


def test_mean_hsv_for_solid_red_image():
    features = mean_hsv(Image.new("RGB", (4, 4), color=(255, 0, 0)))

    assert min(abs(features["hue"]), abs(features["hue"] - 1.0)) < 1e-7
    assert features["saturation"] == 1.0
    assert features["value"] == 1.0


def test_rms_for_uniform_image_is_zero():
    image = Image.new("RGB", (4, 4), color=(100, 150, 200))

    assert rms(image) == 0.0


def test_rms_for_two_level_grayscale_image():
    image = Image.new("L", (2, 1))
    image.putdata([0, 255])

    assert rms(image) == 127.5
