import math

import numpy as np
import pytest
from PIL import Image

from visfeats.chromatic import hsv_features


@pytest.mark.parametrize("return_hsv", [False, True])
def test_hsv_features_on_scikit_images(skimage_images, return_hsv):
    expected_keys = {
        "hue_mean",
        "saturation_mean",
        "value_mean",
        "hue_entropy",
        "saturation_entropy",
        "value_entropy",
    }

    for image in skimage_images:
        if return_hsv:
            features, hsv = hsv_features(image, return_hsv=return_hsv)
            assert hsv.shape[:2] == image.size[::-1]  # (height, width)
            assert hsv.shape[2] == 3  # (channel)
            assert np.all((0.0 <= hsv) & (hsv <= 1.0))  # HSV values are in [0, 1]
        else:
            features = hsv_features(image, return_hsv=return_hsv)

        assert set(features) == expected_keys
        assert all(type(value) is float for value in features.values())
        assert all(math.isfinite(value) for value in features.values())
        assert all(
            0.0 <= features[key] <= 1.0
            for key in (
                "hue_mean",
                "saturation_mean",
                "value_mean",
            )
        )
        assert all(
            features[key] >= 0.0
            for key in (
                "hue_entropy",
                "saturation_entropy",
                "value_entropy",
            )
        )


def test_hsv_features_for_solid_red_image():
    features = hsv_features(Image.new("RGB", (4, 4), color=(255, 0, 0)))

    assert np.isclose(features["hue_mean"], 0.0) or np.isclose(
        features["hue_mean"], 1.0
    )
    assert features["saturation_mean"] == 1.0
    assert features["value_mean"] == 1.0
    assert features["hue_entropy"] == 0.0
    assert features["saturation_entropy"] == 0.0
    assert features["value_entropy"] == 0.0


def test_hsv_features_return_hsv_array():
    image = Image.new("RGB", (5, 3), color=(255, 0, 0))
    features, hsv = hsv_features(image, return_hsv=True)

    assert features == hsv_features(image)
    assert hsv.shape == (3, 5, 3)  # (height, width, channel)
    assert np.allclose(hsv[..., 1], 1.0)
    assert np.allclose(hsv[..., 2], 1.0)
