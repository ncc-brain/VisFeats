import math

import numpy as np
from PIL import Image

from visfeats.chromatic import hsv_features


def test_hsv_features_on_scikit_images(skimage_images):
    expected_keys = {
        "hue_mean",
        "saturation_mean",
        "value_mean",
        "hue_entropy",
        "saturation_entropy",
        "value_entropy",
    }

    for image in skimage_images:
        features = hsv_features(image)

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
