import math

import pytest

from visfeats.spatial_frequency import spatial_frequency_features


@pytest.mark.parametrize("pad", [True, False])
def test_spatial_frequency_features_on_scikit_images(skimage_images, pad):
    for image in skimage_images:
        features = spatial_frequency_features(
            image, size=256, max_frequency=64.0, pad=pad
        )

        assert set(features) == {
            "fourier_slope",
            "fourier_sigma",
            "dimensionality",
            "spectral_centroid",
            "spectral_variance",
        }
        assert all(type(value) is float for value in features.values())
        assert all(math.isfinite(value) for value in features.values())
        assert features["fourier_sigma"] >= 0.0
        assert features["dimensionality"] < 0.0
        assert features["spectral_centroid"] > 0.0
        assert 0.0 <= features["spectral_variance"] <= 1.0
