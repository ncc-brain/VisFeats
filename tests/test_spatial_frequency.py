import math

import numpy as np
import pytest

from visfeats.spatial_frequency import spatial_frequency_features


@pytest.mark.parametrize(
    "return_spectrum,pad,size",
    [(r, p, s) for r in (False, True) for p in (True, False) for s in (256, 512, 1024)],
)
def test_spatial_frequency_features_on_scikit_images(
    skimage_images, pad, size, return_spectrum
):
    for image in skimage_images:
        kwargs = {"size": size, "max_frequency": 64.0, "pad": pad}
        if return_spectrum:
            features, spectrum = spatial_frequency_features(
                image, return_spectrum=True, **kwargs
            )
            assert spectrum.ndim == 2
            assert spectrum.shape[1] == 2  # (frequency, power)
            assert np.isfinite(spectrum).all()
            assert np.all(np.diff(spectrum[:, 0]) > 0)
            assert np.all(spectrum[:, 0] >= 10.0)  # default min_frequency
            assert np.all(spectrum[:, 0] <= kwargs["max_frequency"])
            assert np.all(spectrum[:, 1] > 0)
        else:
            features = spatial_frequency_features(image, **kwargs)

        assert set(features) == {
            "fourier_slope",
            "fourier_deviation",
            "dimensionality",
            "spectral_centroid",
            "spectral_variance",
        }
        assert all(type(value) is float for value in features.values())
        assert all(math.isfinite(value) for value in features.values())
        assert features["fourier_deviation"] >= 0.0
        assert features["dimensionality"] < 0.0
        assert features["spectral_centroid"] > 0.0
        assert 0.0 <= features["spectral_variance"] <= 1.0
