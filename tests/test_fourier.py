import math

from visfeats.fourier import fourier_slope_and_sigma


def test_fourier_slope_and_sigma_on_scikit_images(skimage_images):
    for image in skimage_images:
        features = fourier_slope_and_sigma(image, size=256, max_frequency=64.0)

        assert set(features) == {"fourier_slope", "fourier_sigma"}
        assert all(type(value) is float for value in features.values())
        assert all(math.isfinite(value) for value in features.values())
        assert features["fourier_sigma"] >= 0.0
