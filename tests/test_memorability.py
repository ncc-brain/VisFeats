import math

from visfeats import memorability_resmem

def test_resmem_on_scikit_images(skimage_images):
    for image in skimage_images:
        value = memorability_resmem(image)

        assert type(value) is float
        assert math.isfinite(value)
        assert 0.0 <= value <= 1.0