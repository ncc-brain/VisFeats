import math

from PIL import Image

from visfeats import compressibility_pillow


def test_compressibility_on_scikit_images(skimage_images):
    for image in skimage_images:
        value = compressibility_pillow(image)

        assert type(value) is float
        assert math.isfinite(value)
        assert value < 0.0


def test_compressibility_uniform_image_more_compressible_than_noise():
    uniform = Image.new("RGB", (64, 64), color=(10, 20, 30))
    noise = Image.effect_noise((64, 64), 50).convert("RGB")

    assert compressibility_pillow(uniform) > compressibility_pillow(noise)
