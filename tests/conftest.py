from typing import List

import pytest
from PIL import Image
from skimage import data

# Bundled with scikit-image, so no network access is needed.
# Covers Pillow modes `1` (horse), `L`, `RGB`, and `RGBA` (logo).
SAMPLE_NAMES = [
    "astronaut",
    "brick",
    "camera",
    "cell",
    "checkerboard",
    "chelsea",
    "clock",
    "coffee",
    "coins",
    "colorwheel",
    "grass",
    "gravel",
    "horse",
    "hubble_deep_field",
    "immunohistochemistry",
    "logo",
    "microaneurysms",
    "moon",
    "page",
    "rocket",
    "text",
]


@pytest.fixture(scope="session")
def skimage_images() -> List[Image.Image]:
    """scikit-image sample images as Pillow images in mixed color modes."""
    return [Image.fromarray(getattr(data, name)()) for name in SAMPLE_NAMES]
