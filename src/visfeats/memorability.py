from functools import lru_cache

from PIL import Image


@lru_cache(maxsize=1)
def _load_resmem():
    # Import within function so resmem/torch load only when needed.
    try:
        from resmem import ResMem, transformer
    except ImportError as error:
        raise ImportError(
            "memorability_resmem requires the optional dependency `resmem`. "
            "Install it with: pip install 'visfeats[memorability]'"
        ) from error

    model = ResMem(pretrained=True)
    model.eval()
    return model, transformer


def memorability_resmem(image: Image.Image) -> float:
    """Estimate the intrinsic memorability of an image.

    The image is converted to RGB, preprocessed with the ResMem transformer
    (resized to 227 x 227 and normalized), and passed through the pretrained
    ResMem network, a residual neural network trained to predict human
    memorability scores (Needell and Bainbridge, 2022). The model is loaded
    once per process and reused across calls.

    Parameters
    ----------
    image : PIL.Image.Image
        Input image in any Pillow mode; converted to RGB internally.

    Returns
    -------
    float
        Predicted memorability score in [0, 1]; higher values indicate a
        more memorable image.

    References
    ----------
    .. [1] Needell, C. D., & Bainbridge, W. A. (2022).
        Embracing new techniques in deep learning for estimating image memorability.
        Computational Brain & Behavior. https://doi.org/10.1007/s42113-022-00126-5
    """
    model, transformer = _load_resmem()
    image = image.convert("RGB")
    image_x = transformer(image)
    prediction = model(image_x.view(-1, 3, 227, 227))
    return float(prediction.detach().numpy().squeeze())
