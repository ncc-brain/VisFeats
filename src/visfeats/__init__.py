import logging
from collections.abc import Sequence
from importlib.util import find_spec
from typing import Dict, Union

from PIL import Image

from .chromatic import hsv_features
from .fourier import fourier_slope_and_sigma
from .memorability import memorability_resmem
from .misc import rms

logger = logging.getLogger(__name__)

_FEATURES = {
    "hsv_features": hsv_features,
    "fourier_slope_and_sigma": fourier_slope_and_sigma,
    "rms": rms,
    "memorability_resmem": memorability_resmem,
}

__all__ = [
    "extract_all_features",
    "fourier_slope_and_sigma",
    "hsv_features",
    "rms",
    "memorability_resmem",
]


def extract_all_features(
    image: Image.Image,
    features: Union[str, Sequence[str]] = "all",
) -> Dict[str, float]:
    """Extract selected visual features from an image.

    Parameters
    ----------
    image : PIL.Image.Image
        Input image in any Pillow mode.
    features : str or sequence of str, default="all"
        ``"all"`` selects every package-level feature. A feature name or
        sequence of feature names selects only those features. Available
        names are ``"hsv_features"``, ``"fourier_slope_and_sigma"``,
        ``"rms"``, and ``"memorability_resmem"``.

    Returns
    -------
    dict of {str: float}
        A flat mapping of feature names to values. Multi-value features
        contribute their own keys. ``memorability_resmem`` is skipped with a
        warning if the optional ``resmem`` package is not installed.

    Raises
    ------
    ValueError
        If any requested feature name is not available.
    """
    if isinstance(features, str):
        selected = list(_FEATURES) if features == "all" else [features]
    else:
        selected = list(dict.fromkeys(features))

    unknown = set(selected) - _FEATURES.keys()
    if unknown:
        names = ", ".join(sorted(unknown))
        raise ValueError(f"Unknown feature name(s): {names}")

    extracted: Dict[str, float] = {}
    for name in selected:
        if name == "memorability_resmem" and find_spec("resmem") is None:
            logger.warning(
                "Skipping memorability_resmem because the optional `resmem` "
                "package is not installed. Install it with: "
                "pip install 'visfeats[resmem]'"
            )
            continue

        result = _FEATURES[name](image)
        if isinstance(result, dict):
            extracted.update(result)
        else:
            extracted[name] = float(result)

    return extracted
