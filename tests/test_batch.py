import logging

import pytest
from PIL import Image

import visfeats


def test_extract_all_features_selects_requested_features():
    image = Image.new("RGB", (2, 2), color=(10, 20, 30))

    features = visfeats.extract_all_features(
        image,
        features=["rms", "hsv_features"],
    )

    assert set(features) == {
        "rms",
        "hue_mean",
        "saturation_mean",
        "value_mean",
        "hue_entropy",
        "saturation_entropy",
        "value_entropy",
    }


def test_extract_all_features_skips_missing_resmem(monkeypatch, caplog):
    # A flat image has no spectral content beyond DC, so use a gradient.
    image = Image.linear_gradient("L").convert("RGB")
    monkeypatch.setattr(visfeats, "find_spec", lambda name: None)

    with caplog.at_level(logging.WARNING, logger="visfeats"):
        features = visfeats.extract_all_features(image)

    assert "rms" in features
    assert "hue_mean" in features
    assert "fourier_slope" in features
    assert "fourier_sigma" in features
    assert "dimensionality" in features
    assert "spectral_centroid" in features
    assert "spectral_variance" in features
    assert "memorability_resmem" not in features
    assert "Skipping memorability_resmem" in caplog.text


def test_extract_all_features_rejects_unknown_names():
    image = Image.new("RGB", (2, 2))

    with pytest.raises(ValueError, match="Unknown feature name"):
        visfeats.extract_all_features(image, features=["unknown"])
