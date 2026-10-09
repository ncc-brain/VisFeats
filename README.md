# VisFeats: Extraction of Global Visual Features from Images

[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
![Python Version](https://img.shields.io/badge/python-3.10%2B-blue)

VisFeats is a library for extracting global visual features from images.

## Installation

VisFeats is installable via PyPI:

```bash
pip install visfeats
```

Or via GitHub (for the latest development version):

```bash
pip install git+https://github.com/ncc-brain/VisFeats.git
```

## Usage

To use VisFeats, first load an image using PIL:

```python
from PIL import Image
image = Image.open("path/to/your/image.jpg")
```

All functions in VisFeats take a PIL ``Image.Image`` as input and return the corresponding feature values. Available features are described below.

### Extract multiple features

Use `extract_all_features` to collect all package-level features, or pass a
feature name or list of names to select a subset. The result is a flat
dictionary. If `resmem` is not installed, its memorability feature is skipped
with a warning.

```python
from visfeats import extract_all_features

features = extract_all_features(image, features=["rms", "hsv_features"])
```

### Mean and entropy of HSV channels

Chromatic features include the mean and entropy of the HSV channels. The `hsv_features` function returns a dictionary with six keys: `hue_mean`, `saturation_mean`, `value_mean`, `hue_entropy`, `saturation_entropy`, and `value_entropy`.

```python
from visfeats import hsv_features
hsv_features = hsv_features(image)
```

### Spatial frequency features

The `spatial_frequency_features` function returns a dictionary with five keys: `fourier_slope`, `fourier_sigma`, `dimensionality`, `spectral_centroid`, and `spectral_variance`. Dimensionality is determined by taking the two-dimensional fast Fourier transform of the image, rotationally averaging it over the entire frequency domain, and ranking the spectral components by magnitude. It is the slope of the resulting spectrum (log magnitude versus log rank). Spectral centroid is the center of mass of the power spectrum, i.e. the mean spatial frequency weighted by the power in each frequency band. Spectral variance is the circular variance of the power spectrum over orientation (0 when power is concentrated along one orientation, 1 when it is spread evenly).

```python
from visfeats import spatial_frequency_features
spatial_frequency = spatial_frequency_features(image)
```

### Memorability

Estimates intrinsic memorability using ResMem pretrained model described by [Needell and Bainbridge (2022)](https://doi.org/10.1007/s42113-022-00126-5). To estimate memorability, install the `resmem` package:

```bash
pip install resmem
```

Then you can use the `memorability_resmem` function to estimate the memorability of an image:

```python
from visfeats import memorability_resmem
resmem_score = memorability_resmem(image)
```

### Other features

- Root-mean-square (RMS) Contrast
