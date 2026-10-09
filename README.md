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
dictionary.

```python
from visfeats import extract_all_features

features = extract_all_features(image, features=["rms", "hsv_features"])
```

### Mean and entropy of HSV channels

Chromatic features describe color in HSV (Hue, Saturation, Value) space, including channel means and entropies. The `hsv_features` function returns a dictionary with six keys:

- `hue_mean`
- `saturation_mean`
- `value_mean`
- `hue_entropy`
- `saturation_entropy`
- `value_entropy`

```python
from visfeats import hsv_features
features = hsv_features(image)
```

### Spatial frequency features

Spatial frequency features capture the distribution of different spatial frequencies in an image, which relate to the level of detail and texture. The `spatial_frequency_features` function extracts these features:

- `fourier_slope`
- `fourier_sigma`
- `dimensionality`
- `spectral_centroid`
- `spectral_variance`

```python
from visfeats import spatial_frequency_features
features = spatial_frequency_features(image)
```

### Memorability

While memorability is often considered a subjective property, recent research has shown that it can be predicted from image content. Here the `memorability_resmem` function estimates intrinsic memorability using the pretrained ResMem model described by [Needell and Bainbridge (2022)](https://doi.org/10.1007/s42113-022-00126-5).

ResMem is an optional dependency. It can be installed along with VisFeats using:

```bash
pip install "visfeats[resmem]"
```

Or separately with:

```bash
pip install resmem
```

Then you can use the `memorability_resmem` function to estimate the memorability of an image:

```python
from visfeats import memorability_resmem
resmem_score = memorability_resmem(image)
```

### Other features

#### RMS contrast

The `rms` function computes the root-mean-square contrast as the population standard deviation of the image's grayscale pixel intensities.

```python
from visfeats import rms
rms_contrast = rms(image)
```
