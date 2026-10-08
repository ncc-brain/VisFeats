# VisFeats: Extraction of Global Visual Features from Images

![Python Version](https://img.shields.io/badge/python-3.10%2B-blue)

VisFeats is a library for extracting global visual features from images.

## Installation

VisFeats is installable directly from GitHub:

```bash
pip install git+https://github.com/qian-chu/visfeats.git
```

## Usage

To use VisFeats, first load an image using PIL:

```python
from PIL import Image

image = Image.open("path/to/your/image.jpg")
```

All functions in VisFeats take a PIL ``Image.Image`` as input and return the corresponding feature values. For instance:

### Basic features

- Hue, saturation, and value (HSV)
- Root-mean-square (RMS) Contrast

## Memorability

Estimates intrinsic memorability using ResMem pretrained model described by [Needell and Bainbridge (2022)](https://doi.org/10.1007/s42113-022-00126-5). To estimate memorability, install the `resmem` package:

```bash
pip install resmem"
```

Then you can use the `memorability_resmem` function to estimate the memorability of an image:

```python
from visfeats import memorability_resmem

resmem_score = memorability_resmem(image)
```
