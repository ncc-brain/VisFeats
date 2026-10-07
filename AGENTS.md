# AGENTS.md

Guidance for AI agents and contributors working on VisFeats, a library for extracting global visual features from images.

## Project layout

- `src/visfeats/` - library code, one module per feature family (`simple.py`, `fourier.py`, `spectral.py`, `phog.py`, `memorability.py`). Shared helpers live in `_utils.py`.
- `src/visfeats/__init__.py` - public API, including `extract_all_features`.
- `tests/` - pytest suite; `tests/conftest.py` provides shared fixtures.
- `setup.cfg` / `pyproject.toml` - packaging (setuptools, `src` layout) and dependencies. Install with `pip install -e ".[dev]"`. `resmem` (and its PyTorch dependency) is optional: it lives in the `memorability` extra and is imported lazily, so core features must work without it.

## Core conventions

### Input type: Pillow images

- Public feature functions take a `PIL.Image.Image` as their input, not a NumPy array. (Update type hints and docstrings that still say "image array".)
- Exceptions are allowed only when a function is inherently array-based (for example a helper operating on an intermediate spectrum). Keep those private (underscore-prefixed) or in `_utils.py`.
- Return plain Python types (`float`, or a dictionary mapping strings to floats), not NumPy scalars, so results are serializable.
- A function that computes a single value returns a `float`. Return a dictionary only when several related values are computed together (for example `mean_hsv` returning hue, saturation, and value).

### Color mode: convert inside each function

- Never assume the caller's mode. Pillow images may be `1`, `L`, `P`, `RGB`, `RGBA`, and so on.
- Every function converts to the mode it needs as its first step, then converts to an array:
  - color features (HSV, etc.): `image.convert("RGB")`
  - grayscale features (contrast, Fourier, PHOG, etc.): `image.convert("L")`
- Do this per function rather than relying on upstream conversion, so each function works standalone.
- Mind value ranges: after `np.asarray`, `uint8` data is in `[0, 255]`. skimage converts integer images to `[0, 1]` floats where needed. If you build floats yourself, normalize explicitly and note it in a comment.

### Python version compatibility

- `pyproject.toml` declares `requires-python = ">=3.6"`. All library code (`src/`) and tests must stay runnable on Python 3.6; check the declared minimum before using any newer feature.
- Avoid syntax and stdlib features newer than 3.6, including:
  - built-in generics in annotations (`list[int]`, `dict[str, float]`, `tuple[int, ...]`); use `typing.List`, `typing.Dict`, `typing.Tuple`
  - `X | Y` union types; use `typing.Optional` / `typing.Union`
  - `from __future__ import annotations` (3.7+), the walrus operator `:=` (3.8+), positional-only parameters `/` (3.8+), `match` statements (3.10+)
  - `dataclasses` (3.7+), `functools.cached_property` (3.8+), `math.prod` (3.8+), `str.removeprefix` (3.9+), `zip(strict=True)` (3.10+)
- f-strings are fine (3.6). Use `typing` imports rather than newer typing helpers (`Literal`, `Protocol`, `TypedDict` are 3.8+).
- Keep dependency requirements compatible with the declared minimum; do not add a dependency that drops Python 3.6 without raising `requires-python` deliberately and updating this file.
- Known violations to fix when touched: `dict[str, float]` in `src/visfeats/__init__.py`, and `list[...]` annotations in `tests/conftest.py`.

### Keep implementations simple

- Prefer existing `skimage` and `scipy` (and NumPy) functions over hand-written algorithms. Do not reinvent wheels. Examples: `skimage.color.rgb2hsv`, `skimage.feature.hog`, `scipy.stats.circmean`, `scipy.fft`.
- Choose the shortest readable implementation that is correct. Avoid unnecessary abstraction, premature optimization, and clever vectorization tricks.
- Add a custom implementation only when no suitable library function exists, and add a short comment explaining why. Cite the paper or reference for the feature in the docstring or README.
- Keep functions small and single-purpose. Avoid new dependencies unless clearly justified; declare any in `pyproject.toml`.
- Comment only what needs clarification.

## Testing

- Use `pytest`. Run with `python -m pytest`.
- Use the `skimage_images` fixture from `tests/conftest.py`. It provides scikit-image's bundled sample images as a list of `Image.Image` objects in mixed color modes (`1`, `L`, `RGB`, `RGBA`), which also exercises color-mode handling.
- Every feature function needs at least a smoke test: it runs on every example image without raising and returns the expected structure (keys, finite float values). Prefer `pytest.mark.parametrize` or a loop over the fixture.
- Do not commit test images; the fixture builds Pillow images in memory from scikit-image's bundled `skimage.data`. Tests should not need network access.
- For deterministic features, add a small targeted test (for example a uniform image gives zero contrast) in addition to the smoke test.
- Test code must also follow the Python 3.6 compatibility rules above.
- Do not weaken or remove tests to make a change pass.

## Code style

- Formatting is enforced by `ruff format` through pre-commit. Set it up once with `pip install pre-commit; pre-commit install`, and run `pre-commit run --all-files` before pushing.
- Use type hints on public functions and NumPy-style docstrings (`Parameters`, `Returns`, and `References` when applicable). Start with a one-line summary, then briefly describe how the value is computed (color conversion, library function, aggregation). In `Returns`, document the dictionary keys when a dictionary is returned.
- Multi-value feature functions return a flat dictionary mapping strings to floats, with stable, descriptive key names. `extract_all_features` builds the final flat dictionary, naming single-value features itself (for example `{"rms": rms(image)}`), so keys merge without collisions.

## Adding a new feature

1. Add a function in the appropriate module (or a new one) taking `image: Image.Image`.
2. Convert to the required color mode at the top of the function.
3. Implement with `skimage`/`scipy` where possible.
4. Export it in `src/visfeats/__init__.py` and include it in `extract_all_features`.
5. Add a smoke test over `skimage_images`, then run `python -m pytest` and `pre-commit run --all-files`.
6. Document the feature, with its reference, in `README.md`.

## Sustainability checklist

- Small, focused changes; do not mix refactors with feature work.
- Keep README and docstrings in sync with behavior.
- Pin or bound dependencies only when needed, and keep the dependency list minimal.
- No secrets, large binaries, or generated caches in the repository.
