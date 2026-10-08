# AGENTS.md

Guidance for AI agents and contributors working on VisFeats, a library for extracting global visual features from images.

## Project layout

- `src/visfeats/` - library code organized by feature family: `chromatic.py`, `fourier.py`, `memorability.py`, `misc.py`, `phog.py`, and `spectral.py`. Shared array conversion helpers live in `_utils.py`.
- `src/visfeats/__init__.py` - package-level public API, including `extract_all_features`.
- `tests/` - pytest suite; `tests/conftest.py` provides shared fixtures.
- `pyproject.toml` - packaging (setuptools, `src` layout), runtime dependencies, and optional extras. Install development dependencies with `pip install -e ".[dev]"`. `resmem` (and its PyTorch dependency) is optional: it is imported lazily by `memorability_resmem` and is available through the `resmem` extra.

## Core conventions

### Input type: Pillow images

- Image-based feature functions take a `PIL.Image.Image` as input, not a NumPy array. The array-based `spectral_features` function is an exception.
- Keep other array-based helpers private (underscore-prefixed) or in `_utils.py`.
- Return plain Python types (`float`, or a dictionary mapping strings to floats), not NumPy scalars, so results are serializable.
- A function that computes a single value returns a `float`. Return a dictionary only when several related values are computed together (for example `hsv_features` returning HSV means and entropies).

### Image color-mode handling

- Never assume the caller's Pillow mode. Images may be `1`, `L`, `P`, `RGB`, `RGBA`, and so on.
- Each image-based feature converts to the mode it needs before converting to an array:
  - color features (HSV, etc.): `image.convert("RGB")`
  - grayscale features (contrast, Fourier, PHOG, etc.): `image.convert("L")`
- Do this per function rather than relying on upstream conversion, so each function works standalone.
- Mind value ranges: after `np.asarray`, `uint8` data is in `[0, 255]`. skimage converts integer images to `[0, 1]` floats where needed. If you build floats yourself, normalize explicitly and note it in a comment.

### Python version compatibility

- `pyproject.toml` declares `requires-python = ">=3.10"`, and CI tests 3.10-3.14. All library code (`src/`) and tests must run on every version in that range; do not use features newer than 3.10.
- Built-in generics (`list[int]`) and `X | Y` unions are fine.
- Keep dependency requirements compatible with the declared minimum.

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
- Test code must run on every Python version supported by the project (currently Python 3.10-3.14 in CI).
- Do not weaken or remove tests to make a change pass.

## Code style

- Formatting is enforced by `ruff format` through pre-commit. Set it up once with `pip install pre-commit; pre-commit install`, and run `pre-commit run --all-files` before pushing.
- Use type hints on public functions and NumPy-style docstrings (`Parameters`, `Returns`, and `References` when applicable). Start with a one-line summary, then briefly describe how the value is computed (color conversion, library function, aggregation). In `Returns`, document dictionary keys in a table. Private helpers may use concise docstrings without full numpydoc sections.
- In `Parameters`, document defaults after the type as `name : type, default=value` (for example, `bins : int, default=100`); do not repeat the default in the description.
- Put multi-step processing details in a NumPy-style `Notes` section rather than before `Parameters`.
- Multi-value feature functions return a flat dictionary mapping strings to floats, with stable, descriptive key names.

## Adding a new feature

1. Add a function in the appropriate module (or a new one) taking `image: Image.Image`.
2. Convert to the required color mode at the top of the function.
3. Implement with `skimage`/`scipy` where possible.
4. Export it from `src/visfeats/__init__.py` if it is intended to be part of the package-level public API.
5. Add a smoke test over `skimage_images`, then run `python -m pytest` and `pre-commit run --all-files`.
6. Document the feature, with its reference, in `README.md`.

## Sustainability checklist

- Small, focused changes; do not mix refactors with feature work.
- Keep README and docstrings in sync with behavior.
- Pin or bound dependencies only when needed, and keep the dependency list minimal.
- No secrets, large binaries, or generated caches in the repository.
