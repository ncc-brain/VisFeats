from importlib.metadata import version as _version

project = "VisFeats"
author = "Qian Chu"
release = _version("visfeats")
version = ".".join(release.split(".")[:2])
html_title = "VisFeats"

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "numpydoc",
]
autosummary_generate = True
numpydoc_show_class_members = False
# resmem pulls in PyTorch; it is not needed to build the docs
autodoc_mock_imports = ["resmem"]

html_theme = "pydata_sphinx_theme"
html_theme_options = {
    "github_url": "https://github.com/ncc-brain/VisFeats",
}

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "pillow": ("https://pillow.readthedocs.io/en/stable", None),
    "skimage": ("https://scikit-image.org/docs/stable", None),
    "scipy": ("https://docs.scipy.org/doc/scipy", None),
}
