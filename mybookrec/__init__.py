"""Base folder for book recommendation source code."""

import os
import pathlib

ROOT_DIR = pathlib.Path(__file__).resolve().parent.parent
# DATA_DIR defaults to <repo>/data but can be overridden via MYBOOKREC_DATA_DIR.
# Useful on Colab where the package code is cloned to /content/ but data lives on Drive.
DATA_DIR = pathlib.Path(os.environ.get("MYBOOKREC_DATA_DIR", str(ROOT_DIR / "data")))
