"""MyBookRec package root.

Re-exports `DATA_DIR` and `ROOT_DIR` for backward compatibility with the existing
CLIs that imported them directly. New code should prefer `mybookrec.settings.get_settings()`
for everything (paths, API keys, serving config) so behaviour is controlled in one place.
"""

from __future__ import annotations

import pathlib

from mybookrec.settings import get_settings

_settings = get_settings()

ROOT_DIR: pathlib.Path = pathlib.Path(__file__).resolve().parent.parent
DATA_DIR: pathlib.Path = _settings.data_dir

__all__ = ["ROOT_DIR", "DATA_DIR", "get_settings"]
