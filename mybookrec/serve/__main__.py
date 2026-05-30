"""Uvicorn entrypoint for the MyBookRec serving API.

Usage:
    .venv/bin/python -m mybookrec.serve

Honours `serve_host` and `serve_port` from `mybookrec.settings`.
"""

from __future__ import annotations

import uvicorn

from mybookrec.settings import get_settings


def main() -> None:
    """Run uvicorn against `mybookrec.serve.app:app`."""
    settings = get_settings()
    uvicorn.run(
        "mybookrec.serve.app:app",
        host=settings.serve_host,
        port=settings.serve_port,
    )


if __name__ == "__main__":
    main()
