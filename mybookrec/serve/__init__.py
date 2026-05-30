"""FastAPI serving layer for MyBookRec.

Exposes a real-time `/recommend` endpoint backed by the trained two-tower model
and a FAISS index. See `app.py` for the application object and `__main__.py`
for the uvicorn entrypoint (`.venv/bin/python -m mybookrec.serve`).
"""
