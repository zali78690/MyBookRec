# syntax=docker/dockerfile:1.7
#
# Multi-stage build for the MyBookRec serving API.
#
# Stage 1 (`builder`): uses the official `uv` image to resolve and install Python deps from
# the pinned uv.lock into a system venv at /opt/venv. The full toolchain (uv + git + build
# headers) stays in this stage and never ships in the final image.
#
# Stage 2 (`runtime`): a slim python:3.12-slim layer that copies just the resolved venv +
# package source. No uv, no apt build tools — ~250 MB instead of ~1.2 GB.
#
# Build:   docker build -t mybookrec-serve:latest .
# Run:     docker run --rm -p 8000:8000 \
#              -v "$PWD/data:/app/data:ro" -v "$PWD/checkpoints:/app/checkpoints:ro" \
#              --env-file .env mybookrec-serve:latest
#
# Healthcheck: curl -fsS http://localhost:8000/healthz

ARG PYTHON_VERSION=3.12-slim
ARG UV_VERSION=0.11.6

# ---------- Stage 1: builder ----------
FROM ghcr.io/astral-sh/uv:${UV_VERSION} AS uv

FROM python:${PYTHON_VERSION} AS builder

COPY --from=uv /uv /usr/local/bin/uv

ENV UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_PYTHON_DOWNLOADS=never \
    VIRTUAL_ENV=/opt/venv

WORKDIR /app

# Install only deps first so the layer caches across source changes.
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --no-dev

# Now copy source + reinstall to register the project itself in the venv.
COPY mybookrec ./mybookrec
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

# ---------- Stage 2: runtime ----------
FROM python:${PYTHON_VERSION} AS runtime

# Run as non-root for the standard security reason.
RUN useradd --create-home --shell /usr/sbin/nologin app

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    KMP_DUPLICATE_LIB_OK=TRUE

WORKDIR /app

COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /app/mybookrec /app/mybookrec
COPY pyproject.toml /app/pyproject.toml

USER app

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request, sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/healthz').status == 200 else 1)"

CMD ["python", "-m", "mybookrec.serve"]
