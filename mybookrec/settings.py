"""Central configuration for MyBookRec.

All runtime knobs live here so callers don't reach into `os.environ` or hard-code
paths. Values come from (in order of precedence) environment variables, `.env`,
then the defaults below.

The settings object is a process-wide singleton accessed via `get_settings()`.
This keeps Pydantic validation costs paid once and lets tests substitute a
custom instance via `set_settings()` in fixtures.
"""

from __future__ import annotations

import pathlib
from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """Runtime configuration for data paths, API credentials, and the serving layer."""

    model_config = SettingsConfigDict(
        env_file=str(REPO_ROOT / ".env"),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ----- paths -----
    data_dir: pathlib.Path = Field(
        default=REPO_ROOT / "data",
        description="Root data directory. Override via MYBOOKREC_DATA_DIR for Colab/Cloud.",
        validation_alias="mybookrec_data_dir",
    )
    checkpoints_dir: pathlib.Path = Field(
        default=REPO_ROOT / "checkpoints",
        description="Where trained .pt and .faiss files live.",
        validation_alias="mybookrec_checkpoints_dir",
    )

    # ----- embeddings -----
    embed_model_name: str = Field(
        default="mixedbread-ai/mxbai-embed-large-v1",
        description=(
            "HF model id for description embeddings. mxbai-embed-large-v1 is Matryoshka-trained "
            "so we can truncate cleanly via `embed_dim`. Top of MTEB at time of writing."
        ),
    )
    embed_dim: int = Field(
        default=512,
        ge=64,
        le=1024,
        description=(
            "Truncate the embedding vector to this many leading dims (Matryoshka). "
            "512 is the sweet spot for mxbai: ~98% of the 1024-dim quality at half the size."
        ),
    )
    embed_model_run: str = Field(
        default="v1_minilm",
        description=(
            "Subdirectory under data/transformed/ that holds the active embedding-model artifacts "
            "(book_embeddings, author_embeddings, item_features, user_features, train_user_features). "
            "Flip to 'v2_mxbai' after the mxbai Colab pass completes — all feature builders + "
            "TransformedArtifacts honour this without code changes."
        ),
    )

    # ----- experiment tracking (MLflow) -----
    mlflow_tracking_uri: str = Field(
        default=f"sqlite:///{REPO_ROOT}/mlruns.db",
        description="MLflow backend store URI. SQLite file by default; swap to http://... for a remote server.",
    )
    mlflow_artifact_root: pathlib.Path = Field(
        default=REPO_ROOT / "mlruns",
        description="Where MLflow writes run artifacts (checkpoints, logs).",
    )
    mlflow_experiment_name: str = Field(
        default="two_tower",
        description="Default experiment name for training runs.",
    )

    # ----- ingestion APIs -----
    google_books_api_key: str | None = Field(
        default=None,
        description="Google Books API key (free, 1k req/day). Optional — set to enable enrichment.",
    )
    openlibrary_user_agent: str = Field(
        default="MyBookRec/0.1 (https://github.com/zali78690/MyBookRec)",
        description="Open Library asks every client to identify itself in User-Agent.",
    )
    ingest_rate_limit_per_sec: float = Field(
        default=5.0,
        description="Max API requests per second per source (defensive default; both APIs allow more).",
        ge=0.1,
        le=50.0,
    )
    ingest_request_timeout_sec: float = Field(
        default=10.0,
        description="HTTP request timeout for ingestion fetchers.",
        gt=0.0,
    )

    # ----- serving -----
    serve_host: str = Field(default="0.0.0.0", description="FastAPI bind host.")
    serve_port: int = Field(default=8000, ge=1, le=65535, description="FastAPI bind port.")
    serve_model_path: pathlib.Path | None = Field(
        default=None,
        description="Checkpoint to load at server start. Defaults to <checkpoints_dir>/two_tower_mac.pt.",
    )
    serve_index_path: pathlib.Path | None = Field(
        default=None,
        description="Pre-built FAISS index. If None, built on the fly at startup (~2s).",
    )
    serve_default_top_k: int = Field(default=10, ge=1, le=200)
    serve_oversample: int = Field(
        default=5,
        ge=1,
        le=50,
        description="Pull K*oversample candidates from FAISS before post-rank filters.",
    )
    serve_min_avg_rating: float = Field(default=3.5, ge=0.0, le=5.0)

    # ----- medallion layer roots -----
    @property
    def raw_dir(self) -> pathlib.Path:
        """UCSD-style raw inputs (gitignored, DVC-tracked).

        Returns:
            Path to data/raw/.
        """
        return self.data_dir / "raw"

    @property
    def transformed_dir(self) -> pathlib.Path:
        """Backwards-compatible transformed layer (the existing pipeline writes here).

        Returns:
            Path to data/transformed/.
        """
        return self.data_dir / "transformed"

    @property
    def bronze_dir(self) -> pathlib.Path:
        """Bronze layer: raw API responses, immutable, partitioned by source/date.

        Returns:
            Path to data/bronze/.
        """
        return self.data_dir / "bronze"

    @property
    def silver_dir(self) -> pathlib.Path:
        """Silver layer: cleaned parquets matching the UCSD training schema.

        Returns:
            Path to data/silver/.
        """
        return self.data_dir / "silver"

    @property
    def gold_dir(self) -> pathlib.Path:
        """Gold layer: feature-ready artifacts the serving layer loads.

        Returns:
            Path to data/gold/.
        """
        return self.data_dir / "gold"

    def resolved_serve_model_path(self) -> pathlib.Path:
        """Concrete model path with default fallback.

        Returns:
            `serve_model_path` if set, else `<checkpoints_dir>/two_tower_mac.pt`.
        """
        if self.serve_model_path is not None:
            return self.serve_model_path
        return self.checkpoints_dir / "two_tower_mac.pt"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide Settings singleton.

    Cached so Pydantic validation runs once per process. Tests that need
    overrides should call `get_settings.cache_clear()` after setting env vars.

    Returns:
        The shared Settings instance.
    """
    return Settings()
