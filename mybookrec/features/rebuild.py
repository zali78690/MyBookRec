"""One-step regeneration of every embedding-model-specific feature artifact.

When you swap embedding models (e.g. MiniLM → mxbai), the only thing that comes
from Colab is `book_embeddings.npy` in the new model_run dir. Everything
downstream — author embeddings, item features, user features (personal + bulk),
train_user_features — is a deterministic function of that file plus the
model-independent `shared/` artifacts.

This script kicks off the three feature builders in dependency order, each
sub-shelled with `MYBOOKREC_EMBED_MODEL_RUN=<target>` so they read + write
under the right subdirectory. Each step is independently re-runnable.

Usage:
    .venv/bin/python -m mybookrec.features.rebuild --model-run v2_mxbai
    .venv/bin/python -m mybookrec.features.rebuild --model-run v1_minilm  # default
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time

from mybookrec.settings import get_settings

# Order matters: build_user_features + build_train_user_features write the *_basic
# variants; build_author_features then reads those plus book_embeddings to produce
# the unsuffixed (with-author) variants.
BUILDERS: tuple[str, ...] = (
    "mybookrec.features.build_user_features",
    "mybookrec.features.build_train_user_features",
    "mybookrec.features.build_author_features",
)


def parse_args() -> argparse.Namespace:
    """Parse CLI args.

    Returns:
        argparse.Namespace with model_run + optional --skip / --only filters.
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--model-run",
        default=None,
        help=(
            "Target subdir under data/transformed/ (e.g. v1_minilm, v2_mxbai). Defaults to settings.embed_model_run."
        ),
    )
    parser.add_argument(
        "--only",
        action="append",
        choices=BUILDERS,
        default=None,
        help="Restrict to the named builder(s) (repeatable). Default: run all three in order.",
    )
    return parser.parse_args()


def run_builder(module: str, env: dict[str, str]) -> None:
    """Sub-shell one feature builder, surfacing its stdout/stderr live.

    Args:
        module: Dotted module path (e.g. "mybookrec.features.build_user_features").
        env: Environment dict to pass to the subprocess (MYBOOKREC_EMBED_MODEL_RUN
            should be set).

    Raises:
        SystemExit: If the subprocess returns a non-zero exit code.
    """
    cmd = [sys.executable, "-m", module]
    print(f"\n[rebuild] → {' '.join(cmd)}", flush=True)
    t0 = time.time()
    result = subprocess.run(cmd, env=env, check=False)
    elapsed = time.time() - t0
    if result.returncode != 0:
        print(f"[rebuild] ✗ {module} failed after {elapsed:.0f}s", file=sys.stderr)
        raise SystemExit(result.returncode)
    print(f"[rebuild] ✓ {module} done in {elapsed:.0f}s", flush=True)


def main() -> None:
    """Run the feature-rebuild pipeline for the requested model run."""
    args = parse_args()
    model_run = args.model_run or get_settings().embed_model_run
    builders = tuple(args.only) if args.only else BUILDERS

    env = {**os.environ, "MYBOOKREC_EMBED_MODEL_RUN": model_run}
    print(f"[rebuild] model_run = {model_run}")
    print(f"[rebuild] builders  = {list(builders)}")

    t_total = time.time()
    for module in builders:
        run_builder(module, env)
    print(f"\n[rebuild] all done in {time.time() - t_total:.0f}s")


if __name__ == "__main__":
    main()
