"""IO subpackage for MyBookRec runtime + eval artifacts.

Three focused modules:

- `artifacts` — typed lazy-loaders (`TransformedArtifacts`) for everything under
  data/transformed/. Used by serve, ingest, recommend.
- `checkpoints` — feature-set registry, checkpoint loading, item-feature loading,
  device selection, batched encoder. Used by serve, recommend, training, index build.
- `eval_data` — held-out test-pair sampling, train-rated exclude dict,
  interactions-split loader. Eval/training only.

This `__init__` intentionally does not re-export — callers import the submodule
they need so the dependency direction is visible in the source.
"""
