# SemantiTrace

Reference implementation for semantic canary injection and black-box audit
experiments in multimodal retrieval-augmented generation systems.

This repository is intentionally code-only. Paper LaTeX sources, compiled PDFs,
review notes, downloaded literature PDFs, generated figures, large datasets,
model snapshots, AMLT result directories, and other experiment artifacts are
kept locally and ignored by Git.

## What is implemented

The code implements a modular visual-canary auditing pipeline:

1. **Anchor mining**: find retrieval-isolated, editable image regions using
   encoder embeddings, OCR/semantic canvas heuristics, and mask utilities.
2. **Canary generation**: construct trigger prompts, trap signatures, and probe
   queries for text-mutation and natural-object insertion modes.
3. **Semantic injection**: route guidance context into deterministic, inpainting,
   or FLUX-style editors, including dual retrieval/readability guidance and
   optional latent mask blending.
4. **Retrieval and verification**: build visual retrieval indexes, run black-box
   probe queries, score mode-aware responses, and compute canary extraction
   statistics.

Additional modules include Mahalanobis OOD filtering, VLM anomaly prompts,
baseline overlays, robustness transforms, retrieval-capacity analysis, and
AMLT job configs for large-scale runs.

## Repository layout

```text
semantitrace/        Core Python package.
scripts/            Experiment, evaluation, data-preparation, and AMLT helper scripts.
configs/            Local and AMLT YAML configs.
tests/              Lightweight unit tests.
requirements*.txt   Base and real-model dependencies.
```

Ignored local-only paths include `SemantiTrace_ICDE2027/`, `outputs/`,
`pdfs/`, `data*/`, `amlt_*/`, model caches, and generated archives.

## Quick start

```bash
python -m compileall -q semantitrace scripts tests
python -m unittest discover -s tests
```

For the deterministic local pipeline:

```bash
python scripts/run_main_experiment.py \
  --config configs/main_experiment.yaml \
  --dry_run_sample \
  --output_dir outputs/main_dryrun \
  --device cpu
```

For real-model runs, install the optional dependencies and select a real config:

```bash
pip install -r requirements-real.txt
python scripts/run_main_experiment.py \
  --config configs/main_experiment_real.yaml \
  --output_dir outputs/main_real \
  --device cuda
```

Large-scale profiles are submitted through AMLT configs in `configs/`; generated
outputs should remain under ignored local or remote storage paths rather than
being committed.
