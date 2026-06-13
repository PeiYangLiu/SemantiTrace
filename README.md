# SemantiTrace

Reference implementation of **SemantiTrace**, an empirical semantic-canary
auditing system for multimodal retrieval-augmented generation (RAG) databases.

SemantiTrace studies a **visual-context-preserving** RAG setting: a data owner
injects sparse, human-plausible visual canaries into benign images, the suspect
service indexes visual evidence, and the owner later audits whether canary
signatures can be retrieved and transcribed through the service pipeline.

This repository is intentionally **code-only**. Paper LaTeX sources, compiled
PDFs, review notes, downloaded literature, generated figures, large datasets,
model snapshots, AMLT result directories, and experiment artifacts are local-only
and ignored by Git.

## Current protocol

The current codebase implements the clean-only, rank-gated audit protocol used by
the latest SemantiTrace experiments.

- **Mode A: text mutation** mutates existing scene text with minimal
  single-glyph or short-token edits.
- **Mode B: natural-object insertion** inserts a contextually plausible object
  or object-borne label into a structurally compatible scene canvas.
- **Signature-blind Mode-B probes** avoid leaking the inserted color, position,
  font, or full target description in the query. Scene-aware/object-hook query
  materialization lives in `semantitrace/modeb_queries.py` and
  `scripts/materialize_modeb_*query_records.py`.
- **Clean-only baselines** index only distractor images for clean controls; they
  do not include the unwatermarked anchor image that corresponds to the canary.
- **Rank-gated scoring** credits a response only when the paired target image is
  in the retrieved top-k context before signature extraction is counted.

## Latest benchmark snapshot

The result artifacts are not committed, but the scripts/configs in this
repository correspond to the following latest clean-only reruns.

### End-to-end audit over a 1M unique visual index

| Profile | Scale | R@3 | Audit signal | Clean baseline | p-value |
| --- | ---: | ---: | ---: | ---: | ---: |
| Composite A+B overview | Mode A n=125 + Mode B clean-only n=500 | 42.1% | 35.3% | 0.4% | see split rows |
| Mode A subset | n=125 | 61.6% | 48.0% | 0.8% | 1.32e-40 |
| Mode B clean-only | n=500 | 22.6% | 22.6% protected-image hit | 0.0% | 1.29e-32 |
| Caption-only indexing | n=1000 | 5.0% | 0.03% | 0.0% | 0.16 |
| Caption + metadata sidecar | n=1000 | 65.3% | 39.9% | 0.0% | 4.30e-214 |

The composite A+B row is an operating-point overview, not a separately pooled
hypothesis test. Mode A reports rank-gated signature extraction CER; Mode B
reports clean-only target-in-top-3 protected-image hits, with response-only
string extraction retained as a diagnostic.

### Clean-only robustness reruns

| Boundary / transform | Scale | Combined CER | Mode A CER | Mode B CER | Clean CER |
| --- | ---: | ---: | ---: | ---: | ---: |
| JPEG Q75 | n=250 | 29.1% | 40.0% | 18.1% | 0.0% |
| Rescale 0.5x | n=250 | 27.5% | 39.5% | 15.5% | 0.0% |
| Gaussian noise sigma=5 | n=250 | 28.7% | 41.3% | 16.0% | 0.0% |
| Center crop 10% | n=250 | 28.5% | 37.9% | 19.2% | 0.0% |
| OCR blur | n=250 | 12.1% | 6.9% | 17.3% | 0.0% |
| OCR fill | n=250 | 9.1% | 0.0% | 18.1% | 0.0% |

Caption-only normalization is therefore treated as an architectural boundary for
pixel-level canaries, while sidecar-preserving RAG restores a strong provenance
channel.

## Repository layout

```text
semantitrace/        Core Python package and audit primitives.
scripts/            Data preparation, retrieval, scoring, merge, and AMLT helpers.
configs/            Local and AMLT YAML configs for large-scale experiments.
tests/              Lightweight unit tests for scoring, query filtering, and pipeline logic.
requirements.txt    Minimal runtime dependencies.
requirements-real.txt
                    Optional real-model dependencies for CLIP/VLM/diffusion runs.
```

Ignored local-only paths include `SemantiTrace_ICDE2027/`, `outputs/`, `pdfs/`,
`data*/`, `datasets/`, `amlt_*/`, `amlt_results/`, model caches, generated
archives, and submission bundles.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

For real-model experiments:

```bash
pip install -r requirements-real.txt
```

The real-model scripts expect externally managed datasets, model caches, and
service credentials. Do not commit those files; keep them under ignored local
paths or remote experiment storage.

## Quick checks

```bash
python -m compileall -q semantitrace scripts tests
python -m unittest discover -s tests
```

Focused tests for the clean-only/rank-gated changes:

```bash
python -m unittest tests.test_verification tests.test_modeb_queries
```

## Local clean-only protocol pipeline

The paper-scale runs use external image corpora, CLIP/Qwen checkpoints, and AMLT
for throughput. The public repository also includes a fully local deterministic
pipeline that exercises the same clean-only/rank-gated protocol without external
services, GPUs, model downloads, or datasets:

```bash
python scripts/run_local_cleanonly_pipeline.py \
  --output_dir outputs/local_cleanonly_pipeline \
  --num_mode_a 4 \
  --num_mode_b 4 \
  --num_distractors 16 \
  --top_k 3
```

It writes:

- `local_cleanonly_records.json`
- `local_cleanonly_details.jsonl`
- `local_cleanonly_summary.json`
- `local_cleanonly_summary.csv`

This local pipeline is a protocol smoke test, not a substitute for the
paper-scale 1M-index numbers. It verifies that the release can run end to end
with distractor-only clean indexes, Mode-A rank-gated extraction, Mode-B
protected-image hits, local robustness transforms, and caption-only/sidecar
boundary behavior.

## Deterministic main-experiment dry run

```bash
python scripts/run_main_experiment.py \
  --config configs/main_experiment.yaml \
  --dry_run_sample \
  --output_dir outputs/main_dryrun \
  --device cpu
```

## Real-model and AMLT runs

Large-scale runs are driven by AMLT configs in `configs/`. The most recent
clean-only configs are:

- `configs/amlt_semantitrace_modeb_llmquery_sceneaware_v18_cleanonly_msrresrchvc_a100_parallel.yaml`
- `configs/amlt_semantitrace_cleanonly_robustness_a100.yaml`
- `configs/amlt_semantitrace_cleanonly_caption_boundary_a100.yaml`

For AMLT jobs, keep outputs in remote result storage or ignored local folders.
The clean-only runs are designed to be resumable and avoid reading many small
files directly from blob storage: pack inputs as tar bundles, copy/extract them
locally inside the job, and write final outputs back to blob/result storage.

The public repo intentionally keeps only the current workflow entry points. Older
phase-style configs, retry-only configs, object-hook/scene-hook intermediate
Mode-B experiments, obsolete n-scaling launches, and generated artifact/data
snapshots are not part of the public code release.

## Public-repo hygiene

This public repository intentionally contains only source code, tests, and
configuration templates. Do not add paper sources, compiled manuscripts,
downloaded datasets, generated figures, model checkpoints, AMLT results, or
private review artifacts to Git.
