# PORTCULLIS task runner.
# Recipes are plain `uv run` invocations so they behave identically under
# PowerShell, cmd and sh.

set windows-shell := ["powershell.exe", "-NoLogo", "-NoProfile", "-Command"]

_default:
    @just --list

# --- development -----------------------------------------------------------

# Install the workspace and dev tooling.
install:
    uv sync

# Lint, format-check and type-check. Same gates as CI.
lint:
    uv run ruff check .
    uv run ruff format --check .
    uv run mypy packages tests

# Functional tests. Excludes latency guards, which must not be measured
# while the property suite is saturating the CPU.
test:
    uv run pytest -q -m "not slow and not needs_infra and not needs_model"

# Latency guards. Each slow test file runs as its own process, not just
# serially within one: two tight CPU-bound loops back-to-back in the same
# interpreter leave enough residual thermal/scheduler state on this hybrid
# P/E-core machine (ADR-0002) to push the second guard's p95 over budget
# even though each measures clean in isolation - found when the Tier A
# guard flaked at Milestone 5 despite no L0/L1 code changing that milestone.
test-perf:
    uv run pytest -q -p no:randomly -m "slow and not needs_infra and not needs_model" packages/core/tests/test_l0_properties.py
    uv run pytest -q -p no:randomly -m "slow and not needs_infra and not needs_model" packages/core/tests/test_l1_rule_corpus.py

# Everything CI runs, in CI order.
check: lint test test-perf

# Auto-fix what can be auto-fixed.
fix:
    uv run ruff check --fix .
    uv run ruff format .

# --- benchmarks ------------------------------------------------------------

# Milestone 0: measure the L2 candidate latency frontier on this machine.
# Regenerates docs/benchmarks/l2-frontier.{md,json,png}. Takes several minutes.
bench:
    uv run --group bench python -m portcullis.eval.bench_l2_frontier

# --- data --------------------------------------------------------------

# Rebuild the training corpus from its sources: ingest, MinHash dedup,
# family-disjoint split, leakage audit, manifest. Tier 1 (ungated) only.
data:
    uv run --package portcullis-training python -m portcullis.training.data.pipeline

# --- L2 (Milestone 4) -------------------------------------------------------

L2_CHECKPOINT_DIR := env_var('LOCALAPPDATA') / "portcullis" / "l2_checkpoint"
L2_MODEL_DIR := env_var('LOCALAPPDATA') / "portcullis" / "l2_export"

# Weak-label train/validation against the L1 rule taxonomy (ADR-0005).
weak-label:
    uv run --package portcullis-training python -m portcullis.training.l2.weak_label_corpus

# Fine-tune MiniLM-L6 on the masked multi-label objective. CPU, no GPU here.
# Checkpoint written outside the repo entirely (see train.py docstring).
# epochs=1, seq_len=128, batch=32: measured on this machine at Milestone 4 -
# the seq=256 default takes ~83min/epoch here (3.2 rows/s); this configuration
# measured ~33min/epoch (8.0 rows/s). One epoch matches the "prove the
# pipeline" scope agreed for M4, not a claim of a well-trained model.
train-l2:
    uv run --package portcullis-training python -m portcullis.training.l2.train --out-dir "{{L2_CHECKPOINT_DIR}}" --epochs 1 --batch-size 32 --max-seq-len 128

# Fit Platt scaling and produce the reliability diagram (NFR-2).
calibrate-l2:
    uv run --package portcullis-training python -m portcullis.training.l2.calibrate --checkpoint-dir "{{L2_CHECKPOINT_DIR}}"

# ONNX export, INT8 quantise, accuracy-delta check, latency histogram.
export-l2:
    uv run --package portcullis-training python -m portcullis.training.l2.export --checkpoint-dir "{{L2_CHECKPOINT_DIR}}" --model-out-dir "{{L2_MODEL_DIR}}"

# The full M4 pipeline, in order.
l2: weak-label train-l2 calibrate-l2 export-l2

# --- M5: kNN sidecar + fusion + policy --------------------------------------

KNN_INDEX_DIR := env_var('LOCALAPPDATA') / "portcullis" / "knn_index"

# Embed the train-partition attack rows and save the kNN sidecar's index.
# Brute-force numpy, not FAISS - ADR-0003's own measurement at this corpus
# scale (6.8k rows) found sub-millisecond query latency with no ANN library.
build-knn-index:
    uv run --package portcullis-training python -m portcullis.training.knn.build_index --out-dir "{{KNN_INDEX_DIR}}"

# Fit fusion (logistic regression, validation) and evaluate fusion-vs-max()
# on a held-out test sample - the M5 checkpoint artifact.
fit-fusion:
    uv run --package portcullis-training python -m portcullis.training.fusion.fit --knn-index-dir "{{KNN_INDEX_DIR}}" --checkpoint-dir "{{L2_CHECKPOINT_DIR}}" --onnx-path "{{L2_MODEL_DIR}}/l2.int8.onnx" --out-dir "{{L2_CHECKPOINT_DIR}}"

# The full M5 pipeline, in order. Assumes `just l2` has already run.
m5: build-knn-index fit-fusion

# --- evaluation ------------------------------------------------------------

# Regenerate every number in the README. Not yet implemented (Milestone 9).
eval:
    @echo "not implemented until Milestone 9 - README metrics remain TBD"
    @exit 1

# --- runtime ---------------------------------------------------------------

# Bring up Redis, Postgres, gateway and dashboard.
up:
    docker compose -f ops/compose.yaml up -d

down:
    docker compose -f ops/compose.yaml down

# Validate the compose file without starting anything.
compose-check:
    docker compose -f ops/compose.yaml config --quiet
