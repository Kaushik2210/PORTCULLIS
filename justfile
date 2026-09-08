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

# Tests that need the real trained M4/M5 artifacts on disk (`just l2 m5`
# first). Not part of `just check` - a fresh clone has no checkpoint yet,
# and CI does not train one.
test-model:
    uv run pytest -q -m needs_model

# Tests that need real infra (Redis via `just up`). Not part of `just check`
# for the same reason test-model isn't - a fresh clone/CI has none running.
test-infra:
    uv run pytest -q -m needs_infra

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

# --- M6: gateway --------------------------------------------------------

# The fake LLM backend the gateway proxies to by default (ADR-0007) - no
# API key, no cost. Run this in one terminal, `just gateway-serve` in
# another.
gateway-mock-upstream:
    uv run --package portcullis-gateway uvicorn portcullis.gateway.mock_upstream:app --port 8000

# The real gateway, backed by the real M4/M5 checkpoint. Needs `just l2 m5`
# to have already produced that checkpoint, and (by default) the mock
# upstream running on port 8000 - override PORTCULLIS_UPSTREAM_BASE_URL to
# point at a real provider instead.
gateway-serve:
    uv run --package portcullis-gateway uvicorn portcullis.gateway.app:app --port 8001

# Milestone 6's checkpoint artifact: starts both servers itself, fires a
# benign and an injection request at each, and prints the difference.
demo-m6:
    uv run --package portcullis-gateway python -m portcullis.gateway.demo

# --- M7: conversation state machine (L4) ------------------------------------

# Milestone 7's checkpoint artifact: a real multi-turn conversation against
# the trained checkpoint (cumulative-risk escalation) plus the crescendo
# algorithm itself on a synthetic sequence. Needs `just l2 m5` first.
demo-m7:
    uv run --package portcullis-gateway python -m portcullis.gateway.demo_m7

# --- M8: egress inspection (L5) ----------------------------------------------

# Milestone 8's checkpoint artifact: a live system-prompt-leak catch. The
# mock upstream genuinely leaks its system prompt on the classic extraction
# phrase; L5 catches the canary in PORTCULLIS's own response scan. Also
# shows secret redaction (not a hard block) on a fake-AWS-key message.
# Needs `just l2 m5` first.
demo-m8:
    uv run --package portcullis-gateway python -m portcullis.gateway.demo_m8

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
