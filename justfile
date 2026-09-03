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

# Run the test suite (excludes benchmarks and infra-dependent tests).
test:
    uv run pytest -q -m "not slow and not needs_infra and not needs_model"

# Everything CI runs, in CI order.
check: lint test

# Auto-fix what can be auto-fixed.
fix:
    uv run ruff check --fix .
    uv run ruff format .

# --- benchmarks ------------------------------------------------------------

# Milestone 0: measure the L2 candidate latency frontier on this machine.
# Regenerates docs/benchmarks/l2-frontier.{md,json,png}. Takes several minutes.
bench:
    uv run --group bench python -m portcullis.eval.bench_l2_frontier

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
