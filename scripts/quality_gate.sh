#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODE="${1:-}"

cd "$ROOT_DIR"

run_step() {
  local label="$1"
  shift
  printf '==> %s\n' "$label" >&2
  "$@"
}

run_commit_gate() {
  run_step "uv lock --check" uv lock --check
  run_step "ruff check --fix" uv run ruff check --fix .
  run_step "ruff format" uv run ruff format .
  run_step "core tests" uv run pytest -q -m core tests
  run_step "integration tests" uv run pytest -q -m integration tests
  run_step "mypy" uv run mypy src tests
}

run_push_gate() {
  run_step "uv lock --check" uv lock --check
  run_step "ruff check" uv run ruff check .
  run_step "ruff format --check" uv run ruff format --check .
  run_step "core tests" uv run pytest -q -m core tests
  run_step "integration tests" uv run pytest -q -m integration tests
  run_step "mypy" uv run mypy src tests
}

case "$MODE" in
  commit)
    run_commit_gate
    ;;
  push)
    run_push_gate
    ;;
  *)
    printf 'Usage: %s {commit|push}\n' "${BASH_SOURCE[0]}" >&2
    exit 2
    ;;
esac
