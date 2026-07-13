# GoodputLab project interface.
# Phase 1 plan 01-01: skeleton command contract.

# Use bash for consistent shell semantics inside recipes.
SHELL := /usr/bin/env bash

# All targets are recipes (no file prerequisites at the top level).
.PHONY: install-dev lint test compose-config provision up-colocated up-chunked up-disagg up-disagg-tier down health sentinel ollama-smoke figures

install-dev:
	pip install -e ".[dev]"

lint:
	python -m ruff check .
	python -m mypy .

test:
	python -m pytest

compose-config:
	docker compose --profile colocated config >/dev/null
	docker compose --profile chunked config >/dev/null
	docker compose --profile disagg config >/dev/null
	docker compose --profile disagg-tier config >/dev/null

# Phase 1 plan 01-02: idempotent RunPod boot with 1200s budget gate (TOPO-06).
# Execution requires a live GPU pod; not run in CI.
provision:
	bash provision.sh

# Topology bring-up (plan 01-03 ships the docker-compose.yml).
up-colocated:
	docker compose --profile colocated up -d

up-chunked:
	docker compose --profile chunked up -d

up-disagg:
	docker compose --profile disagg up -d

up-disagg-tier:
	docker compose --profile disagg-tier up -d

down:
	docker compose --profile colocated --profile chunked --profile disagg --profile disagg-tier down

# Stub shell script ships with plan 01-06 (sentinel + health probes).
health:
	bash scripts/health.sh all

# Sentinel CLI ships with plan 01-05.
sentinel:
	python3 tests/sentinel.py --mode check --base-url http://localhost:19100/v1

# P5-3 — local M1 Max smoke harness. Requires ollama on PATH and
# `ollama serve` running on :11434. Pull qwen3:8b manually first:
#   ollama pull qwen3:8b
# The bench CLI fails with a clear message if the model is missing.
ollama-smoke:
	@command -v ollama >/dev/null 2>&1 || { \
		echo "ollama not on PATH. Install: https://ollama.com/download"; exit 1; }
	@curl -fsS http://localhost:11434/api/tags >/dev/null 2>&1 || { \
		echo "ollama not reachable on :11434 — start with: ollama serve"; exit 1; }
	python3 -m bench.ollama_smoke --model qwen3:8b --n 8

# P5-4 — generate figures + cost table from bench/results/real/*.json
figures:
	python3 -m bench.figures
