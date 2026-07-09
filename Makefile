# GoodputLab project interface.
# Phase 1 plan 01-01: skeleton command contract.

# Use bash for consistent shell semantics inside recipes.
SHELL := /usr/bin/env bash

# All targets are recipes (no file prerequisites at the top level).
.PHONY: install-dev lint test compose-config provision up-colocated up-chunked up-disagg up-disagg-tier down health sentinel

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

# Stub for plan 01-02. Real implementation lives in the next plan.
provision:
	@echo "see 01-02 (provision.sh ships with the next plan)"

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
