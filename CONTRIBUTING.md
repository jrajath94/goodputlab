# Contributing to GoodputLab

## Commit Standards

All commits must follow [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>(<scope>): <subject>
```

**Types:**
- `feat`: New feature or topology
- `fix`: Bug fix
- `perf`: Performance improvement
- `test`: Test additions or fixes
- `docs`: Documentation or phase notes
- `refactor`: Code refactoring without behavioral change
- `bench`: Benchmark changes
- `ci`: CI/CD pipeline changes
- `style`: Formatting/whitespace-only changes (ruff format, isort
  reorder, blank-line cleanup). No behavior change.
- `chore`: Routine tasks (cleanup, deps)

**Subject:**
- Imperative mood ("add" not "added")
- No period at end
- ≤72 characters
- Lowercase

## Code Quality

All code must pass:

```bash
# Formatting
ruff format .

# Linting
ruff check .

# Type checking (strict mypy)
mypy core control scripts

# Testing
pytest tests/ -v --cov=core --cov=control
```

## Pull Request Process

1. **Branch:** Create from `main` with name `phase-N/description` (e.g., `phase-1/topologies-deployment`)
2. **Commits:** Atomic, conventional format
3. **Tests:** All new code must have tests; runtime behavior verified
4. **Coverage:** Do not decrease coverage
5. **CI:** All workflows must pass before merge
6. **Description:** Clear problem, approach, evidence, tradeoffs

## Code Style

- Python 3.11+
- Type hints on all public APIs (strict mypy)
- Docstrings on public classes/functions (one-line preferred)
- FastAPI for HTTP routing
- Pydantic v2 for schemas
- No dead code or commented-out blocks

## Testing

Tests must cover:
- Router logic (cache affinity, load balancing)
- Admission control (SLO thresholds)
- Autoscaler behavior (PID dynamics)
- Metrics reconciliation (vs vLLM ground truth)
- End-to-end topology verification

Use pytest fixtures for common setup. Docker-based integration tests marked `@pytest.mark.integration`.

## Local Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/

# Lint and type check
ruff check .
mypy core control scripts
```

## Deployment

One-command deployment:
```bash
docker-compose -f deployments/full-stack.yml up -d
./scripts/health_check.sh all
```

All topologies (colocated, chunked, disagg, disagg+tiering) must serve requests end-to-end.

## Benchmarking

Benchmark runs must:
- Use `make bench` for reproducibility
- Include 3+ seeds for variance estimation
- Log hardware (GPU model, vRAM, driver version)
- Save results to `results/` with timestamp

## Phases

Follow 10-week phase plan:
- W1-2: Topologies deployment
- W3-4: Load gen + metrics
- W5-6: Router + admission control
- W7-8: KV tiering + spec decode
- W9: Autoscaler
- W10: Benchmark + report

## Release

Releases follow [semantic versioning](https://semver.org/). Tag on `main` after phase 10 completion.

## Questions?

Open an issue or discussion on GitHub.
