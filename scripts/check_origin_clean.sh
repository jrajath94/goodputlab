#!/usr/bin/env bash
# GoodputLab — invariant: origin/main must never contain candidacy / playbook /
# strategy docs. Per workspace CLAUDE.md + suggestions/PORTFOLIO.md finding #4
# the Anthropic_*, MASTER_EXECUTION_PROMPT_CLAUDE.md, Staff_Level_Projects_Spec_,
# Implementation_Brief / *_Implementation_Brief / EXECUTION_PROMPT / AgentSLA_
# Implementation_Brief files must NEVER be committed or pushed to the public repo.

set -euo pipefail

# Patterns that should NEVER appear on origin/main. CI fails if any do.
LEAK_PATTERNS=(
  "Anthropic_Candidacy_Playbook"
  "MASTER_EXECUTION_PROMPT_CLAUDE"
  "Staff_Level_Projects_Spec_"
  "_Implementation_Brief"
  "EXECUTION_PROMPT"
  "candidacy_playbook"
)

# Allow-list of files in suggestions/ that always live in origin (they are
# public project memory). No other top-level docs matching the leak patterns
# may appear in origin/main.
LEAK_OUTPUT=$(git ls-tree -r --name-only origin/main 2>/dev/null \
  | grep -E "$(IFS='|'; echo "${LEAK_PATTERNS[*]}")" || true)

if [ -n "$LEAK_OUTPUT" ]; then
  echo "::error::origin/main contains leaked strategy/playbook files:" >&2
  echo "$LEAK_OUTPUT" >&2
  echo "::error::history rewrite or repo recreation is required" >&2
  exit 1
fi

echo "OK: origin/main contains no leaked strategy docs"
exit 0
