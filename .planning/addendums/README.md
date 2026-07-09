# Planning Addendums

Locked design constraints that future phase planners MUST honor before
implementing. Each addendum maps to a numbered pitfall in
`.planning/research/PITFALLS.md`.

| File | P# | Phase | Status |
|---|---|---|---|
| `p3-spec-decode-disagg-gate.md` | P3 | 6 SPEC | locked acceptance gate |
| `p10-metrics-reconciliation.md` | P10 | 2 LOAD | design sketch |
| `p12-failure-drills.md` | P5/P1/P12 | 8 BENCH | design sketch |
| `p7-day-1-flag-verification.md` | meta | any | standing checklist |

How to consume:
1. Phase planner reads `addendums/README.md` (this file) up-front.
2. For each entry whose `Phase` matches the current plan, the planner
   copies the relevant acceptance criteria / measurement requirements
   into a new task in `0X-NN-PLAN.md` and references the addendum file in
   `key_links`.
3. Deviations require a written decision note appended to the addendum
   (the file is append-only; the planner cannot silently override).

Drafted 2026-07-09. Origin: `suggestions/feedback.md` addendum section.
