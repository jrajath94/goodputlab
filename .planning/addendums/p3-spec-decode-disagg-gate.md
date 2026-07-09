# Phase 6 Addendum — P3 Spec-Decode × Disagg Acceptance Gate

**Status:** locked design constraint (per `suggestions/feedback.md` addendum #1)
**Owner:** Phase 6 Spec-Decode planner must honor this gate before flipping EAGLE-3 on in the disagg decode pool.
**Refs:**
- `.planning/research/PITFALLS.md` P3 (spec-decode × disagg KV uninit)
- SGLang issue [#19796](https://github.com/sgl-project/sglang/issues/19796) — EAGLE-3 draft head + disagg decode pool can hit uninitialized draft KV (raw memory → NaN logits, crashes, acceptance degradation >10%)
- `.planning/REQUIREMENTS.md` SPEC-01..SPEC-06 (current at v1; SPEC-07..09 will be added by Phase 6 plan)

---

## Threat Model

The EAGLE-3 draft model runs on the **decode pool**. In a disagg topology, the
prefill pool's KV is transferred to the decode pool via NIXL. The draft head
attempts to predict the next token using its own internal KV. If the draft
head's KV cache is not properly seeded after a prefill step (e.g. NIXL
transfer leaves the draft head's buffer uninitialized), the draft model emits
random tokens, the acceptance rate collapses, and the verify step wastes
cycles — net ITL degradation versus the colocated baseline.

P3 ranks high because the failure is **silent** at the HTTP boundary (the
response still arrives, just with worse latency and quality) and only surfaces
in the acceptance-rate histogram.

---

## Acceptance Gate (binary, runs in Phase 6 acceptance harness)

Phase 6 plan must implement, no exceptions:

```
spec_disagg_acceptance_ratio = mean(disagg_pool_acceptance) /
                                mean(colocated_pool_acceptance)

gate(spec_disagg_acceptance_ratio >= 0.85)  → ABORT + AUTO-DISABLE
```

Trigger conditions (any one aborts):
- `spec_disagg_acceptance_ratio < 0.85` over N=200 shared-prefix requests
- Any single NaN logit observed in `/metrics` for the draft head
- Any crash in the draft head process (vLLM stderr)

Required measurements BEFORE the gate decision:
1. Run the same 200-request prefix-shared workload against:
   - (a) colocated topology with EAGLE-3 enabled
   - (b) disagg topology with EAGLE-3 enabled
2. Collect per-request acceptance ratio from vLLM metrics
   (`vllm:eagle_acceptance_ratio_sum` / `vllm:eagle_acceptance_ratio_count`).
3. Compute means; report 95% confidence interval (bootstrap, N=2000 resamples).
4. Compute ratio and bootstrap CI for the ratio.

---

## Auto-Disable and Recovery Flow

The Phase 6 control plane MUST implement:

1. **Detection:** continuous monitor scans `vllm:eagle_acceptance_ratio_sum` per
   pool, every 30 s.
2. **Auto-disable:** if rolling 5-min disagg acceptance ratio drops below
   `0.85 × colocated_baseline`, the router flips the disagg pool's `eagle_enabled`
   flag to `false` without dropping in-flight requests (decode stream
   continues speculatively-disabled until the next request lands).
3. **Logging:** `SPEC_AUTO_DISABLE reason=acceptance_below_threshold ratio=X.XX
   baseline=Y.YY pool=disagg_decode ts=...` is emitted to a
   `goodputlab_spec_events` Prometheus counter (instrumented; not a log-only
   signal).
4. **Re-enable:** only manual, by the human operator, after diffing
   disagg-acceptance-history against the colocated baseline. The auto-disable
   is sticky on purpose — re-arming the speculative decoder without a fix
   risks the same silent collapse.

---

## Phase 6 Plan Required Sections

When Phase 6 is executed (`/gsd:plan-phase 6`), the plan MUST include:

1. A task implementing `tests/test_spec_disagg_gate.py` that, given recorded
   metric samples (CSV/Parquet), computes the ratio + bootstrap CI and asserts
   `ratio >= 0.85`. Test fails closed (returns SystemExit) on under-ratio.
2. A task implementing the continuous monitor + `goodputlab_spec_events`
   counter (auto-disable).
3. A task wiring the auto-disable flag into the router control plane
   (Phase 3 router must expose `eagle_enabled: bool` per pool).
4. A `Day-1 flag-verification` checklist (see addendum
   `p7-day-1-flag-verification.md`) covering EAGLE-3 CLI args for vLLM 0.11.2.

## Anti-Pattern This Gate Prevents

A common v1 mistake: enabling EAGLE-3 only on colocated and reporting the
speedup as the project's win. The bench matrix MUST include EAGLE-on-EAGLE-off
and colocated-disagg pairs on every workload; missing cells = report rewrites.

---

*Locked 2026-07-09 by feedback.md addendum #1. Phase 6 plan must reference this file in `key_links`.*
