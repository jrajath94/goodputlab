# Day-1 Flag-Verification Ritual (Standing Checklist)

**Status:** standing checklist (per `suggestions/feedback.md` addendum #2)
**Owner:** every phase planner that touches vLLM / NIXL / LMCache flags.

---

## Why

NIXL semantics change between vLLM minors. We have already burned budget
debugging a 0.5 → 0.6 flag rename (per PROJECT.md research summary, the
NixlConnector `kv_role` and `--kv-transfer-config` flag set rotated).
Verifying against live docs BEFORE writing config prevents silent garbage
and shortens the cold-to-serving budget.

## Standing Checklist (each plan that adds vLLM / NIXL / LMCache flags)

1. Identify the new flags to introduce this phase (e.g.
   `--enable-chunked-prefill --max-num-batched-tokens 2048`).
2. Fetch the live docs for the pinned version. The current pin is
   `vllm/vllm-openai:v0.11.2`. Use `mcp__context7` against
   `websites/vllm_ai_en_stable` first; cross-check the GitHub `v0.11.2`
   tag for the exact flag name and any deprecation warnings.
3. Record in the plan under `verification_refs`:
   - doc URL (full path, line / anchor if available)
   - access date (ISO 8601)
   - exact CLI flag string the plan uses
4. If the flag is renamed/deprecated versus the older doc, **file a deviation
   note** at the top of the plan and update `.planning/research/SUMMARY.md`.
5. If a flag is NEW (not in our prior docs), confirm it accepts the
   syntax the plan expects by running `docker run --rm vllm/vllm-openai:v0.11.2 python -c "import vllm; help(vllm.engine.arg_utils.EngineArgs)"` and grepping the flag name.

## Plan Template Field

Every 01-NN / 02-NN / ... plan that adds flags MUST include this frontmatter block:

```yaml
flag_verification:
  - flag: --enable-chunked-prefill
    doc_url: https://docs.vllm.ai/en/v0.11.2/serving/engine_args.html
    accessed: 2026-07-09
    confirmed: true
  - flag: --kv-transfer-config
    doc_url: https://docs.vllm.ai/en/v0.11.2/features/disagg.html
    accessed: 2026-07-09
    confirmed: true
```

If a flag is omitted from the verification log, treat the plan as
incomplete — the verifier (`gsd-verifier`) should reject.

## Doc-URL Bookmark Sheet

| Service | URL |
|---|---|
| vLLM 0.11.2 stable docs | `https://docs.vllm.ai/en/v0.11.2/` |
| LMCache 0.3.x docs | `https://docs.lmcache.ai/en/v0.3.x/` |
| NIXL UCX backend | `https://github.com/ai-dynamo/nixl` |
| EAGLE-3 (Spec-Decode) | `https://github.com/SafeAILab/EAGLE` (upstream) |

Refreshing the bookmarks each quarter (or after any incident) keeps the
ritual fast.

---

*Drafted 2026-07-09. Planners should copy the YAML template verbatim into the plan's frontmatter.*
