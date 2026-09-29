"""Block-manager KV-cache measurement for the scaling study.

Method (TRD sections 3-4): vLLM pre-allocates the entire KV pool at
engine init, so ``torch.cuda.memory_allocated()`` does not move with
sequence length. The study therefore measures through vLLM's block
accounting instead:

- Drive the v1 engine manually (``add_request`` / ``step``), never
  ``llm.generate()`` (it frees blocks on return).
- After prefill, read free GPU blocks from the scheduler's block
  manager; step ``D`` decode steps; read again while every request
  still holds its blocks.
- ``kv_bytes_used = used_blocks * block_size * theory_bytes_per_token``.
- The fixed non-KV overhead ``F`` is measured once per campaign right
  after engine load:
  ``F = memory_allocated() - num_gpu_blocks * block_size * theory_bpt``.

The sampling loop is written against the ``EngineDriver`` protocol, so
the CPU-only ``FakeEngineDriver`` (rung 0 of the runbook) exercises the
exact same code path as the GPU driver: add_request, step, block reads,
consistency gates, abort, empty. If the pipeline cannot recover a known
bytes/token from the fake, the pod run is not launched.

Limits: block quantization means a sequence of T tokens holds
``ceil(T / block_size)`` blocks; the per-cell ``measured_bytes_per_token``
column carries that rounding and the regression slope over cells is the
headline number. The fake driver is a model, not a measurement, and
must never be reported as measured data.
"""

from __future__ import annotations

import math
import os
import subprocess
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from bench.kv_scaling.schema import (
    KneePoint,
    KvScalingCampaign,
    KvScalingCell,
    RegressionFit,
)
from bench.kv_scaling.theory import HfKvTheory, config_from_hf_config

# Sweep model, env-overridable. Final campaign decision (TRD section
# 2a): Qwen2.5-0.5B-Instruct at the pinned HF revision below, verified
# against the live HF config 2026-09-28 (24 layers, 2 KV heads, 14 query
# heads, hidden_size 896, head_dim 64, theory 12,288 bytes/token).
# Qwen3-0.6B was evaluated and rejected: its top grid cell needs ~30 GiB
# of KV cache, which does not fit a 24 GB RTX 4090. Re-verify the pin
# against huggingface.co/api/models before launch; if the sha moved,
# record the new value, never silently float.
MODEL_ID = os.environ.get("KV_SCALING_MODEL_ID", "Qwen/Qwen2.5-0.5B-Instruct")
REVISION = os.environ.get("KV_SCALING_REVISION", "7ae557604adf67be50417f59c2c2f167def9a775")

# Decode steps per sample. Small on purpose: enough to leave prefill and
# reach steady decode, few enough to keep the grid fast.
DECODE_TOKENS = 4

# vLLM PagedAttention block size. A fixed input to the error budget, not
# a tuned variable (PRD non-goals). Passed to the LLM constructor and
# verified against the live engine before measuring.
DEFAULT_BLOCK_SIZE = 16

# Rung-0 gate: the fake-block-manager pipeline must recover the
# injected bytes/token with R^2 at least this high.
DRY_RUN_MIN_R2 = 0.98


class TokenizerLike(Protocol):
    """Minimal surface needed for exact-length prompt construction."""

    def encode(self, text: str) -> list[int]: ...
    def decode(self, ids: list[int]) -> str: ...


def cuda_available() -> bool:
    """True when an NVIDIA GPU is visible via nvidia-smi.

    Fork-safe by construction: this deliberately never touches the torch
    CUDA runtime. ``torch.cuda.is_available()`` calls ``cudaGetDeviceCount``,
    which runs ``cuInit`` in the calling process, and a ``cuInit``-poisoned
    parent makes vLLM v1's forked EngineCore die with "Cannot re-initialize
    CUDA in forked subprocess" (campaign kv-scaling-20260928: smoke failed,
    ~$1.18 of idle pod burn). ``nvidia-smi`` runs in a separate process and
    cannot poison the fork.
    """
    try:
        proc = subprocess.run(
            ["nvidia-smi", "-L"],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return proc.returncode == 0 and "GPU" in proc.stdout


# ---------- exact-length prompts ----------

_TILE_SLACK = 64


def build_exact_length_prompts(tokenizer: TokenizerLike, seq_len: int) -> str:
    """Return one prompt string that encodes to exactly ``seq_len`` tokens.

    Tokenizes a base text, tiles the ids past ``seq_len``, decodes, and
    re-encodes to the tiled text's TRUE tokenization before truncating.
    The re-encode step is the fix: naive id tiling merges tokens at tile
    boundaries when the text is re-encoded (Qwen2.5-0.5B: a tile's
    trailing " " and the next tile's leading "The" re-encode as the single
    token " The", losing one token per boundary; this failed the smoke
    gate of campaign kv-scaling-20260928e with 1023 vs 1024). Truncating
    the true encoding at a token boundary is stable, and the final
    re-encode verifies the length instead of silently measuring the wrong
    prompt size.
    """
    if seq_len <= 0:
        raise ValueError(f"seq_len must be > 0, got {seq_len}")
    base = "The quick brown fox jumps over the lazy dog. " * 64
    ids = tokenizer.encode(base)
    while len(ids) < seq_len + _TILE_SLACK:
        ids = ids + ids
    true_ids = tokenizer.encode(tokenizer.decode(ids))
    if len(true_ids) < seq_len:
        raise ValueError(
            f"tiled text re-encoded to {len(true_ids)} tokens, need {seq_len}; "
            "tiling loses too many tokens at tile boundaries"
        )
    prompt = tokenizer.decode(true_ids[:seq_len])
    actual = len(tokenizer.encode(prompt))
    if actual != seq_len:
        raise ValueError(
            f"prompt round trip gave {actual} tokens, expected {seq_len}; "
            "this tokenizer cannot build exact-length prompts"
        )
    return prompt


# ---------- engine driver protocol ----------


@dataclass(frozen=True)
class StepOutput:
    """What one engine step produced for one request."""

    request_id: str
    new_tokens: int
    finished: bool


class BlockPoolLike(Protocol):
    """Uniform surface over the scheduler's GPU block pool."""

    @property
    def num_gpu_blocks(self) -> int: ...
    def get_num_free_blocks(self) -> int: ...


class EngineDriver(Protocol):
    """Manual-step engine surface shared by the vLLM and fake drivers.

    ``prompt_len`` on ``add_request`` is for the fake driver only (no
    tokenizer there); the vLLM driver ignores it and tokenizes the
    prompt string. ``ignore_eos`` asks the engine to decode past EOS
    up to ``max_tokens`` instead of stopping at the first EOS token;
    the sampler sets it so the decode length is exactly fixed.
    """

    @property
    def block_pool(self) -> BlockPoolLike: ...
    def num_free_blocks(self) -> int:
        """Free GPU blocks now (device-synchronized on the real driver)."""
        ...

    def add_request(
        self,
        request_id: str,
        prompt: str,
        max_tokens: int,
        temperature: float,
        prompt_len: int,
        ignore_eos: bool = False,
    ) -> None: ...
    def step(self) -> list[StepOutput]: ...
    def get_num_unfinished_requests(self) -> int: ...
    def abort_request(self, request_id: str) -> None: ...
    def end_sample(self) -> None:
        """Abort leftovers and release the sample's blocks."""
        ...


@dataclass
class SampleReadings:
    """Both block-accounting readings of one sample.

    ``kv_bytes_decode`` is the cell's sample value: KV bytes live while
    every request still held its blocks, after the decode steps.
    ``kv_bytes_prefill`` is the post-prefill reading; the decode reading
    must never be smaller. Token counts are the actual live counts the
    consistency gate checked, not formulas.
    """

    kv_bytes_prefill: int
    kv_bytes_decode: int
    live_tokens_prefill: int
    live_tokens_decode: int
    used_blocks_prefill: int
    used_blocks_decode: int


def _check_block_consistency(
    *,
    used_blocks: int,
    live_tokens: int,
    batch_size: int,
    block_size: int,
    fully_scheduled: bool,
    where: str,
) -> None:
    """Loud gate on the block-manager accounting.

    ``used_blocks * block_size`` must lie within
    ``[live_tokens, live_tokens + batch_size * block_size + 64]`` once
    every request is scheduled (has generated at least one token). Before
    that, only the upper bound applies: unscheduled prompt tokens are not
    in cache yet, so the lower bound cannot hold. A breach means the
    block reader is wrong: stop, do not sweep.
    """
    if used_blocks < 0:
        raise ValueError(f"{where}: negative used blocks: {used_blocks}")
    used_slot_bytes = used_blocks * block_size
    hi = live_tokens + batch_size * block_size + 64
    if used_slot_bytes > hi:
        raise ValueError(
            f"{where}: block accounting reports {used_slot_bytes} token-slots "
            f"but only {live_tokens} tokens are live (limit {hi}); "
            "the block reader is wrong, refusing to sweep"
        )
    if fully_scheduled and used_slot_bytes < live_tokens:
        raise ValueError(
            f"{where}: block accounting reports {used_slot_bytes} token-slots "
            f"below the {live_tokens} live tokens; "
            "the block reader is wrong, refusing to sweep"
        )


def sample_decode_window(
    *,
    driver: EngineDriver,
    request_ids: list[str],
    prompts: list[str],
    prompt_lens: list[int],
    decode_tokens: int = DECODE_TOKENS,
    ignore_eos: bool = True,
    prefill_chunks: int = 1,
    theory_bpt: int,
    block_size: int,
    log: Callable[[str], None] | None = None,
) -> SampleReadings:
    """Run one sample through the real sampling path and read the blocks.

    Adds one request per prompt (``max_tokens=decode_tokens``,
    ``temperature=0``, ``ignore_eos=True``), steps until every request
    is past prefill, reads the prefill block count, then freezes the
    decode block count at the last post-step where every request is
    still alive (no finishes yet): chunked prefill staggers a batch
    into waves that finish on different steps, and finished waves free
    their blocks, so a reading taken on the all-finished step would
    undercount the batch. ``prefill_chunks`` is the caller's chunk
    estimate (ceil(batch*seq_len / max_num_batched_tokens)); it only
    sizes the step budget. Both readings pass the consistency gate;
    the decode reading must be >= the prefill reading. Always ends
    the sample (abort + release) even when a gate fires.

    ``ignore_eos`` is True in production: the study measures KV bytes
    per live token, and EOS timing is irrelevant to that quantity, so
    the sampler disables EOS and the decode length is exactly
    ``decode_tokens`` for every request. With ``ignore_eos=False``
    (tests only), a request that emits EOS frees its blocks mid-sample
    and the loud guards below refuse to sweep instead of measuring a
    partial batch.
    """
    batch = len(request_ids)
    if batch == 0:
        raise ValueError("need at least one request")
    if not (len(prompts) == batch and len(prompt_lens) == batch):
        raise ValueError("request_ids, prompts, and prompt_lens must match")
    if any(pl <= 0 for pl in prompt_lens):
        raise ValueError(f"prompt_lens must be > 0, got {prompt_lens}")
    if decode_tokens < 1:
        raise ValueError(f"decode_tokens must be >= 1, got {decode_tokens}")
    if theory_bpt <= 0:
        raise ValueError(f"theory_bpt must be > 0, got {theory_bpt}")
    if block_size <= 0:
        raise ValueError(f"block_size must be > 0, got {block_size}")

    pool = driver.block_pool
    total_blocks = pool.num_gpu_blocks
    if not isinstance(total_blocks, int) or total_blocks <= 0:
        raise ValueError(f"block pool reports invalid total: {total_blocks!r}")

    for rid, prompt, plen in zip(request_ids, prompts, prompt_lens, strict=True):
        driver.add_request(
            rid, prompt, max_tokens=decode_tokens, temperature=0.0, prompt_len=plen,
            ignore_eos=ignore_eos,
        )

    generated = {rid: 0 for rid in request_ids}
    finished = {rid: False for rid in request_ids}
    prefill_used: int | None = None
    prefill_live = 0
    decode_used: int | None = None
    decode_live = 0
    # Guard against an engine that never finishes: prefill steps plus the
    # decode steps plus slack. A healthy run finishes in
    # prefill_chunks + decode_tokens steps.
    max_steps = prefill_chunks + decode_tokens + 8

    def _emit(msg: str) -> None:
        if log is not None:
            log(msg)

    def _live_tokens() -> tuple[list[str], int]:
        """Unfinished request ids and their live token count.

        Finished requests hold no cache (their blocks are freed), so
        they contribute nothing to the live count the gate checks.
        """
        active = [rid for rid in request_ids if not finished[rid]]
        live = sum(
            prompt_lens[i] + generated[rid]
            for i, rid in enumerate(request_ids)
            if not finished[rid]
        )
        return active, live

    try:
        for step_idx in range(max_steps):
            active, live_before = _live_tokens()
            # Pre-step state: on the finishing step this is the decode
            # reading, with every request's blocks still held.
            free_before = driver.num_free_blocks()
            _check_block_consistency(
                used_blocks=total_blocks - free_before,
                live_tokens=live_before,
                batch_size=batch,
                block_size=block_size,
                fully_scheduled=all(generated[rid] >= 1 for rid in active),
                where=f"pre-step {step_idx}",
            )
            for out in driver.step():
                if out.request_id not in generated:
                    raise ValueError(f"engine returned unknown request id {out.request_id!r}")
                if out.new_tokens < 0:
                    raise ValueError(f"engine returned negative new_tokens for {out.request_id!r}")
                generated[out.request_id] += out.new_tokens
                if out.finished:
                    finished[out.request_id] = True
            if all(finished.values()):
                # The decode reading was frozen at the last post-step with
                # every request still alive (see below): on this finishing
                # step finished waves may already have freed their blocks.
                # If nothing was ever frozen (all finished on step 0), the
                # post-loop check below raises the clean error.
                if decode_used is not None:
                    _emit(
                        f"decode reading: {decode_used} blocks used, "
                        f"{decode_live} tokens live "
                        f"(unfinished now: {driver.get_num_unfinished_requests()})"
                    )
                # No post-step gate here: the finished-step block state is
                # engine-defined (eager vs lazy freeing), not measurable.
                break
            active, live_after = _live_tokens()
            free_after = driver.num_free_blocks()
            scheduled = all(generated[rid] >= 1 for rid in active)
            _check_block_consistency(
                used_blocks=total_blocks - free_after,
                live_tokens=live_after,
                batch_size=batch,
                block_size=block_size,
                fully_scheduled=scheduled,
                where=f"post-step {step_idx}",
            )
            if prefill_used is None and any(finished.values()):
                # A request finished while prefill had not completed for
                # the whole batch: the prefill reading cannot represent
                # the full batch. This check runs BEFORE the prefill
                # assignment below, so a finished wave can never sneak
                # a partial-batch prefill past it (the old order
                # assigned first and failed later with the misleading
                # decode-freeze message).
                raise ValueError(
                    "a request finished before prefill completed for the "
                    "whole batch; the prefill reading cannot represent "
                    "the full batch, refusing to sweep"
                )
            if prefill_used is None and scheduled:
                prefill_used = total_blocks - free_after
                prefill_live = live_after
                _emit(f"prefill reading: {prefill_used} blocks used, {prefill_live} tokens live")
            if prefill_used is not None and not any(finished.values()):
                # Last post-step with every request still alive: freeze the
                # decode reading here. Later steps may free finished waves'
                # blocks, which must not shrink the batch's reading.
                decode_used = total_blocks - free_after
                decode_live = live_after
        else:
            raise ValueError(
                f"requests did not finish within {max_steps} steps "
                f"(finished={finished}); the engine is misbehaving"
            )
        if prefill_used is None:
            raise ValueError("prefill never completed; no request produced a token")
        if decode_used is None:
            raise ValueError(
                "no post-decode step had every request still alive; "
                "the batch finished too staggered to read, refusing to sweep"
            )
        if decode_used < prefill_used:
            raise ValueError(
                f"decode reading ({decode_used} blocks) is smaller than the "
                f"prefill reading ({prefill_used} blocks); blocks were freed "
                "mid-sample, refusing to sweep"
            )

        def to_bytes(used: int) -> int:
            return used * block_size * theory_bpt

        return SampleReadings(
            kv_bytes_prefill=to_bytes(prefill_used),
            kv_bytes_decode=to_bytes(decode_used),
            live_tokens_prefill=prefill_live,
            live_tokens_decode=decode_live,
            used_blocks_prefill=prefill_used,
            used_blocks_decode=decode_used,
        )
    finally:
        driver.end_sample()


# ---------- fake engine driver (rung 0 dry run) ----------


class FakeBlockPool:
    """Stub pool exposing ``num_gpu_blocks`` / ``get_num_free_blocks()``.

    Mirrors the recipe the real driver probes for, so the sampling path
    cannot tell the fake apart by surface.
    """

    def __init__(self, total_blocks: int) -> None:
        if total_blocks <= 0:
            raise ValueError(f"total_blocks must be > 0, got {total_blocks}")
        self._total_blocks = total_blocks
        self._used_blocks = 0

    @property
    def num_gpu_blocks(self) -> int:
        return self._total_blocks

    def get_num_free_blocks(self) -> int:
        return self._total_blocks - self._used_blocks

    def _set_used_blocks(self, used: int) -> None:
        if used < 0 or used > self._total_blocks:
            raise ValueError(
                f"fake pool of {self._total_blocks} blocks cannot hold {used} used blocks"
            )
        self._used_blocks = used


@dataclass
class _FakeRequest:
    prompt_len: int
    max_tokens: int
    ignore_eos: bool = False
    generated: int = 0
    finished: bool = False
    freed: bool = False


class FakeEngineDriver:
    """Pure-Python engine stand-in for rung-0 dry runs.

    Mirrors the sampling recipe: the first step completes prefill and
    produces one token, each later step decodes one token, and a
    request's blocks are freed eagerly when it finishes (so the
    sampler's take-the-pre-finishing-step decode reading is exercised).
    Live requests hold ``ceil((prompt_len + generated) / block_size)``
    blocks. The prompt STRING is ignored; the fake is driven by
    ``prompt_len`` because there is no tokenizer here. Deterministic:
    no RNG, no noise, same calls give the same blocks.

    ``eos_after`` simulates a model that emits EOS early: when set, a
    request added with ``ignore_eos=False`` finishes ``eos_after``
    tokens after its prefill completes instead of running to
    ``max_tokens``. ``ignore_eos=True`` (the sampler's production
    setting) disables the early finish. ``None`` (default) means no
    EOS simulation at all, which is what the rung-0 dry run uses.
    """

    def __init__(
        self,
        *,
        total_blocks: int,
        block_size: int,
        fixed_overhead_bytes: int,
        eos_after: int | None = None,
    ) -> None:
        if block_size <= 0:
            raise ValueError(f"block_size must be > 0, got {block_size}")
        if fixed_overhead_bytes <= 0:
            raise ValueError(f"fixed_overhead_bytes must be > 0, got {fixed_overhead_bytes}")
        if eos_after is not None and eos_after < 1:
            raise ValueError(f"eos_after must be >= 1, got {eos_after}")
        self._block_size = block_size
        self.fixed_overhead_bytes = fixed_overhead_bytes
        self._eos_after = eos_after
        self._pool = FakeBlockPool(total_blocks)
        self._requests: dict[str, _FakeRequest] = {}

    @property
    def block_pool(self) -> FakeBlockPool:
        return self._pool

    def _sync_pool(self) -> None:
        used = 0
        for req in self._requests.values():
            # Blocks are allocated at schedule time (first step), not at
            # add_request: vLLM only enqueues on add_request.
            if not req.freed and req.generated >= 1:
                used += math.ceil((req.prompt_len + req.generated) / self._block_size)
        self._pool._set_used_blocks(used)

    def num_free_blocks(self) -> int:
        return self._pool.get_num_free_blocks()

    def add_request(
        self,
        request_id: str,
        prompt: str,
        max_tokens: int,
        temperature: float,
        prompt_len: int,
        ignore_eos: bool = False,
    ) -> None:
        if request_id in self._requests:
            raise ValueError(f"duplicate request id {request_id!r}")
        if max_tokens < 1:
            raise ValueError(f"max_tokens must be >= 1, got {max_tokens}")
        if prompt_len <= 0:
            raise ValueError(f"prompt_len must be > 0, got {prompt_len}")
        self._requests[request_id] = _FakeRequest(
            prompt_len=prompt_len, max_tokens=max_tokens, ignore_eos=ignore_eos
        )
        self._sync_pool()

    def _finish_after(self, req: _FakeRequest) -> int:
        """Generated-token count at which the request finishes.

        With EOS simulation on and ``ignore_eos=False``, the request
        ends at the early EOS instead of ``max_tokens`` (mirrors the
        real model emitting EOS after a few tokens on the fixed
        prompts). ``ignore_eos=True`` runs to ``max_tokens``.
        """
        if self._eos_after is not None and not req.ignore_eos:
            return min(req.max_tokens, self._eos_after)
        return req.max_tokens

    def step(self) -> list[StepOutput]:
        outputs: list[StepOutput] = []
        for rid, req in self._requests.items():
            if req.finished:
                continue
            req.generated += 1
            if req.generated >= self._finish_after(req):
                req.finished = True
                req.freed = True
            outputs.append(StepOutput(request_id=rid, new_tokens=1, finished=req.finished))
        self._sync_pool()
        return outputs

    def get_num_unfinished_requests(self) -> int:
        return sum(1 for req in self._requests.values() if not req.finished)

    def abort_request(self, request_id: str) -> None:
        req = self._requests.get(request_id)
        if req is None:
            raise ValueError(f"unknown request id {request_id!r}")
        req.freed = True
        self._sync_pool()

    def end_sample(self) -> None:
        for req in self._requests.values():
            req.freed = True
        self._sync_pool()


# ---------- sweep driver ----------


def _prefill_chunks(*, batch_size: int, seq_len: int, max_num_batched_tokens: int) -> int:
    """Chunked-prefill wave count for one cell: ceil(batch*seq_len / max_num_batched_tokens)."""
    return -(-batch_size * seq_len // max_num_batched_tokens)


# Attempt-10 root cause (campaign kv-scaling-20260929k, s2048 b16): the
# engine staggers chunked-prefill waves far wider than one step per
# wave — wave 1's requests finished all 5 decode tokens before wave 4's
# requests each generated one token, so the wave spacing exceeds 5
# engine steps and the old (chunks + 1) rule left no all-alive step.
_MEASURED_WAVE_SPACING_STEPS = 5  # lower bound from the attempt-10 exhibit
_DECODE_WINDOW_MARGIN = 6  # generous margin over the measured bound
EXTENDED_DECODE_TOKENS = _MEASURED_WAVE_SPACING_STEPS * _DECODE_WINDOW_MARGIN  # 30


def _cell_decode_tokens(
    *, batch_size: int, seq_len: int, decode_tokens: int, max_num_batched_tokens: int
) -> int:
    """Decode length for one cell.

    Chunked prefill staggers a batch into waves that finish on different
    steps; the decode reading is frozen at the last step with every
    request still alive, which only exists if the first wave cannot
    finish before the last wave's prefill completes. The window is
    sized from the MEASURED wave spacing (attempt 10: spacing > 5
    engine steps at s2048 b16) with a 6x margin, not from the chunk
    count — the old (chunks + 1) rule assumed one step per wave and
    the engine staggers wider than that. Decode length is arbitrary
    to the measured quantity (KV bytes per live token): a longer
    window only widens the all-alive overlap, it cannot bias the
    bytes/token reading, so margin is free. The regression uses
    measured live token counts, so the extra decode tokens only move
    the x-axis, never the slope. ``batch_size``/``seq_len``/
    ``max_num_batched_tokens`` are kept in the signature (callers
    unchanged) but no longer size the window.
    """
    return max(decode_tokens, EXTENDED_DECODE_TOKENS)


@dataclass
class SweepContext:
    """Everything a cell needs beyond the driver."""

    campaign_id: str
    model_id: str
    model_revision: str
    gpu: str
    vllm_version: str
    theory_bpt: int
    baseline_bytes: int  # F: fixed non-KV overhead, measured once per campaign
    block_size: int
    decode_tokens: int
    method: str
    max_num_batched_tokens: int = 8192  # pinned engine chunk budget (vLLM default)


def run_cell(
    *,
    driver: EngineDriver,
    ctx: SweepContext,
    seq_len: int,
    batch_size: int,
    n_samples: int,
    prompts: list[str],
    log: Callable[[str], None] | None = None,
) -> KvScalingCell:
    """Measure one (seq_len, batch_size) cell, recording ``n_samples``.

    Each prompt must encode to exactly ``seq_len`` tokens (built and
    verified by the caller). Each sample runs the decode window through
    the shared sampling path; the cell's sample value is the KV bytes
    live during decode from block accounting. ``total_tokens`` is the
    live token count at the decode reading, identical across samples.
    """
    if n_samples <= 0:
        raise ValueError(f"n_samples must be > 0, got {n_samples}")
    if len(prompts) != batch_size:
        raise ValueError(f"need {batch_size} prompts, got {len(prompts)}")
    # Chunked prefill staggers a batch into ceil(batch*seq_len /
    # max_num_batched_tokens) waves that finish on different steps. The
    # decode reading is frozen at the last step with every request still
    # alive, which only exists if the first wave cannot finish before the
    # last wave's prefill completes: decode for EXTENDED_DECODE_TOKENS
    # (measured wave spacing x margin; the old chunks+1 rule assumed one
    # step per wave and attempt 10's s2048 b16 grid died on it).
    chunks = _prefill_chunks(
        batch_size=batch_size, seq_len=seq_len, max_num_batched_tokens=ctx.max_num_batched_tokens
    )
    cell_decode_tokens = _cell_decode_tokens(
        batch_size=batch_size,
        seq_len=seq_len,
        decode_tokens=ctx.decode_tokens,
        max_num_batched_tokens=ctx.max_num_batched_tokens,
    )
    if log is not None and cell_decode_tokens != ctx.decode_tokens:
        log(
            f"cell s{seq_len} b{batch_size}: {chunks} prefill chunks, "
            f"decode_tokens {ctx.decode_tokens} -> {cell_decode_tokens}"
        )
    decode_samples: list[int] = []
    prefill_samples: list[int] = []
    live_counts: list[int] = []
    for i in range(n_samples):
        rids = [f"{ctx.campaign_id}-s{seq_len}-b{batch_size}-n{i}-r{k}" for k in range(batch_size)]
        reading = sample_decode_window(
            driver=driver,
            request_ids=rids,
            prompts=prompts,
            prompt_lens=[seq_len] * batch_size,
            decode_tokens=cell_decode_tokens,
            prefill_chunks=chunks,
            theory_bpt=ctx.theory_bpt,
            block_size=ctx.block_size,
            log=log,
        )
        decode_samples.append(reading.kv_bytes_decode)
        prefill_samples.append(reading.kv_bytes_prefill)
        live_counts.append(reading.live_tokens_decode)
    if any(c != live_counts[0] for c in live_counts):
        raise ValueError(
            f"live token count changed across samples: {live_counts}; "
            "the engine is not deterministic, refusing to sweep"
        )
    return KvScalingCell(
        model_id=ctx.model_id,
        gpu=ctx.gpu,
        vllm_version=ctx.vllm_version,
        seq_len=seq_len,
        batch_size=batch_size,
        n_samples=n_samples,
        kv_bytes_used_samples=decode_samples,
        kv_bytes_prefill_samples=prefill_samples,
        total_tokens=live_counts[0],
        baseline_bytes=ctx.baseline_bytes,
        method=ctx.method,
    )


# ---------- regression + knee ----------


def fit_bytes_per_token(
    total_tokens: Sequence[float],
    kv_bytes: Sequence[float],
) -> RegressionFit:
    """Least-squares fit of per-cell p50 KV bytes vs ``total_tokens``.

    The samples are already KV-only bytes from block accounting, so no
    baseline is subtracted. Returns slope (bytes per token), intercept,
    R^2, and residual standard error. Raises ValueError for fewer than
    2 points or zero variance in ``total_tokens``.
    """
    xs = list(total_tokens)
    ys = list(kv_bytes)
    if len(xs) != len(ys):
        raise ValueError("total_tokens and kv_bytes must have equal length")
    n = len(xs)
    if n < 2:
        raise ValueError(f"need at least 2 points for a fit, got {n}")
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    sxx = sum((x - mean_x) ** 2 for x in xs)
    if sxx == 0:
        raise ValueError("total_tokens has zero variance; slope is undefined")
    sxy = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys, strict=True))
    slope = sxy / sxx
    intercept = mean_y - slope * mean_x
    sse = sum((y - (slope * x + intercept)) ** 2 for x, y in zip(xs, ys, strict=True))
    sst = sum((y - mean_y) ** 2 for y in ys)
    r2 = 1.0 - sse / sst if sst > 0 else 0.0
    residual_se = (sse / (n - 2)) ** 0.5 if n > 2 else 0.0
    return RegressionFit(
        slope_bpt=slope,
        intercept_bytes=intercept,
        r2=r2,
        residual_se=residual_se,
    )


def find_knee(
    cells: Sequence[KvScalingCell],
    baseline_bytes: int,
) -> KneePoint:
    """Return the smallest measured point where KV dominates GPU memory.

    KV share at a cell is ``p50_bytes / (baseline_bytes + p50_bytes)``,
    where ``baseline_bytes`` is ``F``. The knee is the first cell (in
    ascending token order) where that share reaches 0.5. Raises
    ValueError when no measured cell reaches it, instead of inventing a
    knee beyond the data.
    """
    if baseline_bytes < 0:
        raise ValueError(f"baseline_bytes must be >= 0, got {baseline_bytes}")
    ordered = sorted(cells, key=lambda c: c.total_tokens)
    if not ordered:
        raise ValueError("no cells to search for a knee")
    for cell in ordered:
        share = cell.p50_bytes / (baseline_bytes + cell.p50_bytes)
        if share >= 0.5:
            return KneePoint(
                total_tokens=cell.total_tokens,
                kv_share=share,
                batch_size=cell.batch_size,
            )
    raise ValueError("KV share never reached 0.5 in the measured range")


def validate_against_theory(
    measured_bpt: float,
    theory_bpt: int,
    tol: float = 0.15,
) -> float:
    """Gate the sweep runbook calls before publishing any plot.

    Compares the fitted slope (``measured_bpt``) against the
    architectural prediction (``theory_bpt``). Returns the relative
    error ``|measured - theory| / theory``. Raises ValueError when the
    error exceeds ``tol`` (default 15%), failing loudly instead of
    publishing a plot the theory cannot explain. Also raises on
    invalid inputs (non-positive theory, negative measurement,
    negative tolerance).
    """
    if theory_bpt <= 0:
        raise ValueError(f"theory_bpt must be > 0, got {theory_bpt}")
    if measured_bpt < 0:
        raise ValueError(f"measured_bpt must be >= 0, got {measured_bpt}")
    if tol < 0:
        raise ValueError(f"tol must be >= 0, got {tol}")
    rel_err = abs(measured_bpt - theory_bpt) / theory_bpt
    if rel_err > tol:
        raise ValueError(
            f"measured slope {measured_bpt:.1f} B/token deviates {rel_err:.1%} "
            f"from theory {theory_bpt} B/token (tolerance {tol:.1%}); "
            "refusing to publish"
        )
    return rel_err


def check_dry_run_gate(
    regression: RegressionFit,
    injected_bpt: int,
    block_size: int,
    min_seq_len: int,
) -> None:
    """Rung-0 gate: the fake pipeline must recover the injected slope.

    The recovered slope must land within block-quantization bounds of
    the injected bytes/token (never below it, at most one block of
    rounding per ``min_seq_len`` tokens above it) with R^2 >= 0.98.
    Raises ValueError otherwise: the pod run is not launched.
    """
    if injected_bpt <= 0:
        raise ValueError(f"injected_bpt must be > 0, got {injected_bpt}")
    lo = injected_bpt * 0.999
    hi = injected_bpt * (1.0 + block_size / min_seq_len)
    if not lo <= regression.slope_bpt <= hi:
        raise ValueError(
            f"rung-0 gate failed: recovered slope {regression.slope_bpt:.1f} "
            f"outside block-quantization bounds [{lo:.1f}, {hi:.1f}] "
            f"of injected {injected_bpt}"
        )
    if regression.r2 < DRY_RUN_MIN_R2:
        raise ValueError(f"rung-0 gate failed: R^2 {regression.r2:.4f} < {DRY_RUN_MIN_R2}")


# ---------- cost preflight ----------


@dataclass
class CostPreflight:
    """Cost estimate printed (and refused on) before any GPU work."""

    campaign_id: str
    model_id: str
    model_revision: str
    seq_lens: list[int]
    batch_sizes: list[int]
    n_samples: int
    n_cells: int
    est_wall_seconds: float
    usd_per_hour: float | None
    est_cost_usd: float | None
    approved: bool
    out_dir: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "campaign_id": self.campaign_id,
            "model_id": self.model_id,
            "model_revision": self.model_revision,
            "seq_lens": self.seq_lens,
            "batch_sizes": self.batch_sizes,
            "n_samples": self.n_samples,
            "n_cells": self.n_cells,
            "est_wall_seconds": self.est_wall_seconds,
            "usd_per_hour": self.usd_per_hour,
            "est_cost_usd": self.est_cost_usd,
            "approved": self.approved,
            "out_dir": self.out_dir,
        }


def build_cost_preflight(
    *,
    campaign_id: str,
    model_id: str,
    model_revision: str,
    seq_lens: list[int],
    batch_sizes: list[int],
    n_samples: int,
    secs_per_cell: float,
    usd_per_hour: float | None,
    out_dir: Path,
    approved: bool,
) -> CostPreflight:
    """Build the preflight record. Wall time is a rough estimate.

    ``secs_per_cell`` is a planning assumption, not a measurement; the
    pod page's hourly rate is the source of truth at launch.
    """
    n_cells = len(seq_lens) * len(batch_sizes)
    est_wall = n_cells * secs_per_cell
    est_cost = None if usd_per_hour is None else est_wall / 3600.0 * usd_per_hour
    return CostPreflight(
        campaign_id=campaign_id,
        model_id=model_id,
        model_revision=model_revision,
        seq_lens=list(seq_lens),
        batch_sizes=list(batch_sizes),
        n_samples=n_samples,
        n_cells=n_cells,
        est_wall_seconds=est_wall,
        usd_per_hour=usd_per_hour,
        est_cost_usd=est_cost,
        approved=approved,
        out_dir=str(out_dir),
    )


def print_cost_preflight(preflight: CostPreflight) -> None:
    """Print the preflight block the runbook requires before GPU work."""
    rate = (
        f"${preflight.usd_per_hour:.2f}/hr"
        if preflight.usd_per_hour is not None
        else "unknown (pass --usd-per-hour)"
    )
    cost = f"${preflight.est_cost_usd:.2f}" if preflight.est_cost_usd is not None else "unknown"
    print("=== KV-scaling cost preflight ===")
    print(f"campaign:      {preflight.campaign_id}")
    print(f"model:         {preflight.model_id}")
    print(f"revision:      {preflight.model_revision}")
    print(f"grid:          seq_lens={preflight.seq_lens} batches={preflight.batch_sizes}")
    print(f"n_samples:     {preflight.n_samples}")
    print(f"pending cells: {preflight.n_cells}")
    print(f"est wall time: {preflight.est_wall_seconds / 60:.1f} min (rough)")
    print(f"hourly rate:   {rate}")
    print(f"est cost:      {cost}")
    print(f"output dir:    {preflight.out_dir}")
    print(f"approved:      {preflight.approved}")


# ---------- rung 0: fake-block-manager dry run ----------


def run_dry_run_campaign(
    *,
    campaign_id: str = "kv-scaling-dry-run",
    out_dir: Path = Path("bench/results/kv_scaling"),
    seq_lens: list[int] | None = None,
    batch_sizes: list[int] | None = None,
    n_samples: int = 5,
    theory_bpt: int = 12288,
    block_size: int = DEFAULT_BLOCK_SIZE,
    decode_tokens: int = DECODE_TOKENS,
    total_blocks: int = 1 << 15,
    fixed_overhead_bytes: int = 2_097_152,
    log: Callable[[str], None] | None = None,
) -> Path:
    """Rung 0: run the full pipeline against the fake block manager.

    Uses the REAL sampling code path (add_request / step / block reads /
    gates / regression / knee / writers) with a known injected
    bytes/token. Passes only if ``check_dry_run_gate`` passes. Needs no
    GPU, no torch, no vLLM. The fake pool is auto-sized to the grid so a
    large grid cannot silently overflow it. The injected fixed overhead
    is deliberately small (2 MiB) so the test grid crosses the KV-share
    knee; the real F is measured on the pod.
    """
    from bench.kv_scaling.results import write_campaign, write_summary_json

    seq_lens = seq_lens or [512, 2048, 8192]
    batch_sizes = batch_sizes or [1, 4, 16]
    if len(seq_lens) * len(batch_sizes) < 2:
        raise ValueError(
            "the dry run needs at least 2 grid cells to fit a slope; "
            f"got seq_lens={seq_lens} batches={batch_sizes}"
        )
    # Worst case live blocks: every sequence at full length plus decodes.
    # The decode length is per-cell (chunked-prefill waves), so size for
    # the same formula run_cell uses.
    needed = max(
        bs
        * math.ceil(
            (
                sl
                + _cell_decode_tokens(
                    batch_size=bs,
                    seq_len=sl,
                    decode_tokens=decode_tokens,
                    max_num_batched_tokens=8192,
                )
            )
            / block_size
        )
        for sl in seq_lens
        for bs in batch_sizes
    )
    pool_blocks = max(total_blocks, needed + 64)
    driver = FakeEngineDriver(
        total_blocks=pool_blocks,
        block_size=block_size,
        fixed_overhead_bytes=fixed_overhead_bytes,
    )
    ctx = SweepContext(
        campaign_id=campaign_id,
        model_id="dry-run/fake-block-manager",
        model_revision="fake",
        gpu="none (fake)",
        vllm_version="fake-0",
        theory_bpt=theory_bpt,
        baseline_bytes=fixed_overhead_bytes,
        block_size=block_size,
        decode_tokens=decode_tokens,
        method=(
            "fake block-manager dry run (rung 0), manual step loop, "
            f"max_tokens={decode_tokens}, ignore_eos=True, block_size={block_size}"
        ),
    )
    cells: list[KvScalingCell] = []
    for seq_len in seq_lens:
        for batch_size in batch_sizes:
            # Prompt strings are carriers only; the fake is driven by
            # prompt_len, which run_cell sets to seq_len.
            prompts = [f"fake-prompt-{seq_len}"] * batch_size
            cells.append(
                run_cell(
                    driver=driver,
                    ctx=ctx,
                    seq_len=seq_len,
                    batch_size=batch_size,
                    n_samples=n_samples,
                    prompts=prompts,
                    log=log,
                )
            )
    regression = fit_bytes_per_token(
        [float(c.total_tokens) for c in cells],
        [c.p50_bytes for c in cells],
    )
    check_dry_run_gate(regression, theory_bpt, block_size, min(seq_lens))
    knee = find_knee(cells, fixed_overhead_bytes)
    campaign = KvScalingCampaign(
        campaign_id=campaign_id,
        model_revision="fake",
        cells=cells,
        theory_bytes_per_token=theory_bpt,
        regression=regression,
        knee=knee,
    )
    out = out_dir / campaign_id
    write_campaign(campaign, results_root=out_dir, campaign_id=campaign_id)
    write_summary_json(out, campaign)
    print(
        f"dry-run PASS: recovered slope {regression.slope_bpt:.1f} B/token "
        f"vs injected {theory_bpt} (R^2={regression.r2:.4f}), "
        f"knee at {knee.total_tokens} tokens; artifacts in {out}"
    )
    return out


# ---------- vLLM sweep (GPU only) ----------


def run_vllm_sweep(
    *,
    model_id: str = MODEL_ID,
    revision: str = REVISION,
    seq_lens: list[int] | None = None,
    batch_sizes: list[int] | None = None,
    n_samples: int = 5,
    campaign_id: str = "kv-scaling",
    out_dir: Path = Path("bench/results/kv_scaling"),
    gpu_memory_utilization: float = 0.9,
    max_model_len: int = 32768,
    block_size: int = DEFAULT_BLOCK_SIZE,
    enforce_eager: bool = True,
    decode_tokens: int = DECODE_TOKENS,
    seed: int = 0,
    approve_cost: bool = False,
    dry_run: bool = False,
    usd_per_hour: float | None = None,
    secs_per_cell: float = 120.0,
) -> Path:
    """Run the full KV-scaling sweep.

    With ``dry_run=True`` this runs the rung-0 fake pipeline on CPU. For
    the real sweep it prints a cost preflight, writes ``preflight.json``,
    and refuses to start without ``approve_cost`` (or
    ``APPROVE_GPU_SPEND=yes``) unless the grid is a single smoke cell.
    Raises RuntimeError on a box without CUDA instead of faking data.
    """
    from bench.kv_scaling.results import write_preflight_json

    seq_lens = seq_lens if seq_lens is not None else [512, 2048, 8192]
    batch_sizes = batch_sizes if batch_sizes is not None else [1, 4, 16]
    if dry_run:
        return run_dry_run_campaign(
            campaign_id=campaign_id,
            out_dir=out_dir,
            seq_lens=seq_lens,
            batch_sizes=batch_sizes,
            n_samples=n_samples,
            block_size=block_size,
            decode_tokens=decode_tokens,
            log=print,
        )
    approved = approve_cost or os.environ.get("APPROVE_GPU_SPEND") == "yes"
    preflight = build_cost_preflight(
        campaign_id=campaign_id,
        model_id=model_id,
        model_revision=revision,
        seq_lens=seq_lens,
        batch_sizes=batch_sizes,
        n_samples=n_samples,
        secs_per_cell=secs_per_cell,
        usd_per_hour=usd_per_hour,
        out_dir=out_dir / campaign_id,
        approved=approved,
    )
    print_cost_preflight(preflight)
    write_preflight_json(out_dir / campaign_id, preflight.as_dict())
    if preflight.n_cells > 1 and not approved:
        raise RuntimeError(
            "refusing to start a multi-cell GPU sweep without cost approval: "
            "pass --approve-cost (single-cell smoke configs are exempt)"
        )
    if not cuda_available():
        raise RuntimeError("run_vllm_sweep requires a CUDA GPU; refusing to fake measurements")
    return _run_vllm_sweep_impl(  # pragma: no cover
        model_id=model_id,
        revision=revision,
        seq_lens=seq_lens,
        batch_sizes=batch_sizes,
        n_samples=n_samples,
        campaign_id=campaign_id,
        out_dir=out_dir,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
        block_size=block_size,
        enforce_eager=enforce_eager,
        decode_tokens=decode_tokens,
        seed=seed,
    )


class _PoolAdapter:
    """Uniform block-pool surface over whatever the engine exposes."""

    def __init__(
        self,
        total_blocks: int,
        free_fn: Callable[[], int],
        source: str,
    ) -> None:
        self._total_blocks = total_blocks
        self._free_fn = free_fn
        self.source = source

    @property
    def num_gpu_blocks(self) -> int:
        return self._total_blocks

    def get_num_free_blocks(self) -> int:
        return int(self._free_fn())


class VllmEngineDriver:
    """Manual-step driver over the vLLM v1 offline engine. GPU only."""

    def __init__(
        self,
        llm: Any,
        engine: Any,
        pool: _PoolAdapter,
        block_size: int,
        log: Callable[[str], None],
    ) -> None:
        self._llm = llm
        self._engine = engine
        self._pool = pool
        self._block_size = block_size
        self._log = log
        self._added: list[str] = []
        self._seen_len: dict[str, int] = {}
        self._shape_logged = False
        self._sampling_params_cls: Any = None

    @property
    def block_pool(self) -> _PoolAdapter:
        return self._pool

    def _torch(self) -> Any:
        import torch  # lazy: only present with CUDA

        return torch

    def num_free_blocks(self) -> int:
        self._torch().cuda.synchronize()
        return self._pool.get_num_free_blocks()

    def add_request(
        self,
        request_id: str,
        prompt: str,
        max_tokens: int,
        temperature: float,
        prompt_len: int,
        ignore_eos: bool = False,
    ) -> None:
        if self._sampling_params_cls is None:
            from vllm import SamplingParams  # lazy: vLLM only on GPU pods

            self._sampling_params_cls = SamplingParams
        params = self._sampling_params_cls(
            max_tokens=max_tokens, temperature=temperature, ignore_eos=ignore_eos
        )
        # prompt_len is for the fake driver; vLLM tokenizes the string.
        self._engine.add_request(request_id, prompt, params)
        self._added.append(request_id)

    def step(self) -> list[StepOutput]:
        raw_outputs: list[Any] = self._engine.step()
        outputs: list[StepOutput] = []
        for ro in raw_outputs:
            rid = str(getattr(ro, "request_id", ""))
            fin = getattr(ro, "finished", False)
            finished = bool(fin() if callable(fin) else fin)
            token_ids: list[int] = []
            for o in getattr(ro, "outputs", None) or []:
                tids = getattr(o, "token_ids", None)
                if tids:
                    token_ids = list(tids)
                    break
            prev = self._seen_len.get(rid, 0)
            if not self._shape_logged:
                self._log(
                    f"request-output shape: token_ids len={len(token_ids)}, "
                    f"previously seen={prev} "
                    "(cumulative when len grows, per-step otherwise)"
                )
                self._shape_logged = True
            # token_ids may be cumulative or per-step; handle both.
            new_tokens = len(token_ids) - prev if len(token_ids) >= prev else len(token_ids)
            self._seen_len[rid] = prev + new_tokens
            outputs.append(StepOutput(request_id=rid, new_tokens=new_tokens, finished=finished))
        return outputs

    def get_num_unfinished_requests(self) -> int:
        return int(self._engine.get_num_unfinished_requests())

    def abort_request(self, request_id: str) -> None:
        self._engine.abort_request(request_id)

    def end_sample(self) -> None:
        for rid in self._added:
            try:
                self._engine.abort_request(rid)
            except Exception as exc:  # noqa: BLE001 - best-effort cleanup
                self._log(f"abort_request({rid!r}) during cleanup: {exc}")
        self._added.clear()
        torch = self._torch()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()


def _probe_block_pool(engine: Any, log: Callable[[str], None]) -> _PoolAdapter:
    """Find the GPU block pool on the v1 engine; log the API surface.

    The engine must run IN-PROCESS (VLLM_ENABLE_V1_MULTIPROCESSING=0):
    with the default multiprocess mode, ``engine.engine_core`` is a
    SyncMPClient whose scheduler lives in the EngineCore subprocess and
    is unreachable from the parent (killed campaign
    kv-scaling-20260928d). In-process, ``engine.engine_core`` is an
    InprocClient whose ``.engine_core`` is the real EngineCore, so the
    first path below resolves. Fallbacks cover older layouts; when
    nothing resolves the sweep fails loudly instead of guessing.
    """
    attempts: list[str] = []
    getters: list[tuple[str, Callable[[], Any]]] = [
        (
            "engine_core.engine_core.scheduler.kv_cache_manager.block_pool",
            lambda: engine.engine_core.engine_core.scheduler.kv_cache_manager.block_pool,
        ),
        (
            "engine_core.scheduler.kv_cache_manager.block_pool",
            lambda: engine.engine_core.scheduler.kv_cache_manager.block_pool,
        ),
        (
            "engine_core.scheduler.block_manager",
            lambda: engine.engine_core.scheduler.block_manager,
        ),
        ("scheduler.block_manager", lambda: engine.scheduler.block_manager),
    ]
    for name, getter in getters:
        try:
            pool = getter()
        except AttributeError as exc:
            attempts.append(f"{name}: missing ({exc})")
            continue
        public_attrs = sorted(a for a in dir(pool) if not a.startswith("_"))
        log(f"block pool candidate {name}: public attrs={public_attrs}")
        total = getattr(pool, "num_gpu_blocks", None)
        free_fn = getattr(pool, "get_num_free_blocks", None)
        free_name = "get_num_free_blocks"
        if not callable(free_fn):
            free_fn = getattr(pool, "get_num_free_gpu_blocks", None)
            free_name = "get_num_free_gpu_blocks"
        if isinstance(total, int) and total > 0 and callable(free_fn):
            log(f"block pool OK via {name}: num_gpu_blocks={total}, free reader={free_name}")
            return _PoolAdapter(total, free_fn, source=name)
        attempts.append(f"{name}: unusable (num_gpu_blocks={total!r})")
    raise RuntimeError(
        f"could not find a readable GPU block pool on this vLLM build; attempts: {attempts}"
    )


def _build_vllm_driver(
    llm: Any,
    *,
    block_size: int,
    theory_bpt: int,
    log: Callable[[str], None],
) -> tuple[VllmEngineDriver, int, int]:
    """Probe the engine, verify block_size, measure F.

    Returns (driver, F, max_num_batched_tokens): the chunk budget is a
    constructor input pinned in _llm_kwargs; verify the engine agrees
    because the per-cell decode-token math depends on it.
    """
    import torch  # lazy: only present with CUDA

    engine = llm.llm_engine
    log(f"engine object: {type(engine).__module__}.{type(engine).__name__}")
    pool = _probe_block_pool(engine, log)
    # block_size is a constructor input; verify the engine agrees.
    for path in ("cache_config.block_size", "engine_core.cache_config.block_size"):
        obj: Any = engine
        try:
            for part in path.split("."):
                obj = getattr(obj, part)
        except AttributeError:
            continue
        if int(obj) != block_size:
            raise ValueError(
                f"engine reports block_size={obj} via {path} but the sweep "
                f"was configured with block_size={block_size}"
            )
        log(f"block_size verified via {path}: {obj}")
        break
    else:
        log(
            "warning: could not read block_size back from the engine; "
            f"trusting the configured value {block_size}"
        )
    max_batched: int | None = None
    for path in (
        "vllm_config.scheduler_config.max_num_batched_tokens",
        "scheduler_config.max_num_batched_tokens",
        "engine_core.vllm_config.scheduler_config.max_num_batched_tokens",
    ):
        obj2: Any = engine
        try:
            for part in path.split("."):
                obj2 = getattr(obj2, part)
        except AttributeError:
            continue
        max_batched = int(obj2)
        log(f"max_num_batched_tokens read via {path}: {max_batched}")
        break
    if max_batched is None:
        log(
            "warning: could not read max_num_batched_tokens back from the "
            "engine; trusting the configured value 8192"
        )
        max_batched = 8192
    if max_batched != 8192:
        raise ValueError(
            f"engine reports max_num_batched_tokens={max_batched} but the "
            "sweep pinned 8192 in _llm_kwargs; refusing to sweep"
        )
    allocated = int(torch.cuda.memory_allocated())
    pool_bytes = pool.num_gpu_blocks * block_size * theory_bpt
    fixed = allocated - pool_bytes
    log(
        f"F measurement: memory_allocated={allocated}, pool_bytes={pool_bytes} "
        f"({pool.num_gpu_blocks} blocks x {block_size} x {theory_bpt} B/token), "
        f"F={fixed}"
    )
    if not 0 < fixed < allocated:
        raise ValueError(
            f"F={fixed} is not in (0, {allocated}); the pool accounting is wrong, refusing to sweep"
        )
    driver = VllmEngineDriver(llm, engine, pool, block_size, log)
    return driver, fixed, max_batched


def _raw_config_json(model_id: str, revision: str) -> dict[str, Any]:
    """Fetch the raw config.json from the Hub at the pinned revision.

    Plain HTTPS + json, no transformers: transformers' to_dict() drops
    torch_dtype on the versions vLLM 0.11.2 pulls in (that lossiness
    killed campaign kv-scaling-20260928c through the old AutoConfig
    fallback), but the raw file carries it. Only used when no engine
    config path yields a dict.
    """
    import json
    import urllib.request

    url = f"https://huggingface.co/{model_id}/resolve/{revision}/config.json"
    with urllib.request.urlopen(url, timeout=60) as resp:
        data = json.load(resp)
    if not isinstance(data, dict):
        raise ValueError(f"unexpected config.json payload for {model_id}@{revision}")
    return data


def _resolve_theory(llm: Any, model_id: str, revision: str) -> Any:
    """Build the HfKvTheory from the engine config, else from HF directly.

    vLLM's wrapped HF configs are lossy: vLLM 0.11.2's
    hf_text_config.to_dict() drops torch_dtype even though the raw
    config.json carries it (killed campaign kv-scaling-20260928b), and
    transformers' own to_dict() proved equally lossy on the pod
    (killed kv-scaling-20260928c via the old AutoConfig fallback). So
    the element width comes from the engine's authoritative runtime
    dtype first (vLLM resolves model_config.dtype before the model
    loads), then the raw config.json at the pinned revision. The
    provenance string records which sources were used.
    """
    engine = llm.llm_engine
    cfg: dict[str, Any] | None = None
    source = ""
    for path in (
        "model_config.hf_text_config",
        "model_config.hf_config",
        "engine_core.model_config.hf_text_config",
    ):
        obj: Any = engine
        try:
            for part in path.split("."):
                obj = getattr(obj, part)
            cfg = obj.to_dict()
            source = f"engine:{path}"
            break
        except AttributeError:
            continue
    if cfg is None:
        cfg = _raw_config_json(model_id, revision)
        source = "hub:config.json"
    if "torch_dtype" not in cfg:
        eng_dtype = getattr(getattr(engine, "model_config", None), "dtype", None)
        if eng_dtype is not None:
            cfg = dict(cfg)
            cfg["torch_dtype"] = str(eng_dtype)
            source += "+runtime-dtype"
    return config_from_hf_config(cfg, revision=revision), source


def _llm_kwargs(
    *,
    model_id: str,
    revision: str,
    gpu_memory_utilization: float,
    max_model_len: int,
    block_size: int,
    enforce_eager: bool,
    seed: int,
) -> dict[str, Any]:
    """Keyword args for the vLLM ``LLM`` constructor.

    ``enable_prefix_caching=False`` is load-bearing for the measurement:
    the sweep builds identical prompts for every request in a batch, and
    with prefix caching on vLLM shares KV blocks across the batch's
    requests, so block accounting undercounts live tokens (killed
    campaign kv-scaling-20260928g's grid: 1056 token-slots for 2050
    live tokens at batch 2). With caching off, every request holds its
    own blocks and the accounting matches the measurement model.

    ``max_num_batched_tokens`` is pinned (vLLM's default) because the
    per-cell decode-token math depends on the prefill chunk count, which
    depends on it; the driver reads it back from the engine to verify.
    """
    return {
        "model": model_id,
        "revision": revision,
        "gpu_memory_utilization": gpu_memory_utilization,
        "max_model_len": max_model_len,
        "max_num_batched_tokens": 8192,
        "block_size": block_size,
        "enforce_eager": enforce_eager,
        "seed": seed,
        "enable_prefix_caching": False,
    }


def _run_vllm_sweep_impl(  # pragma: no cover - GPU only, never runs on CPU CI
    *,
    model_id: str,
    revision: str,
    seq_lens: list[int],
    batch_sizes: list[int],
    n_samples: int,
    campaign_id: str,
    out_dir: Path,
    gpu_memory_utilization: float,
    max_model_len: int,
    block_size: int,
    enforce_eager: bool,
    decode_tokens: int,
    seed: int,
) -> Path:
    import torch  # lazy: only present with CUDA
    import vllm as _vllm
    from vllm import LLM  # lazy: vLLM only exists on GPU pods

    from bench.kv_scaling.results import (
        read_cell_json,
        write_cell_json,
    )

    vllm_version = str(_vllm.__version__)
    log: Callable[[str], None] = print
    log(f"loading {model_id} @ {revision} with vLLM {vllm_version}")
    # The block-pool reader needs the scheduler in THIS process.
    # vLLM_ENABLE_V1_MULTIPROCESSING=0 makes LLMEngine use InprocClient
    # (real EngineCore in-process) instead of SyncMPClient (scheduler
    # unreachable across the subprocess boundary; killed campaign
    # kv-scaling-20260928d). setdefault: an explicit "1" is respected,
    # and the probe then fails loudly instead of reading a wrong pool.
    # (The bootstrap exports this too; this covers direct invocation.)
    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    llm = LLM(
        **_llm_kwargs(
            model_id=model_id,
            revision=revision,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            block_size=block_size,
            enforce_eager=enforce_eager,
            seed=seed,
        )
    )
    theory, theory_source = _resolve_theory(llm, model_id, revision)
    log(f"theory from {theory_source}: {theory.bytes_per_token} B/token")
    driver, fixed_overhead, max_num_batched_tokens = _build_vllm_driver(
        llm, block_size=block_size, theory_bpt=theory.bytes_per_token, log=log
    )
    gpu_name = str(torch.cuda.get_device_name(0))
    ctx = SweepContext(
        campaign_id=campaign_id,
        model_id=model_id,
        model_revision=theory.revision,
        gpu=gpu_name,
        vllm_version=vllm_version,
        theory_bpt=theory.bytes_per_token,
        baseline_bytes=fixed_overhead,
        block_size=block_size,
        decode_tokens=decode_tokens,
        max_num_batched_tokens=max_num_batched_tokens,
        method=(
            f"vllm-{vllm_version} offline engine, manual step loop, "
            f"block-manager accounting, max_tokens={decode_tokens}, "
            f"ignore_eos=True, enforce_eager={enforce_eager}"
        ),
    )
    tokenizer = llm.get_tokenizer()
    campaign_out = out_dir / campaign_id
    campaign_out.mkdir(parents=True, exist_ok=True)

    cells: list[KvScalingCell] = []
    for seq_len in seq_lens:
        for batch_size in batch_sizes:
            cell_path = campaign_out / f"cell_s{seq_len}_b{batch_size}.json"
            cell = read_cell_json(
                cell_path,
                seq_len=seq_len,
                batch_size=batch_size,
                n_samples=n_samples,
                model_id=model_id,
            )
            if cell is not None:
                log(f"resume: reusing {cell_path}")
            else:
                prompt = build_exact_length_prompts(tokenizer, seq_len)
                cell = run_cell(
                    driver=driver,
                    ctx=ctx,
                    seq_len=seq_len,
                    batch_size=batch_size,
                    n_samples=n_samples,
                    prompts=[prompt] * batch_size,
                    log=log,
                )
                write_cell_json(cell_path, cell)
                log(
                    f"cell done: seq_len={seq_len} batch={batch_size} "
                    f"p50={cell.p50_bytes:,.0f} bytes"
                )
            cells.append(cell)

    _finish_campaign_artifacts(
        cells=cells,
        campaign_id=campaign_id,
        theory=theory,
        fixed_overhead=fixed_overhead,
        out_dir=out_dir,
        campaign_out=campaign_out,
        log=log,
    )
    return campaign_out


def _finish_campaign_artifacts(
    *,
    cells: list[KvScalingCell],
    campaign_id: str,
    theory: HfKvTheory,
    fixed_overhead: int,
    out_dir: Path,
    campaign_out: Path,
    log: Callable[[str], None],
) -> None:
    """Regression fit, knee, and campaign/summary JSON for multi-cell runs.

    A slope needs at least 2 points, so a single-cell (smoke) run skips
    the fit: its gates are the block-consistency checks inside
    ``sample_decode_window`` plus a positive F, all done before this
    point. Fitting a regression on one point is undefined (killed
    campaign kv-scaling-20260928f's smoke cell with "need at least 2
    points for a fit, got 1" — the first campaign to get past cell
    measurement). The smoke's cell JSON is already on disk and the full
    grid resumes it via ``read_cell_json``.
    """
    from bench.kv_scaling.results import write_campaign, write_summary_json

    if len(cells) >= 2:
        regression = fit_bytes_per_token(
            [float(c.total_tokens) for c in cells],
            [c.p50_bytes for c in cells],
        )
        log(
            f"regression: slope={regression.slope_bpt:.1f} B/token "
            f"(theory {theory.bytes_per_token}), R^2={regression.r2:.4f}"
        )
        # Runbook gate: do not publish a plot the theory cannot explain.
        validate_against_theory(regression.slope_bpt, theory.bytes_per_token)

        knee = find_knee(cells, fixed_overhead)
        log(f"knee: {knee.total_tokens} tokens, KV share {knee.kv_share:.2f}")

        campaign = KvScalingCampaign(
            campaign_id=campaign_id,
            model_revision=theory.revision,
            cells=cells,
            theory_bytes_per_token=theory.bytes_per_token,
            regression=regression,
            knee=knee,
        )
        write_campaign(campaign, results_root=out_dir, campaign_id=campaign_id)
        write_summary_json(campaign_out, campaign)
        log(f"campaign written to {campaign_out}")
    else:
        log(
            "single-cell run: skipping the campaign regression fit "
            "(a slope needs at least 2 points); cell JSON already written"
        )


__all__ = [
    "DEFAULT_BLOCK_SIZE",
    "DECODE_TOKENS",
    "DRY_RUN_MIN_R2",
    "MODEL_ID",
    "REVISION",
    "BlockPoolLike",
    "CostPreflight",
    "EngineDriver",
    "FakeBlockPool",
    "FakeEngineDriver",
    "SampleReadings",
    "StepOutput",
    "SweepContext",
    "TokenizerLike",
    "VllmEngineDriver",
    "_finish_campaign_artifacts",
    "_llm_kwargs",
    "build_cost_preflight",
    "build_exact_length_prompts",
    "check_dry_run_gate",
    "cuda_available",
    "find_knee",
    "fit_bytes_per_token",
    "print_cost_preflight",
    "run_cell",
    "run_dry_run_campaign",
    "run_vllm_sweep",
    "sample_decode_window",
    "validate_against_theory",
]
