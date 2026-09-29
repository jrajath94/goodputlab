"""Tests for the KV-cache scaling study (block-manager methodology).

Everything GPU-free runs here on CPU: the fake block manager drives
the REAL sampling code path (add_request / step / block reads / gates),
so rung 0 of the runbook is tested, not just asserted. The vLLM driver
is GPU-gated and never touched by these tests.
"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path

import pytest

from bench.kv_scaling.__main__ import build_parser, main
from bench.kv_scaling.measure import (
    DECODE_TOKENS,
    FakeEngineDriver,
    StepOutput,
    SweepContext,
    _cell_decode_tokens,
    _finish_campaign_artifacts,
    _llm_kwargs,
    _prefill_chunks,
    build_cost_preflight,
    build_exact_length_prompts,
    check_dry_run_gate,
    find_knee,
    fit_bytes_per_token,
    print_cost_preflight,
    run_cell,
    run_dry_run_campaign,
    run_vllm_sweep,
    sample_decode_window,
    validate_against_theory,
)
from bench.kv_scaling.results import (
    SCHEMA_VERSION,
    read_cell_json,
    write_campaign,
    write_cell_json,
    write_preflight_json,
    write_summary_json,
)
from bench.kv_scaling.schema import (
    KneePoint,
    KvScalingCampaign,
    KvScalingCell,
    RegressionFit,
)
from bench.kv_scaling.theory import HfKvTheory

BPT = 12288  # Qwen2.5-0.5B theory value (TRD section 2)
BLOCK = 16


class WordTokenizer:
    """Deterministic stub: one token per whitespace-separated word."""

    def encode(self, text: str) -> list[int]:
        return list(range(len(text.split())))

    def decode(self, ids: list[int]) -> str:
        return "w " * len(ids)


class BrokenTokenizer:
    """decode/encode round trip never preserves length."""

    def encode(self, text: str) -> list[int]:
        return [1] * (len(text.split()) // 2)

    def decode(self, ids: list[int]) -> str:
        return "w " * len(ids)


class MergingTokenizer:
    """Reproduces the Qwen2.5-0.5B tile-boundary merge (campaign e).

    The boundary sequence " " + "The" encodes as ONE token (" The"),
    while a lone trailing " " is its own token. Naive tile-decode-encode
    therefore loses one token per tile boundary; the builder must resolve
    merges against the true encoding before truncating.
    """

    def __init__(self) -> None:
        self._words: dict[str, int] = {}

    def _word_id(self, word: str) -> int:
        if word not in self._words:
            self._words[word] = 3 + len(self._words)
        return self._words[word]

    def encode(self, text: str) -> list[int]:
        ids: list[int] = []
        for piece in re.findall(r" The| |[^ ]+", text):
            if piece == " The":
                ids.append(2)
            elif piece == " ":
                ids.append(1)
            else:
                ids.append(self._word_id(piece))
        return ids

    def decode(self, ids: list[int]) -> str:
        inv = {v: k for k, v in self._words.items()}
        parts = {1: " ", 2: " The"}
        parts.update({i: w for i, w in inv.items()})
        return "".join(parts[i] for i in ids)


def _ctx(**over: object) -> SweepContext:
    base: dict[str, object] = {
        "campaign_id": "test",
        "model_id": "test/model",
        "model_revision": "abc123",
        "gpu": "none (fake)",
        "vllm_version": "fake-0",
        "theory_bpt": BPT,
        "baseline_bytes": 1_000_000,
        "block_size": BLOCK,
        "decode_tokens": DECODE_TOKENS,
        "method": "test",
    }
    base.update(over)
    return SweepContext(**base)  # type: ignore[arg-type]


def _sample(seq_len: int = 64, batch_size: int = 2, decode_tokens: int = DECODE_TOKENS):
    driver = FakeEngineDriver(total_blocks=4096, block_size=BLOCK, fixed_overhead_bytes=1_000_000)
    rids = [f"r{k}" for k in range(batch_size)]
    reading = sample_decode_window(
        driver=driver,
        request_ids=rids,
        prompts=["x"] * batch_size,
        prompt_lens=[seq_len] * batch_size,
        decode_tokens=decode_tokens,
        theory_bpt=BPT,
        block_size=BLOCK,
    )
    return driver, reading


# ---------- fake engine driver ----------


def test_fake_driver_is_deterministic():
    def run() -> list[int]:
        driver = FakeEngineDriver(total_blocks=4096, block_size=BLOCK, fixed_overhead_bytes=1)
        seen = []
        driver.add_request("a", "prompt", 4, 0.0, 64)
        driver.add_request("b", "prompt", 4, 0.0, 64)
        for _ in range(4):
            driver.step()
            seen.append(driver.num_free_blocks())
        driver.end_sample()
        return seen

    assert run() == run()


def test_fake_driver_block_math():
    driver = FakeEngineDriver(total_blocks=100, block_size=BLOCK, fixed_overhead_bytes=1)
    assert driver.block_pool.num_gpu_blocks == 100
    assert driver.num_free_blocks() == 100
    driver.add_request("a", "prompt", 4, 0.0, 64)
    # prefill not stepped yet: nothing allocated
    assert driver.num_free_blocks() == 100
    driver.step()  # prefill + first token: ceil(65/16) = 5 blocks
    assert driver.num_free_blocks() == 95
    assert driver.get_num_unfinished_requests() == 1
    driver.abort_request("a")
    assert driver.num_free_blocks() == 100


def test_fake_driver_rejects_bad_inputs():
    driver = FakeEngineDriver(total_blocks=100, block_size=BLOCK, fixed_overhead_bytes=1)
    with pytest.raises(ValueError):
        driver.add_request("a", "p", 0, 0.0, 64)
    with pytest.raises(ValueError):
        driver.add_request("a", "p", 4, 0.0, 0)
    driver.add_request("a", "p", 4, 0.0, 64)
    with pytest.raises(ValueError):
        driver.add_request("a", "p", 4, 0.0, 64)  # duplicate
    with pytest.raises(ValueError):
        FakeEngineDriver(total_blocks=0, block_size=BLOCK, fixed_overhead_bytes=1)


def test_fake_pool_exhaustion_fails_loudly():
    driver = FakeEngineDriver(total_blocks=4, block_size=BLOCK, fixed_overhead_bytes=1)
    driver.add_request("a", "p", 4, 0.0, 64)  # needs 5 blocks at schedule
    with pytest.raises(ValueError, match="cannot hold"):
        driver.step()


# ---------- sampling path through the fake ----------


def test_sample_decode_window_happy_path():
    driver, reading = _sample(seq_len=64, batch_size=2)
    # prefill: ceil(65/16) = 5 blocks per seq, 10 total
    assert reading.used_blocks_prefill == 10
    assert reading.kv_bytes_prefill == 10 * BLOCK * BPT
    assert reading.live_tokens_prefill == 2 * 65
    # decode reading: last all-alive state, 3 tokens decoded per request
    assert reading.live_tokens_decode == 2 * 67
    assert reading.used_blocks_decode == 10  # ceil(67/16) = 5 per seq
    assert reading.kv_bytes_decode == 10 * BLOCK * BPT
    assert reading.kv_bytes_decode >= reading.kv_bytes_prefill
    # sample cleaned up: pool fully free again
    assert driver.num_free_blocks() == driver.block_pool.num_gpu_blocks


def test_sample_decode_window_block_growth():
    # seq_len 62: prefill ceil(63/16)=4, decode ceil(65/16)=5 per seq:
    # the decode reading must be strictly larger.
    _, reading = _sample(seq_len=62, batch_size=1)
    assert reading.used_blocks_decode == 5
    assert reading.used_blocks_prefill == 4
    assert reading.kv_bytes_decode > reading.kv_bytes_prefill


def test_sample_decode_window_validates_inputs():
    driver = FakeEngineDriver(total_blocks=4096, block_size=BLOCK, fixed_overhead_bytes=1)
    with pytest.raises(ValueError):
        sample_decode_window(
            driver=driver,
            request_ids=[],
            prompts=[],
            prompt_lens=[],
            theory_bpt=BPT,
            block_size=BLOCK,
        )
    with pytest.raises(ValueError):
        sample_decode_window(
            driver=driver,
            request_ids=["a"],
            prompts=["x", "y"],
            prompt_lens=[64, 64],
            theory_bpt=BPT,
            block_size=BLOCK,
        )
    with pytest.raises(ValueError):
        sample_decode_window(
            driver=driver,
            request_ids=["a"],
            prompts=["x"],
            prompt_lens=[64],
            theory_bpt=0,
            block_size=BLOCK,
        )


class LyingPoolDriver(FakeEngineDriver):
    """Reports far fewer free blocks than reality: gate must fire."""

    def num_free_blocks(self) -> int:
        return 0


def test_consistency_gate_fires_on_lying_block_reader():
    driver = LyingPoolDriver(total_blocks=4096, block_size=BLOCK, fixed_overhead_bytes=1)
    with pytest.raises(ValueError, match="block accounting"):
        sample_decode_window(
            driver=driver,
            request_ids=["a"],
            prompts=["x"],
            prompt_lens=[64],
            theory_bpt=BPT,
            block_size=BLOCK,
        )
    # even on gate failure the sample is cleaned up (check the real
    # pool state: this driver's num_free_blocks always lies)
    assert driver.block_pool.get_num_free_blocks() == 4096


def test_step_output_shape():
    out = StepOutput(request_id="r", new_tokens=1, finished=False)
    assert out.new_tokens == 1 and not out.finished


# ---------- run_cell ----------


def test_run_cell_records_decode_samples():
    driver = FakeEngineDriver(total_blocks=4096, block_size=BLOCK, fixed_overhead_bytes=1_000_000)
    cell = run_cell(
        driver=driver,
        ctx=_ctx(),
        seq_len=64,
        batch_size=2,
        n_samples=3,
        prompts=["x", "x"],
    )
    assert cell.n_samples == 3
    # Extended decode window (30): freeze at post-step 29 with
    # generated (29, 29) -> 12 blocks; prefill still reads at step 1.
    assert cell.kv_bytes_used_samples == [12 * BLOCK * BPT] * 3
    assert cell.kv_bytes_prefill_samples == [10 * BLOCK * BPT] * 3
    assert cell.total_tokens == 2 * 93  # (seq_len + 29 decoded) * batch
    assert cell.baseline_bytes == 1_000_000
    assert cell.p50_bytes == 12 * BLOCK * BPT
    assert cell.measured_bytes_per_token == pytest.approx(cell.p50_bytes / cell.total_tokens)


def test_run_cell_rejects_bad_inputs():
    driver = FakeEngineDriver(total_blocks=4096, block_size=BLOCK, fixed_overhead_bytes=1)
    with pytest.raises(ValueError):
        run_cell(
            driver=driver,
            ctx=_ctx(),
            seq_len=64,
            batch_size=2,
            n_samples=0,
            prompts=["x", "x"],
        )
    with pytest.raises(ValueError):
        run_cell(
            driver=driver,
            ctx=_ctx(),
            seq_len=64,
            batch_size=2,
            n_samples=2,
            prompts=["x"],
        )


# ---------- exact-length prompts ----------


def test_build_exact_length_prompts():
    prompt = build_exact_length_prompts(WordTokenizer(), 128)
    assert len(WordTokenizer().encode(prompt)) == 128


def test_build_exact_length_prompts_rejects_broken_tokenizer():
    with pytest.raises(ValueError, match="round trip"):
        build_exact_length_prompts(BrokenTokenizer(), 128)


def test_build_exact_length_prompts_rejects_nonpositive():
    with pytest.raises(ValueError):
        build_exact_length_prompts(WordTokenizer(), 0)


def _theory() -> HfKvTheory:
    return HfKvTheory(
        n_layers=24,
        n_kv_heads=2,
        head_dim=64,
        bytes_per_element=2,
        bytes_per_token=BPT,
        revision="rev",
    )


def test_finish_campaign_artifacts_skips_fit_for_single_cell(tmp_path):
    # Campaign f's smoke died fitting a regression on its 1 cell
    # ("need at least 2 points for a fit, got 1"). The smoke must pass
    # with only its cell JSON on disk; the grid resumes it later.
    logs: list[str] = []
    _finish_campaign_artifacts(
        cells=[_cell(1027, 12_976_128.0)],
        campaign_id="smoke",
        theory=_theory(),
        fixed_overhead=1_000_000,
        out_dir=tmp_path,
        campaign_out=tmp_path / "smoke",
        log=logs.append,
    )
    assert not (tmp_path / "smoke").exists()
    assert any("single-cell" in m for m in logs)


def test_finish_campaign_artifacts_writes_campaign_for_two_cells(tmp_path):
    logs: list[str] = []
    _finish_campaign_artifacts(
        cells=[_cell(1000, BPT * 1000.0), _cell(2000, BPT * 2000.0)],
        campaign_id="grid",
        theory=_theory(),
        fixed_overhead=1_000_000,
        out_dir=tmp_path,
        campaign_out=tmp_path / "grid",
        log=logs.append,
    )
    assert (tmp_path / "grid" / "cells.json").exists()
    assert (tmp_path / "grid" / "summary.json").exists()
    payload = json.loads((tmp_path / "grid" / "summary.json").read_text())
    assert payload["slope_bytes_per_token"] == pytest.approx(BPT)
    assert payload["r2"] == pytest.approx(1.0)


def test_llm_kwargs_disable_prefix_caching():
    # Campaign g's grid died because identical prompts + prefix caching
    # made vLLM share KV blocks across the batch's requests: block
    # accounting reported 1056 token-slots for 2050 live tokens. The
    # measurement model needs every request to hold its own blocks.
    kwargs = _llm_kwargs(
        model_id="m",
        revision="r",
        gpu_memory_utilization=0.9,
        max_model_len=32768,
        block_size=16,
        enforce_eager=True,
        seed=0,
    )
    assert kwargs["enable_prefix_caching"] is False
    assert kwargs["max_num_batched_tokens"] == 8192
    assert kwargs["model"] == "m"
    assert kwargs["block_size"] == 16


def test_prefill_chunks_and_cell_decode_tokens():
    # 16384 prompt tokens at a chunk budget of 8192 -> 2 waves.
    assert _prefill_chunks(batch_size=16, seq_len=1024, max_num_batched_tokens=8192) == 2
    # Attempt-10 (campaign kv-scaling-20260929k, s2048 b16): the engine
    # staggers chunked-prefill waves wider than 5 engine steps, so the
    # old (chunks + 1) rule left no all-alive step. The window is now
    # sized from the MEASURED spacing (5 steps) with a 6x margin = 30,
    # independent of the chunk count; decode length is arbitrary to
    # the measured bytes/token, so margin is free.
    assert (
        _cell_decode_tokens(
            batch_size=16, seq_len=1024, decode_tokens=4, max_num_batched_tokens=8192
        )
        == 30
    )
    assert (
        _cell_decode_tokens(
            batch_size=16, seq_len=32768, decode_tokens=4, max_num_batched_tokens=8192
        )
        == 30
    )
    # A longer caller-supplied window is still honored.
    assert (
        _cell_decode_tokens(
            batch_size=16, seq_len=1024, decode_tokens=64, max_num_batched_tokens=8192
        )
        == 64
    )


class _ChunkedFakeDriver(FakeEngineDriver):
    """Fake engine with chunked-prefill stagger: wave-2 requests are not
    scheduled on the first step, so the waves finish on different steps
    and finished waves free their blocks (exactly what killed campaign
    kv-scaling-20260928h's grid at batch 16). ``wave2_suffix`` matches
    the request-id tail run_cell builds (``...-r1``)."""

    def __init__(self, *args, wave2_suffix="-r1", **kwargs):
        super().__init__(*args, **kwargs)
        self._wave2_suffix = wave2_suffix
        self._steps = 0

    def _is_wave2(self, rid):
        return rid.endswith(self._wave2_suffix)

    def step(self):  # noqa: D102
        self._steps += 1
        if self._steps == 1:
            held = {}
            for rid in list(self._requests):
                if self._is_wave2(rid):
                    held[rid] = self._requests.pop(rid)
            try:
                return super().step()
            finally:
                self._requests.update(held)
                self._sync_pool()
        return super().step()

    def end_sample(self):  # noqa: D102
        self._steps = 0
        super().end_sample()


def test_run_cell_survives_staggered_prefill_waves():
    # Campaign h: wave 1 finished (freeing its blocks) before wave 2
    # finished, so the old all-finished-step decode reading undercounted
    # the batch and tripped the gate. The decode reading is now frozen at
    # the last step with every request still alive.
    ctx = _ctx(max_num_batched_tokens=64)  # 2x64 tokens -> 2 chunks
    driver = _ChunkedFakeDriver(total_blocks=4096, block_size=BLOCK, fixed_overhead_bytes=1_000_000)
    cell = run_cell(
        driver=driver,
        ctx=ctx,
        seq_len=64,
        batch_size=2,
        n_samples=2,
        prompts=["x", "x"],
        log=None,
    )
    # decode_tokens=30 (extended window): wave 1 finishes at end of
    # step 30, wave 2 at end of step 31. Frozen at end of step 29:
    # wave 1 holds 93 tokens, wave 2 holds 92 -> 6 blocks each;
    # live = 93 + 92 = 185.
    assert cell.kv_bytes_used_samples[0] == 12 * BLOCK * BPT
    assert cell.total_tokens == 185


def test_sample_decode_window_fails_loud_on_finish_before_prefill():
    # decode_tokens=1 with staggered waves: wave 1 finishes before wave 2
    # even generates, so no reading can represent the full batch.
    driver = _ChunkedFakeDriver(total_blocks=4096, block_size=BLOCK, fixed_overhead_bytes=1_000_000)
    with pytest.raises(ValueError, match="before prefill"):
        sample_decode_window(
            driver=driver,
            request_ids=["s-r0", "s-r1"],
            prompts=["x", "x"],
            prompt_lens=[64, 64],
            decode_tokens=1,
            prefill_chunks=2,
            theory_bpt=BPT,
            block_size=BLOCK,
        )


class _EosWaveFakeDriver(FakeEngineDriver):
    """Fake engine with chunked-prefill wave stagger AND early EOS.

    Request ``i`` of the current sample (insertion order) joins wave
    ``i * waves // batch`` and generates its first token on step
    ``wave``; earlier waves can finish (freeing their blocks) before
    later waves finish prefill — exactly what killed campaign
    kv-scaling-20260928i's grid at s2048 b16 (4 prefill chunks, EOS
    after ~4 tokens on the fixed prompts). ``eos_after`` is honored
    only for requests added with ``ignore_eos=False``.
    """

    def __init__(self, *args, waves: int, **kwargs):
        if waves < 1:
            raise ValueError(f"waves must be >= 1, got {waves}")
        super().__init__(*args, **kwargs)
        self._waves = waves
        self._pending: list[str] = []
        self._wave_of: dict[str, int] = {}
        self._steps = 0

    def add_request(self, request_id: str, *args, **kwargs):  # noqa: D102
        super().add_request(request_id, *args, **kwargs)
        self._pending.append(request_id)

    def _assign_waves(self) -> None:
        n = len(self._pending)
        for i, rid in enumerate(self._pending):
            self._wave_of[rid] = i * self._waves // n

    def step(self):  # noqa: D102
        if not self._wave_of:
            self._assign_waves()
        self._steps += 1
        outputs: list[StepOutput] = []
        for rid, req in self._requests.items():
            if req.finished:
                continue
            if self._wave_of[rid] > self._steps - 1:
                continue  # wave not prefilling yet: unscheduled
            req.generated += 1
            if req.generated >= self._finish_after(req):
                req.finished = True
                req.freed = True
            outputs.append(StepOutput(request_id=rid, new_tokens=1, finished=req.finished))
        self._sync_pool()
        return outputs

    def end_sample(self):  # noqa: D102
        self._pending = []
        self._wave_of = {}
        self._steps = 0
        super().end_sample()


def _eos_wave_driver(waves: int, eos_after: int) -> _EosWaveFakeDriver:
    return _EosWaveFakeDriver(
        total_blocks=4096,
        block_size=BLOCK,
        fixed_overhead_bytes=1_000_000,
        eos_after=eos_after,
        waves=waves,
    )


def test_fake_driver_eos_after_honors_ignore_eos():
    # eos_after=4: the request added with ignore_eos=False finishes at
    # 4 generated tokens; ignore_eos=True runs to max_tokens.
    driver = FakeEngineDriver(
        total_blocks=4096, block_size=BLOCK, fixed_overhead_bytes=1_000_000, eos_after=4
    )
    driver.add_request("eos", "p", max_tokens=8, temperature=0.0, prompt_len=64, ignore_eos=False)
    driver.add_request("kept", "p", max_tokens=8, temperature=0.0, prompt_len=64, ignore_eos=True)
    finish_step: dict[str, int] = {}
    for step_idx in range(1, 9):
        for out in driver.step():
            if out.finished and out.request_id not in finish_step:
                finish_step[out.request_id] = step_idx
    assert finish_step == {"eos": 4, "kept": 8}


def test_fake_driver_rejects_bad_eos_after():
    with pytest.raises(ValueError, match="eos_after"):
        FakeEngineDriver(
            total_blocks=4096, block_size=BLOCK, fixed_overhead_bytes=1_000_000, eos_after=0
        )


@pytest.mark.parametrize(
    ("waves", "eos_after"),
    [(2, 2), (4, 4)],
    ids=["2-wave", "4-wave"],
)
def test_staggered_eos_trips_guard_without_ignore_eos(waves, eos_after):
    # Without ignore_eos, the first wave emits EOS (freeing its blocks)
    # before the last wave's prefill completes, so the prefill reading
    # could never represent the full batch: the prefill-timing guard
    # must refuse with its loud error instead of measuring a partial
    # batch. This is the campaign-20260928i s2048 b16 failure,
    # reproduced on CPU. The guard trips exactly when eos_after <=
    # waves (early wave finishes before late wave prefills). Since the
    # guard-ordering fix it fires before the prefill assignment, not
    # as the decode-freeze message.
    driver = _eos_wave_driver(waves, eos_after)
    with pytest.raises(ValueError, match="before prefill"):
        sample_decode_window(
            driver=driver,
            request_ids=[f"r{k}" for k in range(4)],
            prompts=["x"] * 4,
            prompt_lens=[64] * 4,
            decode_tokens=5,
            ignore_eos=False,
            prefill_chunks=waves,
            theory_bpt=BPT,
            block_size=BLOCK,
        )


@pytest.mark.parametrize("waves", [2, 4], ids=["2-wave", "4-wave"])
def test_ignore_eos_reads_full_batch_through_stagger(waves):
    # ignore_eos=True is the sampler default: the early EOS is
    # disabled, so every request runs the full decode length and the
    # last-all-alive freeze reads the whole batch — 4 requests x 5
    # blocks at the freeze step. Live tokens at the freeze: 2-wave
    # freezes with per-request generated counts (4,4,3,3) -> 270;
    # 4-wave freezes with (4,3,2,1) -> 266.
    driver = _eos_wave_driver(waves, eos_after=waves)
    reading = sample_decode_window(
        driver=driver,
        request_ids=[f"r{k}" for k in range(4)],
        prompts=["x"] * 4,
        prompt_lens=[64] * 4,
        decode_tokens=5,
        prefill_chunks=waves,
        theory_bpt=BPT,
        block_size=BLOCK,
    )
    assert reading.used_blocks_decode == 20
    assert reading.live_tokens_decode == (270 if waves == 2 else 266)
    assert reading.kv_bytes_decode == 20 * BLOCK * BPT
    assert reading.used_blocks_prefill == 20


def test_run_cell_with_eos_waves_and_ignore_eos():
    # End to end through run_cell with 4 prefill waves and early EOS:
    # the sampler default (ignore_eos=True) keeps every request alive
    # for the full decode, so both samples read the whole batch
    # identically and the wave assignment resets per sample.
    ctx = _ctx(max_num_batched_tokens=64)  # 4x64 tokens -> 4 chunks
    driver = _eos_wave_driver(waves=4, eos_after=4)
    cell = run_cell(
        driver=driver,
        ctx=ctx,
        seq_len=64,
        batch_size=4,
        n_samples=2,
        prompts=["x"] * 4,
        log=None,
    )
    assert cell.kv_bytes_used_samples == [24 * BLOCK * BPT] * 2
    assert cell.total_tokens == 366  # 4x64 prompt + (29,28,27,26) generated at the freeze


# ---------- attempt-10 fixes: measured-spacing decode window + guard order ----------


class _WideWaveFakeDriver(FakeEngineDriver):
    """Fake engine with WIDE chunked-prefill wave spacing.

    Wave ``w`` of ``waves`` generates its first token on step
    ``1 + w * spacing`` (wave assignment is ``i * waves // n`` in
    insertion order, as in _EosWaveFakeDriver). With spacing wider
    than the decode window, wave 0 finishes (freeing its blocks)
    before the last wave's prefill completes — exactly what killed
    attempt 10's grid at s2048 b16 (4 waves, wave 1 finished 5 decode
    tokens before wave 4 finished prefill: spacing > 5 engine steps).
    """

    def __init__(self, *args, waves: int, spacing: int, **kwargs):
        if waves < 1:
            raise ValueError(f"waves must be >= 1, got {waves}")
        if spacing < 1:
            raise ValueError(f"spacing must be >= 1, got {spacing}")
        super().__init__(*args, **kwargs)
        self._waves = waves
        self._spacing = spacing
        self._pending: list[str] = []
        self._wave_of: dict[str, int] = {}
        self._steps = 0

    def add_request(self, request_id: str, *args, **kwargs):  # noqa: D102
        super().add_request(request_id, *args, **kwargs)
        self._pending.append(request_id)

    def _assign_waves(self) -> None:
        n = len(self._pending)
        for i, rid in enumerate(self._pending):
            self._wave_of[rid] = i * self._waves // n

    def _first_step(self, rid: str) -> int:
        return 1 + self._wave_of[rid] * self._spacing

    def step(self):  # noqa: D102
        if not self._wave_of:
            self._assign_waves()
        self._steps += 1
        outputs: list[StepOutput] = []
        for rid, req in self._requests.items():
            if req.finished:
                continue
            if self._steps < self._first_step(rid):
                continue  # wave not prefilling yet: unscheduled
            req.generated += 1
            if req.generated >= self._finish_after(req):
                req.finished = True
                req.freed = True
            outputs.append(StepOutput(request_id=rid, new_tokens=1, finished=req.finished))
        self._sync_pool()
        return outputs

    def end_sample(self):  # noqa: D102
        self._pending = []
        self._wave_of = {}
        self._steps = 0
        super().end_sample()


def _wide_wave_driver(waves: int, spacing: int) -> _WideWaveFakeDriver:
    return _WideWaveFakeDriver(
        total_blocks=4096,
        block_size=BLOCK,
        fixed_overhead_bytes=1_000_000,
        waves=waves,
        spacing=spacing,
    )


def test_attempt10_stagger_refuses_loud_with_prefill_timing_error():
    # Attempt-10 reproduction on CPU: 4 waves, 6-step spacing (wider
    # than the measured >5). Wave 0 finishes at step 5, before wave 3's
    # first token (step 19): the prefill reading could never represent
    # the full batch, so the prefill-timing guard must fire its loud
    # error — not the misleading decode-freeze message the old guard
    # order produced.
    driver = _wide_wave_driver(waves=4, spacing=6)
    with pytest.raises(ValueError, match="before prefill"):
        sample_decode_window(
            driver=driver,
            request_ids=[f"r{k}" for k in range(4)],
            prompts=["x"] * 4,
            prompt_lens=[64] * 4,
            decode_tokens=5,
            prefill_chunks=20,  # budget only; the stagger, not the budget, is the point
            theory_bpt=BPT,
            block_size=BLOCK,
        )


def test_run_cell_survives_wide_wave_stagger_with_extended_window():
    # The decode-extension fix, end to end: 2 waves, 6-step spacing.
    # The old (chunks + 1) window (3 tokens here) let wave 0 finish at
    # step 3, before wave 1's first token (step 7): no all-alive step,
    # the cell refuses. The measured-spacing window (30) keeps wave 0
    # alive until step 30; freeze at step 29 with generated (29, 23)
    # -> 12 blocks, 180 live tokens.
    ctx = _ctx(max_num_batched_tokens=64)  # 2x64 tokens -> 2 chunks
    driver = _wide_wave_driver(waves=2, spacing=6)
    cell = run_cell(
        driver=driver,
        ctx=ctx,
        seq_len=64,
        batch_size=2,
        n_samples=1,
        prompts=["x", "x"],
        log=None,
    )
    assert cell.kv_bytes_used_samples == [12 * BLOCK * BPT]
    assert cell.total_tokens == 180


def test_prefill_timing_guard_fires_before_prefill_assignment():
    # Guard-ordering pin: wave 0 finishes its 5 decode tokens during
    # step 5 — the same step wave 1 generates its first token — so at
    # the post-step every active request is scheduled AND a request
    # has finished. The prefill-timing guard must fire before the
    # prefill assignment; the old order assigned a partial-batch
    # prefill first and failed later with the misleading
    # decode-freeze message.
    driver = _wide_wave_driver(waves=2, spacing=4)
    with pytest.raises(ValueError, match="before prefill"):
        sample_decode_window(
            driver=driver,
            request_ids=["s-r0", "s-r1"],
            prompts=["x", "x"],
            prompt_lens=[64, 64],
            decode_tokens=5,
            prefill_chunks=2,
            theory_bpt=BPT,
            block_size=BLOCK,
        )


def test_build_exact_length_prompts_survives_tile_boundary_merge():
    # MergingTokenizer loses one token per tile boundary on re-encode
    # (the Qwen2.5-0.5B failure that killed campaign e's smoke). Force the
    # truncation past a tile boundary so the old naive tiling would fail.
    tok = MergingTokenizer()
    base_ids = tok.encode("The quick brown fox jumps over the lazy dog. " * 64)
    base_len = len(base_ids)
    assert base_len > 0
    tiled_true = tok.encode(tok.decode(base_ids + base_ids))
    assert len(tiled_true) == 2 * base_len - 1  # one merge at the boundary
    prompt = build_exact_length_prompts(tok, base_len + 37)
    assert len(tok.encode(prompt)) == base_len + 37


# ---------- regression + knee ----------


def test_fit_recovers_known_slope():
    xs = [100.0, 200.0, 400.0, 800.0]
    ys = [BPT * x for x in xs]
    fit = fit_bytes_per_token(xs, ys)
    assert fit.slope_bpt == pytest.approx(BPT)
    assert fit.r2 == pytest.approx(1.0)
    assert fit.intercept_bytes == pytest.approx(0.0)


def test_fit_rejects_degenerate_inputs():
    with pytest.raises(ValueError):
        fit_bytes_per_token([1.0], [2.0])
    with pytest.raises(ValueError):
        fit_bytes_per_token([3.0, 3.0], [1.0, 2.0])
    with pytest.raises(ValueError):
        fit_bytes_per_token([1.0, 2.0], [1.0])


def _cell(total_tokens: int, p50: float, batch: int = 1) -> KvScalingCell:
    return KvScalingCell(
        model_id="m",
        gpu="g",
        vllm_version="v",
        seq_len=64,
        batch_size=batch,
        n_samples=2,
        kv_bytes_used_samples=[int(p50), int(p50)],
        total_tokens=total_tokens,
        baseline_bytes=1_000_000,
        method="test",
    )


def test_find_knee():
    cells = [
        _cell(100, 100_000.0),  # share ~0.09
        _cell(200, 900_000.0),  # share ~0.47
        _cell(400, 1_200_000.0),  # share ~0.545 -> knee
    ]
    knee = find_knee(cells, 1_000_000)
    assert knee.total_tokens == 400
    assert knee.kv_share == pytest.approx(1_200_000 / 2_200_000)
    assert knee.batch_size == 1


def test_find_knee_raises_when_never_reached():
    with pytest.raises(ValueError, match="never reached"):
        find_knee([_cell(100, 10_000.0)], 1_000_000)
    with pytest.raises(ValueError):
        find_knee([], 1_000_000)


# ---------- theory gate ----------


def test_validate_against_theory_passes_within_tol():
    assert validate_against_theory(12000.0, BPT) == pytest.approx(abs(12000.0 - BPT) / BPT)


def test_validate_against_theory_boundary():
    assert validate_against_theory(BPT * 1.15, BPT, tol=0.15) == pytest.approx(0.15)


def test_validate_against_theory_raises_loudly():
    with pytest.raises(ValueError, match="refusing to publish"):
        validate_against_theory(100.0, BPT)
    with pytest.raises(ValueError):
        validate_against_theory(12000.0, 0)
    with pytest.raises(ValueError):
        validate_against_theory(-1.0, BPT)
    with pytest.raises(ValueError):
        validate_against_theory(12000.0, BPT, tol=-0.1)


# ---------- rung-0 gate ----------


def test_check_dry_run_gate_passes():
    reg = RegressionFit(slope_bpt=BPT * 1.005, intercept_bytes=0.0, r2=0.999, residual_se=1.0)
    check_dry_run_gate(reg, BPT, BLOCK, 512)  # no raise


def test_check_dry_run_gate_rejects_bad_slope():
    reg = RegressionFit(slope_bpt=BPT * 1.5, intercept_bytes=0.0, r2=0.999, residual_se=1.0)
    with pytest.raises(ValueError, match="rung-0 gate failed"):
        check_dry_run_gate(reg, BPT, BLOCK, 512)


def test_check_dry_run_gate_rejects_low_r2():
    reg = RegressionFit(slope_bpt=BPT * 1.005, intercept_bytes=0.0, r2=0.90, residual_se=1.0)
    with pytest.raises(ValueError, match="R\\^2"):
        check_dry_run_gate(reg, BPT, BLOCK, 512)


def test_full_dry_run_pipeline_recovers_injected_slope(tmp_path: Path):
    out = run_dry_run_campaign(
        campaign_id="rung0-test",
        out_dir=tmp_path,
        seq_lens=[64, 128, 256],
        batch_sizes=[1, 2],
        n_samples=3,
        theory_bpt=BPT,
        log=None,
    )
    payload = json.loads((out / "cells.json").read_text())
    assert payload["schema_version"] == "1.1"
    campaign = payload["campaign"]
    slope = campaign["regression"]["slope_bpt"]
    assert BPT * 0.999 <= slope <= BPT * (1 + BLOCK / 64)
    assert campaign["regression"]["r2"] >= 0.98
    assert campaign["theory_bytes_per_token"] == BPT
    assert (out / "cells.csv").exists()
    assert (out / "summary.json").exists()
    # every cell's decode samples are KV bytes, identical across samples
    for cell in campaign["cells"]:
        assert len(set(cell["kv_bytes_used_samples"])) == 1
        assert cell["measured_bytes_per_token"] == pytest.approx(
            cell["p50_bytes"] / cell["total_tokens"]
        )


# ---------- schema ----------


def test_schema_rejects_invalid_cells():
    good = {
        "model_id": "m",
        "gpu": "g",
        "vllm_version": "v",
        "seq_len": 64,
        "batch_size": 2,
        "n_samples": 2,
        "kv_bytes_used_samples": [100, 120],
        "total_tokens": 134,
        "baseline_bytes": 1000,
        "method": "t",
    }
    KvScalingCell(**good)  # no raise
    with pytest.raises(ValueError):
        KvScalingCell(**{**good, "seq_len": 0})
    with pytest.raises(ValueError):
        KvScalingCell(**{**good, "kv_bytes_used_samples": []})
    with pytest.raises(ValueError):
        KvScalingCell(**{**good, "n_samples": 3})
    with pytest.raises(ValueError):
        KvScalingCell(**{**good, "extra_field": 1})
    with pytest.raises(ValueError):
        KvScalingCell(
            **{**good, "kv_bytes_prefill_samples": [100]}  # wrong length
        )


def test_schema_percentiles_and_measured_column():
    cell = KvScalingCell(
        model_id="m",
        gpu="g",
        vllm_version="v",
        seq_len=64,
        batch_size=1,
        n_samples=4,
        kv_bytes_used_samples=[100, 200, 300, 400],
        total_tokens=67,
        baseline_bytes=0,
        method="t",
    )
    assert cell.p50_bytes == pytest.approx(250.0)
    assert cell.p95_bytes == pytest.approx(385.0)
    assert cell.measured_bytes_per_token == pytest.approx(250.0 / 67)


# ---------- results writer ----------


def _campaign() -> KvScalingCampaign:
    cells = [
        KvScalingCell(
            model_id="m",
            gpu="g",
            vllm_version="v",
            seq_len=64,
            batch_size=1,
            n_samples=2,
            kv_bytes_used_samples=[5 * BLOCK * BPT] * 2,
            kv_bytes_prefill_samples=[5 * BLOCK * BPT] * 2,
            total_tokens=67,
            baseline_bytes=1_000_000,
            method="t",
        ),
        KvScalingCell(
            model_id="m",
            gpu="g",
            vllm_version="v",
            seq_len=128,
            batch_size=2,
            n_samples=2,
            kv_bytes_used_samples=[18 * BLOCK * BPT] * 2,
            total_tokens=262,
            baseline_bytes=1_000_000,
            method="t",
        ),
    ]
    reg = fit_bytes_per_token([float(c.total_tokens) for c in cells], [c.p50_bytes for c in cells])
    return KvScalingCampaign(
        campaign_id="cid",
        model_revision="rev1",
        cells=cells,
        theory_bytes_per_token=BPT,
        regression=reg,
        knee=find_knee(cells, 1_000_000),
    )


def test_write_campaign_csv_round_trip(tmp_path: Path):
    campaign = _campaign()
    csv_path, json_path = write_campaign(campaign, results_root=tmp_path, campaign_id="cid")
    assert SCHEMA_VERSION == "1.1"
    with csv_path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 2
    row = rows[0]
    assert row["schema_version"] == "1.1"
    assert row["model_revision"] == "rev1"
    assert row["theory_bytes_per_token"] == str(BPT)
    assert float(row["measured_bytes_per_token"]) == pytest.approx(
        float(row["p50_bytes"]) / float(row["total_tokens"])
    )
    assert row["sample_0"] == str(5 * BLOCK * BPT)
    assert row["sample_1"] == str(5 * BLOCK * BPT)
    payload = json.loads(json_path.read_text())
    assert payload["schema_version"] == "1.1"
    assert len(payload["campaign"]["cells"]) == 2
    assert payload["campaign"]["cells"][0]["kv_bytes_used_samples"] == [5 * BLOCK * BPT] * 2


def test_write_summary_json_contract(tmp_path: Path):
    campaign = _campaign()
    path = write_summary_json(tmp_path, campaign)
    payload = json.loads(path.read_text())
    for key in (
        "campaign_id",
        "measured_date",
        "slope_bytes_per_token",
        "intercept_bytes",
        "r2",
        "knee_total_tokens_by_batch",
        "theory_bytes_per_token",
        "model_revision",
    ):
        assert key in payload, key
    assert payload["slope_bytes_per_token"] == pytest.approx(campaign.regression.slope_bpt)


def test_write_preflight_json(tmp_path: Path):
    preflight = build_cost_preflight(
        campaign_id="c",
        model_id="m",
        model_revision="r",
        seq_lens=[64],
        batch_sizes=[1],
        n_samples=2,
        secs_per_cell=60.0,
        usd_per_hour=0.5,
        out_dir=tmp_path / "c",
        approved=False,
    )
    assert preflight.n_cells == 1
    assert preflight.est_wall_seconds == pytest.approx(60.0)
    assert preflight.est_cost_usd == pytest.approx(60.0 / 3600 * 0.5)
    path = write_preflight_json(tmp_path, preflight.as_dict())
    assert json.loads(path.read_text())["n_cells"] == 1


def test_print_cost_preflight(capsys):
    preflight = build_cost_preflight(
        campaign_id="c",
        model_id="m",
        model_revision="r",
        seq_lens=[64, 128],
        batch_sizes=[1, 2],
        n_samples=5,
        secs_per_cell=120.0,
        usd_per_hour=None,
        out_dir=Path("out"),
        approved=True,
    )
    print_cost_preflight(preflight)
    out = capsys.readouterr().out
    assert "pending cells: 4" in out
    assert "unknown" in out  # no rate given


def test_per_cell_json_resume(tmp_path: Path):
    campaign = _campaign()
    cell = campaign.cells[0]
    path = tmp_path / "cell_s64_b1.json"
    write_cell_json(path, cell)
    loaded = read_cell_json(path, seq_len=64, batch_size=1, n_samples=2, model_id="m")
    assert loaded is not None
    assert loaded.kv_bytes_used_samples == cell.kv_bytes_used_samples
    # missing file -> None
    assert (
        read_cell_json(
            tmp_path / "nope.json",
            seq_len=64,
            batch_size=1,
            n_samples=2,
            model_id="m",
        )
        is None
    )
    # grid mismatch -> None (never trust blindly)
    assert read_cell_json(path, seq_len=128, batch_size=1, n_samples=2, model_id="m") is None
    # corrupt JSON -> None
    bad = tmp_path / "bad.json"
    bad.write_text("{not json")
    assert read_cell_json(bad, seq_len=64, batch_size=1, n_samples=2, model_id="m") is None


# ---------- sibling plotter contract ----------


def test_plotter_loads_dry_run_artifacts(tmp_path: Path):
    """The plotter's loaders must accept what the writer produces."""
    from bench.kv_scaling.figures import (
        load_campaign,
        load_campaign_from_json,
    )

    out = run_dry_run_campaign(
        campaign_id="contract-test",
        out_dir=tmp_path,
        seq_lens=[64, 128],
        batch_sizes=[1, 2],
        n_samples=2,
        theory_bpt=BPT,
        log=None,
    )
    kv_campaign, _check = load_campaign_from_json(out / "cells.json")
    assert len(kv_campaign.cells) == 4
    assert all(c.measured_bytes_per_token is not None for c in kv_campaign.cells)
    assert kv_campaign.summary.slope_bytes_per_token > 0
    legacy = load_campaign(out / "cells.csv", out / "summary.json")
    assert len(legacy.cells) == 4
    assert all(c.measured_bytes_per_token is not None for c in legacy.cells)


# ---------- CLI ----------


def test_cli_dry_run_passes(tmp_path: Path):
    code = main(
        [
            "--dry-run",
            "--seq-lens",
            "64,128",
            "--batches",
            "1,2",
            "--n-samples",
            "2",
            "--out",
            str(tmp_path),
            "--campaign-id",
            "cli-dry",
        ]
    )
    assert code == 0
    assert (tmp_path / "cli-dry" / "cells.json").exists()


def test_cli_refuses_multi_cell_without_approval(tmp_path: Path, capsys):
    code = main(
        [
            "--seq-lens",
            "64,128",
            "--batches",
            "1,2",
            "--n-samples",
            "2",
            "--out",
            str(tmp_path),
        ]
    )
    assert code == 2
    err = capsys.readouterr().err
    assert "approve-cost" in err


def test_cli_approval_gets_past_cost_gate_to_cuda_gate(tmp_path: Path, capsys):
    # Approved but no GPU on this box: must fail at the CUDA gate, not
    # the cost gate, proving approval was accepted.
    code = main(
        [
            "--seq-lens",
            "64,128",
            "--batches",
            "1,2",
            "--n-samples",
            "2",
            "--out",
            str(tmp_path),
            "--approve-cost",
        ]
    )
    assert code == 2
    assert "CUDA GPU" in capsys.readouterr().err


def test_cli_smoke_cell_exempt_from_approval(tmp_path: Path, capsys):
    # Single cell without --approve-cost: skips the cost refusal, then
    # fails at the CUDA gate on this CPU-only box.
    code = main(
        [
            "--seq-lens",
            "64",
            "--batches",
            "1",
            "--n-samples",
            "2",
            "--out",
            str(tmp_path),
        ]
    )
    assert code == 2
    assert "CUDA GPU" in capsys.readouterr().err


def test_cli_parser_flags():
    args = build_parser().parse_args(
        [
            "--model",
            "a/b",
            "--revision",
            "deadbeef",
            "--seq-lens",
            "1024,2048",
            "--batches",
            "1,4",
            "--n-samples",
            "3",
            "--max-model-len",
            "4096",
            "--no-enforce-eager",
            "--out",
            "/tmp/x",
            "--approve-cost",
            "--campaign-id",
            "c1",
            "--dry-run",
            "--usd-per-hour",
            "0.5",
            "--block-size",
            "32",
            "--decode-tokens",
            "8",
        ]
    )
    assert args.model == "a/b"
    assert args.model_id is None
    assert args.revision == "deadbeef"
    assert args.seq_lens == [1024, 2048]
    assert args.batches == [1, 4]
    assert args.enforce_eager is False
    assert args.dry_run is True
    assert args.usd_per_hour == 0.5
    assert args.block_size == 32
    assert args.decode_tokens == 8


def test_cli_model_id_alias_wins():
    args = build_parser().parse_args(["--model", "a/b", "--model-id", "c/d"])
    assert args.model_id == "c/d"


def test_run_vllm_sweep_dry_run_flag(tmp_path: Path):
    out = run_vllm_sweep(
        dry_run=True,
        campaign_id="lib-dry",
        out_dir=tmp_path,
        seq_lens=[64, 128],
        batch_sizes=[1, 2],
        n_samples=2,
    )
    assert (out / "cells.json").exists()


def test_dry_run_rejects_single_cell_grid(tmp_path: Path):
    with pytest.raises(ValueError, match="at least 2 grid cells"):
        run_dry_run_campaign(
            campaign_id="one-cell",
            out_dir=tmp_path,
            seq_lens=[64],
            batch_sizes=[1],
            n_samples=2,
        )


def test_knee_point_schema():
    knee = KneePoint(total_tokens=400, kv_share=0.6, batch_size=2)
    assert knee.total_tokens == 400
    with pytest.raises(ValueError):
        KneePoint(total_tokens=0, kv_share=0.6)


def test_cuda_gate_never_touches_torch_cuda_runtime(monkeypatch):
    """Fork-poisoning regression test (campaign kv-scaling-20260928).

    The pre-engine CUDA gate must not call any torch.cuda runtime API:
    torch.cuda.is_available() runs cuInit in the parent, which makes vLLM
    v1's forked EngineCore die with "Cannot re-initialize CUDA in forked
    subprocess" (one dead smoke cell, ~$1.18 of idle pod burn). A hostile
    fake torch explodes on any cuda attribute access; the gate must still
    answer from nvidia-smi alone.
    """
    import subprocess
    import sys
    import types

    from bench.kv_scaling.measure import cuda_available

    touched: list[str] = []

    fake_cuda = types.ModuleType("torch.cuda")

    def _fake_getattr(name: str):
        def _boom(*args, **kwargs):
            touched.append(name)
            raise AssertionError(f"torch.cuda.{name} must not be called before engine creation")

        return _boom

    fake_cuda.__getattr__ = _fake_getattr  # type: ignore[attr-defined]
    fake_torch = types.ModuleType("torch")
    fake_torch.cuda = fake_cuda  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "torch.cuda", fake_cuda)

    class _Proc:
        returncode = 0
        stdout = "GPU 0: NVIDIA GeForce RTX 4090 (UUID: GPU-1234abcd)\n"
        stderr = ""

    monkeypatch.setattr(subprocess, "run", lambda *a, **k: _Proc())

    assert cuda_available() is True
    assert touched == []


def test_cuda_gate_false_without_nvidia_smi(monkeypatch):
    import subprocess

    from bench.kv_scaling.measure import cuda_available

    def _missing(*args, **kwargs):
        raise FileNotFoundError("nvidia-smi")

    monkeypatch.setattr(subprocess, "run", _missing)
    assert cuda_available() is False


def test_cuda_gate_false_when_no_gpu_listed(monkeypatch):
    import subprocess

    from bench.kv_scaling.measure import cuda_available

    class _Proc:
        returncode = 0
        stdout = ""
        stderr = ""

    monkeypatch.setattr(subprocess, "run", lambda *a, **k: _Proc())
    assert cuda_available() is False


def test_resolve_theory_uses_engine_runtime_dtype_when_torch_dtype_dropped():
    """vLLM 0.11.2's hf_text_config.to_dict() drops torch_dtype, and
    transformers' own to_dict() proved equally lossy on the pod.

    Campaign kv-scaling-20260928b died on the engine dict; campaign
    kv-scaling-20260928c died on the old AutoConfig fallback. The fix:
    the element width comes from the engine's authoritative runtime
    dtype (vLLM resolves model_config.dtype before the model loads).
    A fake engine whose config dict lacks torch_dtype but whose
    model_config.dtype is bf16 must yield 12,288 B/token.
    """
    from bench.kv_scaling import measure

    class _FakeDtype:
        # mirrors str(torch.bfloat16) == "torch.bfloat16"
        def __str__(self):
            return "torch.bfloat16"

    class _FakeTextConfig:
        def to_dict(self):
            return {
                "num_hidden_layers": 24,
                "num_attention_heads": 14,
                "num_key_value_heads": 2,
                "hidden_size": 896,
                # no torch_dtype: what vLLM 0.11.2's wrapped config serves
            }

    class _FakeModelConfig:
        hf_text_config = _FakeTextConfig()
        dtype = _FakeDtype()

    class _FakeEngine:
        model_config = _FakeModelConfig()

    class _FakeLlm:
        llm_engine = _FakeEngine()

    theory, source = measure._resolve_theory(_FakeLlm(), "Qwen/Qwen2.5-0.5B-Instruct", "rev123")
    assert source == "engine:model_config.hf_text_config+runtime-dtype"
    assert theory.bytes_per_token == 12_288


def test_resolve_theory_falls_back_to_raw_hub_config(monkeypatch):
    """When no engine config path yields a dict, _resolve_theory reads
    the raw config.json at the pinned revision: plain JSON, no
    transformers to_dict() lossiness."""
    from bench.kv_scaling import measure

    class _FakeLlm:
        llm_engine = object()  # no config attributes at all

    def _fake_raw(model_id, revision):
        assert model_id == "Qwen/Qwen2.5-0.5B-Instruct"
        assert revision == "rev123"
        return {
            "num_hidden_layers": 24,
            "num_attention_heads": 14,
            "num_key_value_heads": 2,
            "hidden_size": 896,
            "torch_dtype": "bfloat16",
        }

    monkeypatch.setattr(measure, "_raw_config_json", _fake_raw)
    theory, source = measure._resolve_theory(_FakeLlm(), "Qwen/Qwen2.5-0.5B-Instruct", "rev123")
    assert source == "hub:config.json"
    assert theory.bytes_per_token == 12_288


def test_probe_block_pool_finds_inproc_engine_core_pool():
    """Campaign kv-scaling-20260928d died because the probe only knew
    the multiprocess layout (engine_core.scheduler...), but with the
    default VLLM_ENABLE_V1_MULTIPROCESSING=1, engine_core is a
    SyncMPClient whose scheduler lives across the subprocess boundary.
    In-process (env var =0), the layout is
    engine_core.engine_core.scheduler.kv_cache_manager.block_pool;
    the probe must resolve it."""
    from bench.kv_scaling import measure

    class _FakePool:
        num_gpu_blocks = 1024

        def get_num_free_blocks(self):
            return 700

    class _FakeKVCacheManager:
        block_pool = _FakePool()

    class _FakeScheduler:
        kv_cache_manager = _FakeKVCacheManager()

    class _FakeEngineCore:
        scheduler = _FakeScheduler()

    class _FakeInprocClient:
        engine_core = _FakeEngineCore()

    class _FakeLLMEngine:
        engine_core = _FakeInprocClient()

    pool = measure._probe_block_pool(_FakeLLMEngine(), log=lambda *a: None)
    assert pool.num_gpu_blocks == 1024
    assert pool.get_num_free_blocks() == 700
    assert pool.source == "engine_core.engine_core.scheduler.kv_cache_manager.block_pool"


def test_probe_block_pool_refuses_unreachable_multiprocess_scheduler():
    """Documents the kv-scaling-20260928d failure: a SyncMPClient-shaped
    engine (scheduler unreachable across the subprocess boundary) must
    fail loudly with the attempted paths, never silently read a wrong
    pool."""
    from bench.kv_scaling import measure

    class _FakeSyncMPClient:
        # no scheduler, no engine_core: nothing reachable from the parent
        pass

    class _FakeLLMEngine:
        engine_core = _FakeSyncMPClient()

    with pytest.raises(RuntimeError, match="could not find a readable GPU block pool"):
        measure._probe_block_pool(_FakeLLMEngine(), log=lambda *a: None)
