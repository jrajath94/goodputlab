"""Tests for bench/kv_scaling/theory.py — pure KV-cache byte math."""

from __future__ import annotations

import pytest

from bench.kv_scaling.theory import (
    bytes_per_element_from_dtype,
    config_from_hf_config,
    head_dim_from_config,
    kv_bytes_per_token,
)

# ---------- kv_bytes_per_token ----------


def test_kv_bytes_per_token_qwen2_5_0_5b() -> None:
    """Qwen2.5-0.5B: 24 layers, 2 KV heads, head_dim 64 (= 896/14), fp16.

    Gives 12,288 B/token. An early draft used head_dim 128 (24,576);
    that was wrong: the real config has hidden_size 896 and 14 query
    heads, so d_head is 64 (TRD section 2).
    """
    assert kv_bytes_per_token(24, 2, 64, 2) == 12288


def test_kv_bytes_per_token_llama3_2_1b() -> None:
    """Llama-3.2-1B: 16 layers, 8 KV heads, head_dim 64, fp16 → 32,768 B/token."""
    assert kv_bytes_per_token(16, 8, 64, 2) == 32768


def test_kv_bytes_per_token_formula() -> None:
    """Result equals 2 * layers * kv_heads * head_dim * bytes_per_element."""
    n_layers, n_kv_heads, head_dim, bpe = 32, 8, 128, 2
    assert kv_bytes_per_token(n_layers, n_kv_heads, head_dim, bpe) == (
        2 * n_layers * n_kv_heads * head_dim * bpe
    )


def test_kv_bytes_per_token_rejects_nonpositive() -> None:
    with pytest.raises(ValueError):
        kv_bytes_per_token(0, 2, 128, 2)
    with pytest.raises(ValueError):
        kv_bytes_per_token(24, 0, 128, 2)
    with pytest.raises(ValueError):
        kv_bytes_per_token(24, 2, 0, 2)
    with pytest.raises(ValueError):
        kv_bytes_per_token(24, 2, 128, 0)


# ---------- head_dim_from_config ----------


def test_head_dim_from_config() -> None:
    assert head_dim_from_config(hidden_size=4096, num_attention_heads=32) == 128


def test_head_dim_from_config_rejects_nondivisible() -> None:
    with pytest.raises(ValueError):
        head_dim_from_config(hidden_size=100, num_attention_heads=32)


# ---------- bytes_per_element_from_dtype ----------


def test_bytes_per_element_from_dtype_fp16() -> None:
    assert bytes_per_element_from_dtype("torch.float16") == 2
    assert bytes_per_element_from_dtype("fp16") == 2


def test_bytes_per_element_from_dtype_bf16() -> None:
    assert bytes_per_element_from_dtype("bfloat16") == 2


def test_bytes_per_element_from_dtype_fp32() -> None:
    assert bytes_per_element_from_dtype("float32") == 4
    assert bytes_per_element_from_dtype("torch.float32") == 4


def test_bytes_per_element_from_dtype_fp8() -> None:
    assert bytes_per_element_from_dtype("float8_e4m3fn") == 1


def test_bytes_per_element_from_dtype_unknown_raises() -> None:
    with pytest.raises(ValueError):
        bytes_per_element_from_dtype("int4")


# ---------- config_from_hf_config ----------


def test_config_from_hf_config_qwen2_5_0_5b() -> None:
    """Qwen2.5-0.5B config: GQA with 14 query heads and 2 KV heads."""
    cfg = config_from_hf_config(
        {
            "num_hidden_layers": 24,
            "num_attention_heads": 14,
            "num_key_value_heads": 2,
            "hidden_size": 896,
            "torch_dtype": "bfloat16",
        }
    )
    assert cfg.n_layers == 24
    assert cfg.n_kv_heads == 2
    assert cfg.head_dim == 64  # 896 / 14
    assert cfg.bytes_per_element == 2
    assert cfg.bytes_per_token == 2 * 24 * 2 * 64 * 2 == 12288


def test_config_from_hf_config_gqa_trap() -> None:
    """32 query heads / 8 KV heads must use the 8 KV heads, not 32."""
    cfg = config_from_hf_config(
        {
            "num_hidden_layers": 32,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "hidden_size": 4096,
            "torch_dtype": "float16",
        }
    )
    assert cfg.n_kv_heads == 8
    assert cfg.bytes_per_token == 2 * 32 * 8 * 128 * 2


def test_config_from_hf_config_falls_back_to_attention_heads() -> None:
    """Older configs without num_key_value_heads are MHA: KV heads = Q heads."""
    cfg = config_from_hf_config(
        {
            "num_hidden_layers": 12,
            "num_attention_heads": 12,
            "hidden_size": 768,
            "torch_dtype": "float32",
        }
    )
    assert cfg.n_kv_heads == 12
    assert cfg.head_dim == 64
    assert cfg.bytes_per_token == 2 * 12 * 12 * 64 * 4


def test_config_from_hf_config_qwen3_0_6b() -> None:
    """Qwen3-0.6B: 28 layers, 16 query heads, 8 KV heads, fp16.

    The campaign rule (TRD section 2): head_dim is ALWAYS derived as
    hidden_size // num_attention_heads (1024 // 16 = 64), even though
    a head_dim of 128 appears in the wild for this model. Using 128
    would silently double the estimate to 114,688. Correct theory:
    2 * 28 * 8 * 64 * 2 = 57,344 B/token, verified by independent
    arithmetic below.
    """
    assert 2 * 28 * 8 * 64 * 2 == 57344  # independent check of the vector
    cfg = config_from_hf_config(
        {
            "num_hidden_layers": 28,
            "num_attention_heads": 16,
            "num_key_value_heads": 8,
            "hidden_size": 1024,
            "head_dim": 128,  # present in the wild; the parser must ignore it
            "torch_dtype": "float16",
        },
        revision="deadbeef1234",
    )
    assert cfg.n_layers == 28
    assert cfg.n_kv_heads == 8
    assert cfg.head_dim == 64  # derived 1024 // 16; the 128 key is ignored
    assert cfg.bytes_per_element == 2
    assert cfg.bytes_per_token == 57344
    assert cfg.revision == "deadbeef1234"


def test_config_from_hf_config_ignores_explicit_head_dim() -> None:
    """The rule: never trust a head_dim key; always derive from the config."""
    cfg = config_from_hf_config(
        {
            "num_hidden_layers": 2,
            "num_attention_heads": 16,
            "num_key_value_heads": 8,
            "hidden_size": 1024,
            "head_dim": 128,
            "torch_dtype": "float16",
        }
    )
    assert cfg.head_dim == 64
    assert cfg.bytes_per_token == 2 * 2 * 8 * 64 * 2


def test_config_from_hf_config_revision_defaults_empty() -> None:
    cfg = config_from_hf_config(
        {
            "num_hidden_layers": 12,
            "num_attention_heads": 12,
            "hidden_size": 768,
            "torch_dtype": "float32",
        }
    )
    assert cfg.revision == ""


def test_head_dim_from_config_rejects_nonpositive_heads() -> None:
    with pytest.raises(ValueError):
        head_dim_from_config(hidden_size=1024, num_attention_heads=0)


def test_config_from_hf_config_rejects_non_integer_field() -> None:
    with pytest.raises(ValueError):
        config_from_hf_config(
            {
                "num_hidden_layers": "twenty-four",
                "num_attention_heads": 14,
                "hidden_size": 896,
                "torch_dtype": "float16",
            }
        )


def test_config_from_hf_config_missing_key_raises() -> None:
    with pytest.raises(KeyError):
        config_from_hf_config({"num_hidden_layers": 12})
