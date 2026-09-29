"""Pure KV-cache theory: bytes-per-token math from model architecture.

Method: for each token, each transformer layer stores one key vector and
one value vector per KV head, each of ``head_dim`` elements. So

    bytes_per_token = 2 * n_layers * n_kv_heads * head_dim * bytes_per_element

The factor of 2 is for keys plus values. With grouped-query attention
(GQA) the KV head count is smaller than the query head count; the
formula must use the KV head count (``num_key_value_heads``), never the
query head count.

Limits: this counts KV-cache payload only. It excludes the vLLM paged
allocator's block-size rounding, activation memory, and weight memory.
Treat it as a lower bound on per-token allocator growth, not as a
prediction of total GPU memory.
"""

from __future__ import annotations

from dataclasses import dataclass

# ---------- pure math ----------


def kv_bytes_per_token(
    n_layers: int,
    n_kv_heads: int,
    head_dim: int,
    bytes_per_element: int,
) -> int:
    """Return exact KV-cache bytes stored per token.

    Every input must be a positive integer. Raises ValueError otherwise.
    """
    for name, value in (
        ("n_layers", n_layers),
        ("n_kv_heads", n_kv_heads),
        ("head_dim", head_dim),
        ("bytes_per_element", bytes_per_element),
    ):
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"{name} must be a positive integer, got {value!r}")
    # 2x: one key vector and one value vector per head per layer.
    return 2 * n_layers * n_kv_heads * head_dim * bytes_per_element


def head_dim_from_config(hidden_size: int, num_attention_heads: int) -> int:
    """Return per-head dimension, raising if hidden_size is not divisible."""
    if num_attention_heads <= 0:
        raise ValueError(f"num_attention_heads must be positive, got {num_attention_heads}")
    if hidden_size % num_attention_heads != 0:
        raise ValueError(
            f"hidden_size {hidden_size} not divisible by num_attention_heads {num_attention_heads}"
        )
    return hidden_size // num_attention_heads


def bytes_per_element_from_dtype(dtype: str) -> int:
    """Map a torch/HF dtype name to bytes per element.

    Accepts common spellings like ``torch.float16``, ``float16``,
    ``fp16``, ``bfloat16``, ``float32``, and ``float8_e4m3fn``.
    Raises ValueError for unknown dtypes rather than guessing.
    """
    key = dtype.strip().lower().replace("torch.", "")
    table = {
        "float32": 4,
        "fp32": 4,
        "float16": 2,
        "fp16": 2,
        "bfloat16": 2,
        "bf16": 2,
        "float8_e4m3fn": 1,
        "float8_e5m2": 1,
        "fp8": 1,
    }
    if key not in table:
        raise ValueError(f"unknown dtype {dtype!r}; cannot infer bytes per element")
    return table[key]


# ---------- HF config parsing ----------


@dataclass(frozen=True)
class HfKvTheory:
    """Theory numbers parsed from one HuggingFace model config dict."""

    n_layers: int
    n_kv_heads: int
    head_dim: int
    bytes_per_element: int
    bytes_per_token: int
    revision: str = ""


def _config_int(config: dict[str, object], key: str) -> int:
    """Read a required integer field.

    Raises KeyError when the key is missing and ValueError when the
    value is not an integer, instead of letting a confusing TypeError
    escape.
    """
    value = config[key]  # KeyError when missing
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{key} must be an integer, got {value!r}")
    return value


def config_from_hf_config(config: dict[str, object], revision: str = "") -> HfKvTheory:
    """Parse a HF ``config.json`` dict into KV theory numbers.

    Uses ``num_key_value_heads`` for GQA models and falls back to
    ``num_attention_heads`` when the key is absent (older MHA configs).
    ``head_dim`` is ALWAYS derived as ``hidden_size // num_attention_heads``
    at runtime, never taken from a ``head_dim`` key and never hardcoded:
    the campaign theory definition (TRD section 2) fixes d_head this way,
    and an explicit head_dim in the wild (e.g. 128 on Qwen3-0.6B) would
    silently double the estimate. Any ``head_dim`` key present in the
    config is deliberately ignored.
    The ``revision`` string is recorded verbatim so the campaign schema
    can pin exactly which checkpoint the theory came from.
    Raises KeyError for missing required keys and ValueError for bad values.
    """
    n_layers = _config_int(config, "num_hidden_layers")
    n_query_heads = _config_int(config, "num_attention_heads")
    # GQA-aware: prefer the KV head count; MHA configs omit the key.
    if "num_key_value_heads" in config:
        n_kv_heads = _config_int(config, "num_key_value_heads")
    else:
        n_kv_heads = n_query_heads
    head_dim = head_dim_from_config(_config_int(config, "hidden_size"), n_query_heads)
    bpe = bytes_per_element_from_dtype(str(config["torch_dtype"]))
    return HfKvTheory(
        n_layers=n_layers,
        n_kv_heads=n_kv_heads,
        head_dim=head_dim,
        bytes_per_element=bpe,
        bytes_per_token=kv_bytes_per_token(n_layers, n_kv_heads, head_dim, bpe),
        revision=revision,
    )


__all__ = [
    "HfKvTheory",
    "bytes_per_element_from_dtype",
    "config_from_hf_config",
    "head_dim_from_config",
    "kv_bytes_per_token",
]
