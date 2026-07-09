"""Deterministic synthetic-text builder for load generation.

Real corpora are out of scope for GoodputLab — the load generator
only needs prompts that *look* like English and have controllable
token budgets.  This module builds paragraphs and documents by
shuffling a small inline word list, with all randomness routed
through a caller-provided ``random.Random`` so two calls with the
same RNG state produce byte-identical text (LOAD-07 contract).
"""

from __future__ import annotations

import random

# Inline word list (~200 generic English words).  Deliberately
# boring — the point is the *shape* of the prompt, not the content.
_WORDS: tuple[str, ...] = (
    "the", "of", "and", "to", "in", "that", "have", "it", "for", "not",
    "on", "with", "he", "as", "you", "do", "at", "this", "but", "his",
    "by", "from", "they", "we", "say", "her", "she", "or", "an", "will",
    "my", "one", "all", "would", "there", "their", "what", "so", "up",
    "out", "if", "about", "who", "get", "which", "go", "me", "when",
    "make", "can", "like", "time", "no", "just", "him", "know", "take",
    "people", "into", "year", "your", "good", "some", "could", "them",
    "see", "other", "than", "then", "now", "look", "only", "come", "its",
    "over", "think", "also", "back", "after", "use", "two", "how", "our",
    "work", "first", "well", "way", "even", "new", "want", "because",
    "any", "these", "give", "day", "most", "us",
    "system", "model", "data", "process", "compute", "function", "value",
    "result", "input", "output", "request", "response", "token", "prompt",
    "context", "query", "search", "document", "vector", "index", "cache",
    "memory", "buffer", "queue", "batch", "stream", "frame", "window",
    "layer", "node", "graph", "tree", "hash", "key", "store", "fetch",
    "load", "save", "update", "delete", "merge", "split", "join", "filter",
    "sort", "rank", "score", "weight", "bias", "gradient", "loss", "epoch",
    "step", "rate", "limit", "count", "size", "length", "depth", "width",
    "height", "scale", "ratio", "delta", "shift", "drift", "trend", "phase",
    "state", "event", "signal", "channel", "wire", "path", "route", "hop",
    "edge", "vertex", "trace", "log", "metric", "stat", "summary", "report",
    "agent", "tool", "call", "return", "error", "warning", "info", "debug",
    "user", "admin", "owner", "guest", "session", "token", "scope", "role",
    "policy", "rule", "config", "schema", "model", "field", "row", "column",
    "table", "database", "transaction", "commit", "rollback", "checkpoint",
    "snapshot", "version", "release", "build", "deploy", "restart", "scale",
    "health", "ready", "warm", "cold", "hit", "miss", "evict", "promote",
)

assert len(_WORDS) >= 200, "word list must have ≥200 entries for shape variety"


def synth_token_count(text: str) -> int:
    """Cheap whitespace tokenizer.  Returns the number of whitespace-separated tokens.

    Used by every workload generator to enforce token budgets.  Real
    LLM serving uses a model-specific tokenizer; for load-gen purposes
    whitespace tokens are an acceptable proxy (±15% accuracy per the
    plan acceptance criteria).
    """
    if not text:
        return 0
    return len(text.split())


def synth_paragraph(rng: random.Random, target_tokens: int) -> str:
    """Return a paragraph of approximately ``target_tokens`` words.

    Period-delimited sentences.  Words drawn uniformly from the inline
    list.  Final word count is within 15% of target; the actual count
    may be slightly under if the target is small.
    """
    if target_tokens <= 0:
        return ""
    words: list[str] = []
    while len(words) < target_tokens:
        n = min(20, target_tokens - len(words))
        words.extend(rng.choice(_WORDS) for _ in range(n))
        if len(words) < target_tokens:
            # Period every ~12 words, with the period word capitalized.
            last = words[-1]
            words[-1] = last.capitalize() + "."
    # Drop excess.
    words = words[:target_tokens]
    text = " ".join(words)
    # Capitalize the very first word.
    return text[0].upper() + text[1:]


def synth_document(rng: random.Random, target_tokens: int) -> str:
    """Return a document of ~``target_tokens`` words, split into paragraphs.

    Paragraphs are separated by blank lines.  Document structure makes
    RAG-style chunked contexts look more realistic.
    """
    if target_tokens <= 0:
        return ""
    # Aim for paragraphs of ~200 words each.
    para_size = 200
    n_paras = max(1, target_tokens // para_size)
    paras: list[str] = []
    remaining = target_tokens
    for i in range(n_paras):
        size = remaining if i == n_paras - 1 else para_size
        paras.append(synth_paragraph(rng, size))
        remaining -= size
        if remaining <= 0:
            break
    return "\n\n".join(paras)


def synth_query(rng: random.Random, target_tokens: int) -> str:
    """Return a question-shaped string of ~``target_tokens`` words.

    Wraps the paragraph in a question prefix to make RAG user queries
    look query-like rather than document-like.  Deterministic for a
    given RNG state.
    """
    if target_tokens <= 0:
        return ""
    body = synth_paragraph(rng, max(1, target_tokens - 6))
    return f"Question: {body} Please answer concisely."
