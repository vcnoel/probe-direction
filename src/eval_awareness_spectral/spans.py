"""Locate the byte-identical *task* token span inside a chat-templated prompt.

The subgraph-isolation pillar needs the exact token indices of the task text
within the full prompt, so we can build the attention graph on *only* those
nodes. The indices must align with how the attention-graph instrumenter tokenizes
the prompt for the forward pass:

    tokenizer(text, return_tensors="pt", max_length=..., truncation=True, padding=True)

which uses ``add_special_tokens=True``. We therefore re-tokenize the same string
the same way (plus ``return_offsets_mapping=True``) and map the task substring's
character span to a contiguous token-index range. Those indices index directly
into the instrumenter's token sequence, so they can be passed straight to the
per-layer subgraph construction.

A task's token subsequence must be *identical* across every framing (that is the
whole premise of the experiment). `assert_task_spans_identical` enforces it and
lets the runner skip a task rather than silently compare mismatched subgraphs.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TaskSpan:
    """Token span [start, end) of the task text within a tokenized prompt."""
    start: int
    end: int
    token_ids: Tuple[int, ...]  # the task tokens, for cross-condition identity checks
    truncated: bool             # True if max_length truncation clipped the task span

    @property
    def indices(self) -> List[int]:
        return list(range(self.start, self.end))

    def __len__(self) -> int:
        return self.end - self.start


class SpanError(RuntimeError):
    """Raised when the task span cannot be located reliably."""


def locate_task_span(
    tokenizer,
    templated_text: str,
    task_text: str,
    max_length: int = 512,
) -> TaskSpan:
    """Return the contiguous token span of ``task_text`` inside ``templated_text``.

    Uses fast-tokenizer offset mappings (the reliable path for gemma/qwen/llama).
    Falls back to a decoded-subsequence search if offsets are unavailable.
    """
    char_start = templated_text.rfind(task_text)
    if char_start < 0:
        raise SpanError("task text not found verbatim in templated prompt")
    char_end = char_start + len(task_text)

    if getattr(tokenizer, "is_fast", False):
        span = _span_via_offsets(tokenizer, templated_text, char_start, char_end, max_length)
        if span is not None:
            return span
        logger.warning("offset-mapping span location failed; falling back to id search")

    return _span_via_id_subsequence(tokenizer, templated_text, task_text, max_length)


def _tok_ids(tokenizer, text: str, max_length: int):
    """Tokenize exactly as the instrumenter does (add_special_tokens=True)."""
    return tokenizer(
        text,
        max_length=max_length,
        truncation=True,
        add_special_tokens=True,
    )


def _span_via_offsets(
    tokenizer, text: str, char_start: int, char_end: int, max_length: int
) -> Optional[TaskSpan]:
    enc = tokenizer(
        text,
        max_length=max_length,
        truncation=True,
        add_special_tokens=True,
        return_offsets_mapping=True,
    )
    offsets = enc["offset_mapping"]
    ids = enc["input_ids"]
    seq_len = len(ids)

    idxs = [
        i
        for i, (a, b) in enumerate(offsets)
        if b > a and a >= char_start and b <= char_end  # inside task; skip (0,0) specials
    ]
    if not idxs:
        return None

    start, end = idxs[0], idxs[-1] + 1
    if idxs != list(range(start, end)):
        logger.warning("task token span is non-contiguous; using [%d,%d)", start, end)

    # Truncation clips the *end* of the sequence; the task sits at the end of the
    # prompt, so a clipped task shows up as the span running to seq_len.
    truncated = end >= seq_len and offsets[-1][1] < char_end
    token_ids = tuple(ids[start:end])
    return TaskSpan(start=start, end=end, token_ids=token_ids, truncated=truncated)


def _span_via_id_subsequence(
    tokenizer, templated_text: str, task_text: str, max_length: int
) -> TaskSpan:
    """Fallback: find the task token id subsequence inside the prompt token ids.

    Tokenizing ``task_text`` in isolation can differ at the boundaries from how it
    tokenizes in context, so we search for the *interior* of the task tokens and
    tolerate boundary drift of one token on each side.
    """
    full_ids = _tok_ids(tokenizer, templated_text, max_length)["input_ids"]
    task_ids = tokenizer(task_text, add_special_tokens=False)["input_ids"]
    if not task_ids:
        raise SpanError("task text tokenizes to empty")

    # search for the longest interior run that appears contiguously in full_ids
    for trim in range(0, min(3, len(task_ids) // 2 + 1)):
        needle = task_ids[trim: len(task_ids) - trim] if trim else task_ids
        pos = _find_subsequence(full_ids, needle)
        if pos >= 0:
            start = pos - trim if trim else pos
            end = start + len(task_ids)
            start = max(start, 0)
            end = min(end, len(full_ids))
            truncated = end >= len(full_ids)
            return TaskSpan(
                start=start, end=end,
                token_ids=tuple(full_ids[start:end]), truncated=truncated,
            )
    raise SpanError("task token subsequence not found in prompt token ids")


def _find_subsequence(haystack: Sequence[int], needle: Sequence[int]) -> int:
    n, m = len(haystack), len(needle)
    if m == 0 or m > n:
        return -1
    first = needle[0]
    for i in range(n - m + 1):
        if haystack[i] == first and list(haystack[i : i + m]) == list(needle):
            return i
    return -1


def reconcile_task_spans(spans_by_condition: dict) -> dict:
    """Keep the majority task tokenization; null out only the outlier condition(s).

    Position-matching can perturb the task's *boundary* token in one frame (e.g. a
    bare ``{task}`` template vs one ending in ``\\n\\n{task}``), shifting its span by
    a token. Rather than discard the whole task, we keep every condition whose task
    token-ids match the modal tokenization and drop only the deviants — so the
    comparable frames still yield subgraphs on an identical node set.
    """
    from collections import Counter
    valid = {k: v for k, v in spans_by_condition.items() if v is not None}
    if len(valid) < 2:
        return dict(spans_by_condition)
    modal_ids, _ = Counter(v.token_ids for v in valid.values()).most_common(1)[0]
    out = {}
    for k, v in spans_by_condition.items():
        out[k] = v if (v is not None and v.token_ids == modal_ids) else None
    dropped = [k for k, v in out.items() if v is None and spans_by_condition.get(k) is not None]
    if dropped:
        logger.warning("dropped %s from subgraph (task tokenization differs from majority)", dropped)
    return out


def assert_task_spans_identical(spans_by_condition: dict) -> bool:
    """True iff every condition's task span has the same token ids (same length).

    The experiment holds task content byte-identical across frames, so the task's
    token *subsequence* must match. If it does not (e.g. a chat template mangled a
    boundary), the subgraphs are not comparable and the task should be skipped.
    """
    items = [(k, v.token_ids) for k, v in spans_by_condition.items() if v is not None]
    if len(items) < 2:
        return True
    ref_key, ref_ids = items[0]
    ok = True
    for key, ids in items[1:]:
        if ids != ref_ids:
            logger.warning(
                "task token span differs between %r (len %d) and %r (len %d); "
                "subgraphs not comparable",
                ref_key, len(ref_ids), key, len(ids),
            )
            ok = False
    return ok
