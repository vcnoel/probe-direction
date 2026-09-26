"""Dependency-free contracts for generation termination, parsing, and alignment.

All token positions in this module refer to an unpadded, per-row token sequence.  Callers
must remove batch padding before constructing a :class:`GenerationRecord` or running the
repetition detector.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Literal, Protocol, TypeAlias, cast

TokenIds: TypeAlias = tuple[int, ...]
TerminationMode: TypeAlias = Literal["post_hoc", "online"]
TERMINATION_CONTRACT_VERSION = 1


class _HasEosTokenId(Protocol):
    eos_token_id: int | Sequence[int] | None


EosSource: TypeAlias = _HasEosTokenId | Mapping[str, object] | None


class TerminationReason(StrEnum):
    """The mutually exclusive reason that generation stopped."""

    EOS = "eos"
    SECONDARY_EOS = "secondary_eos"
    REPETITION_LOOP = "repetition_loop"
    MAX_TOKENS = "max_tokens"
    ERROR = "error"


class FinalParseState(StrEnum):
    """Outcome of separating reasoning tokens from final-answer tokens."""

    EOS_BEFORE_CLOSE = "eos_before_close"
    CLOSE_WITH_EMPTY_FINAL = "close_with_empty_final"
    CAP_AFTER_PARTIAL_FINAL = "cap_after_partial_final"
    LOOP_AFTER_CLOSE = "loop_after_close"
    MISSING_CLOSE = "missing_close"
    VALID_FINAL = "valid_final"


@dataclass(frozen=True)
class TerminalIds:
    """Ordered terminal IDs, with the first ID treated as the primary EOS."""

    ordered: TokenIds

    @property
    def primary(self) -> int | None:
        return self.ordered[0] if self.ordered else None

    @property
    def secondary(self) -> TokenIds:
        return self.ordered[1:]

    def reason_for(self, token_id: int) -> TerminationReason | None:
        if self.primary == token_id:
            return TerminationReason.EOS
        if token_id in self.secondary:
            return TerminationReason.SECONDARY_EOS
        return None


@dataclass(frozen=True)
class RepetitionResult:
    """A conservative repetition decision and its supporting suffix statistics."""

    detected: bool
    start_index: int | None = None
    end_index: int | None = None
    period: int | None = None
    repeated_mass: int = 0
    repeated_ngram_coverage: float = 0.0


@dataclass(frozen=True)
class TerminationResult:
    """Structured termination status used by parsers and scoring code."""

    reason: TerminationReason
    terminal_token_id: int | None
    terminal_index: int | None
    complete: bool
    degenerate: bool
    scoring_eligible: bool


@dataclass(frozen=True)
class FinalSpan:
    """Reasoning-close parse result; ``final_*`` delimit a half-open token span."""

    state: FinalParseState
    close_start: int | None
    close_end: int | None
    final_start: int | None
    final_end: int | None
    final_tokens: TokenIds


@dataclass(frozen=True)
class GenerationAssessment:
    """Termination, repetition, and terminal-set evidence for one generated row."""

    terminal_ids: TerminalIds
    repetition: RepetitionResult
    termination: TerminationResult
    generated_token_count: int
    mode: TerminationMode

    def provenance_dict(self) -> dict[str, bool | float | int | str | None]:
        """Return flat, CSV/JSON-safe termination provenance."""

        return {
            "termination_contract_version": TERMINATION_CONTRACT_VERSION,
            "termination_mode": self.mode,
            "termination_reason": self.termination.reason.value,
            "termination_terminal_token_id": self.termination.terminal_token_id,
            "termination_terminal_index": self.termination.terminal_index,
            "termination_complete": self.termination.complete,
            "termination_degenerate": self.termination.degenerate,
            "termination_scoring_eligible": self.termination.scoring_eligible,
            "termination_declared_ids": ",".join(str(value) for value in self.terminal_ids.ordered),
            "generated_token_count": self.generated_token_count,
            "repetition_detected": self.repetition.detected,
            "repetition_start_index": self.repetition.start_index,
            "repetition_end_index": self.repetition.end_index,
            "repetition_period": self.repetition.period,
            "repetition_repeated_mass": self.repetition.repeated_mass,
            "repetition_ngram_coverage": self.repetition.repeated_ngram_coverage,
        }


@dataclass(frozen=True)
class GenerationRecord:
    """Raw row tokens plus provenance and aligned nonterminal generation positions."""

    raw_tokens: TokenIds
    prompt_length: int
    generated_tokens: TokenIds
    activation_indices: tuple[int, ...]
    provenance: tuple[tuple[str, str], ...] = ()

    @classmethod
    def from_unpadded_tokens(
        cls,
        raw_tokens: Sequence[int],
        *,
        prompt_length: int,
        terminal_ids: TerminalIds,
        provenance: Mapping[str, str] | None = None,
    ) -> GenerationRecord:
        raw = tuple(raw_tokens)
        if not 0 <= prompt_length <= len(raw):
            raise ValueError("prompt_length must be within raw_tokens")
        end = len(raw)
        for index in range(prompt_length, len(raw)):
            if raw[index] in terminal_ids.ordered:
                end = index
                break
        indices = tuple(range(prompt_length, end))
        return cls(
            raw_tokens=raw,
            prompt_length=prompt_length,
            generated_tokens=tuple(raw[index] for index in indices),
            activation_indices=indices,
            provenance=tuple(sorted((provenance or {}).items())),
        )


def _read_eos_token_id(source: EosSource) -> object:
    if source is None:
        return None
    if isinstance(source, Mapping):
        return source.get("eos_token_id")
    return source.eos_token_id


def _coerce_eos_ids(value: object, *, source_name: str) -> TokenIds:
    if value is None:
        return ()
    if isinstance(value, bool):
        raise TypeError(f"{source_name}.eos_token_id must contain integer token IDs")
    if isinstance(value, int):
        return (value,)
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"{source_name}.eos_token_id must be an int, list, tuple, or None")
    result: list[int] = []
    for token_id in cast(Sequence[object], value):
        if isinstance(token_id, bool) or not isinstance(token_id, int):
            raise TypeError(f"{source_name}.eos_token_id must contain integer token IDs")
        result.append(token_id)
    return tuple(result)


def normalize_terminal_ids(
    generation_config: EosSource = None,
    tokenizer: EosSource = None,
) -> TerminalIds:
    """Normalize configured EOS IDs, then append unique tokenizer fallback IDs.

    This intentionally accepts small protocols and mappings rather than importing a model
    or tokenizer package.  Order is stable, and the first configured/fallback ID is primary.
    """

    candidates = _coerce_eos_ids(
        _read_eos_token_id(generation_config), source_name="generation_config"
    ) + _coerce_eos_ids(_read_eos_token_id(tokenizer), source_name="tokenizer")
    ordered: list[int] = []
    seen: set[int] = set()
    for token_id in candidates:
        if token_id not in seen:
            seen.add(token_id)
            ordered.append(token_id)
    return TerminalIds(tuple(ordered))


def detect_repetition(
    tokens: Sequence[int],
    *,
    min_generated_tokens: int = 48,
    min_repeated_mass: int = 48,
    min_repeats: int = 4,
    max_period: int = 32,
    ngram_size: int = 4,
    coverage_window: int = 64,
    min_ngram_coverage: float = 0.90,
) -> RepetitionResult:
    """Detect a suffix loop using exact periods or high repeated n-gram coverage.

    Defaults require at least 48 generated tokens and 48 tokens of repeated mass.  Exact
    suffixes need four copies.  The rolling fallback requires 90% repeated 4-gram coverage
    in a window of at most 64 tokens, making ordinary lists and short refrains conservative
    negatives while still catching long, mildly irregular loops.
    """

    row = tuple(tokens)
    if min_generated_tokens < 1 or min_repeated_mass < 1 or min_repeats < 2:
        raise ValueError("repetition guards must be positive and min_repeats must be at least 2")
    if max_period < 1 or ngram_size < 1 or coverage_window < ngram_size:
        raise ValueError("period, ngram size, and coverage window must be positive and coherent")
    if not 0.0 <= min_ngram_coverage <= 1.0:
        raise ValueError("min_ngram_coverage must be in [0, 1]")
    if len(row) < min_generated_tokens:
        return RepetitionResult(False)

    maximum = min(max_period, len(row) // min_repeats)
    for period in range(1, maximum + 1):
        copies = 1
        while (
            (copies + 1) * period <= len(row)
            and row[-period:] == row[-(copies + 1) * period : -copies * period]
        ):
            copies += 1
        mass = copies * period
        if copies >= min_repeats and mass >= min_repeated_mass:
            return RepetitionResult(
                True,
                start_index=len(row) - mass,
                end_index=len(row),
                period=period,
                repeated_mass=mass,
                repeated_ngram_coverage=1.0,
            )

    window_start = max(0, len(row) - coverage_window)
    window = row[window_start:]
    ngram_count = len(window) - ngram_size + 1
    if ngram_count <= 0:
        return RepetitionResult(False)
    seen_ngrams: set[TokenIds] = set()
    repeated = 0
    for start in range(ngram_count):
        ngram = window[start : start + ngram_size]
        if ngram in seen_ngrams:
            repeated += 1
        else:
            seen_ngrams.add(ngram)
    coverage = repeated / ngram_count
    mass = max(0, repeated + ngram_size - 1)
    if mass >= min_repeated_mass and coverage >= min_ngram_coverage:
        return RepetitionResult(
            True,
            start_index=window_start,
            end_index=len(row),
            period=None,
            repeated_mass=mass,
            repeated_ngram_coverage=coverage,
        )
    return RepetitionResult(False, repeated_mass=mass, repeated_ngram_coverage=coverage)


class RepetitionDetector:
    """Stateful online wrapper around :func:`detect_repetition` with bounded history."""

    def __init__(
        self,
        *,
        min_generated_tokens: int = 48,
        min_repeated_mass: int = 48,
        min_repeats: int = 4,
        max_period: int = 32,
        ngram_size: int = 4,
        coverage_window: int = 64,
        min_ngram_coverage: float = 0.90,
    ) -> None:
        # Reuse the batch detector's validation so online and batch configuration agree.
        detect_repetition(
            (),
            min_generated_tokens=min_generated_tokens,
            min_repeated_mass=min_repeated_mass,
            min_repeats=min_repeats,
            max_period=max_period,
            ngram_size=ngram_size,
            coverage_window=coverage_window,
            min_ngram_coverage=min_ngram_coverage,
        )
        self._min_generated_tokens = min_generated_tokens
        self._min_repeated_mass = min_repeated_mass
        self._min_repeats = min_repeats
        self._max_period = max_period
        self._ngram_size = ngram_size
        self._coverage_window = coverage_window
        self._min_ngram_coverage = min_ngram_coverage
        history_size = max(
            min_generated_tokens,
            min_repeated_mass,
            min_repeats * max_period,
            coverage_window,
        )
        self._tokens: deque[int] = deque(maxlen=history_size)
        self._token_count = 0

    def update(self, token_id: int) -> RepetitionResult:
        """Consume one unpadded row token and return the current deterministic decision."""

        if isinstance(token_id, bool) or not isinstance(token_id, int):
            raise TypeError("token_id must be an integer")
        self._tokens.append(token_id)
        self._token_count += 1
        if self._token_count < self._min_generated_tokens:
            return RepetitionResult(False)
        result = detect_repetition(
            tuple(self._tokens),
            min_generated_tokens=min(self._min_generated_tokens, len(self._tokens)),
            min_repeated_mass=self._min_repeated_mass,
            min_repeats=self._min_repeats,
            max_period=self._max_period,
            ngram_size=self._ngram_size,
            coverage_window=self._coverage_window,
            min_ngram_coverage=self._min_ngram_coverage,
        )
        offset = self._token_count - len(self._tokens)
        return RepetitionResult(
            detected=result.detected,
            start_index=None if result.start_index is None else result.start_index + offset,
            end_index=None if result.end_index is None else result.end_index + offset,
            period=result.period,
            repeated_mass=result.repeated_mass,
            repeated_ngram_coverage=result.repeated_ngram_coverage,
        )

    def extend(self, token_ids: Sequence[int]) -> RepetitionResult:
        """Consume tokens in order and return the decision after the last token."""

        result = RepetitionResult(False)
        for token_id in token_ids:
            result = self.update(token_id)
        return result


class SingleSequenceRepetitionStoppingCriteria:
    """Transformers-compatible online stopper for one unpadded generation row.

    The class deliberately avoids importing Transformers or a tensor library.  Pass an
    instance inside the runtime's ``StoppingCriteriaList``.  Batches are rejected rather
    than globally stopping unrelated rows when one row loops.
    """

    def __init__(self, prompt_length: int, *, detector: RepetitionDetector | None = None) -> None:
        if prompt_length < 0:
            raise ValueError("prompt_length must be nonnegative")
        self._prompt_length = prompt_length
        self._processed = 0
        self._detector = detector or RepetitionDetector()
        self.result = RepetitionResult(False)

    def __call__(self, input_ids: object, scores: object, **kwargs: Any) -> bool:
        del scores, kwargs
        raw = input_ids.tolist() if hasattr(input_ids, "tolist") else input_ids
        if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)) or len(raw) != 1:
            raise ValueError("single-sequence repetition stopping requires a batch of one row")
        row_value = raw[0]
        if not isinstance(row_value, Sequence) or isinstance(row_value, (str, bytes)):
            raise TypeError("input_ids must contain a sequence of integer token IDs")
        row = _coerce_eos_ids(row_value, source_name="input_ids")
        generated = row[self._prompt_length :]
        if len(generated) < self._processed:
            raise ValueError("input_ids became shorter between stopping-criteria calls")
        self.result = self._detector.extend(generated[self._processed :])
        self._processed = len(generated)
        return self.result.detected


def classify_termination(
    tokens: Sequence[int],
    terminal_ids: TerminalIds,
    *,
    generated_cap: int,
    loop: RepetitionResult | None = None,
    error: bool = False,
) -> TerminationResult:
    """Classify one unpadded generated row, preferring errors and detected loops."""

    if generated_cap < 0:
        raise ValueError("generated_cap must be nonnegative")
    if error:
        return TerminationResult(TerminationReason.ERROR, None, None, False, False, False)
    if loop is not None and loop.detected:
        return TerminationResult(
            TerminationReason.REPETITION_LOOP, None, loop.end_index, False, True, False
        )
    for index, token_id in enumerate(tokens):
        reason = terminal_ids.reason_for(token_id)
        if reason is not None:
            return TerminationResult(reason, token_id, index, True, False, True)
    # A nonterminal short row can only arise from an upstream stop/error; conservatively
    # represent it as an incomplete cap rather than claiming successful completion.
    _ = generated_cap
    return TerminationResult(TerminationReason.MAX_TOKENS, None, None, False, False, False)


def assess_generation(
    tokens: Sequence[int],
    *,
    generation_config: EosSource = None,
    tokenizer: EosSource = None,
    generated_cap: int,
    loop: RepetitionResult | None = None,
    mode: TerminationMode = "post_hoc",
    error: bool = False,
) -> GenerationAssessment:
    """Build the complete validity assessment for one unpadded generated row."""

    if mode not in {"post_hoc", "online"}:
        raise ValueError("mode must be 'post_hoc' or 'online'")
    row = tuple(tokens)
    terminal_ids = normalize_terminal_ids(generation_config, tokenizer)
    content_end = next(
        (index for index, token_id in enumerate(row) if token_id in terminal_ids.ordered),
        len(row),
    )
    repetition = detect_repetition(row[:content_end]) if loop is None else loop
    termination = classify_termination(
        row,
        terminal_ids,
        generated_cap=generated_cap,
        loop=repetition,
        error=error,
    )
    return GenerationAssessment(
        terminal_ids=terminal_ids,
        repetition=repetition,
        termination=termination,
        generated_token_count=len(row),
        mode=mode,
    )


def _first_marker(tokens: TokenIds, markers: Sequence[Sequence[int]]) -> tuple[int, int] | None:
    best: tuple[int, int, int] | None = None
    for marker_order, marker_value in enumerate(markers):
        marker = tuple(marker_value)
        if not marker:
            raise ValueError("reasoning close markers must not be empty")
        limit = len(tokens) - len(marker) + 1
        for start in range(max(0, limit)):
            if tokens[start : start + len(marker)] == marker:
                candidate = (start, marker_order, start + len(marker))
                if best is None or candidate[:2] < best[:2]:
                    best = candidate
                break
    return None if best is None else (best[0], best[2])


def parse_final_span(
    tokens: Sequence[int],
    close_markers: Sequence[Sequence[int]],
    termination: TerminationResult,
) -> FinalSpan:
    """Parse the first reasoning close marker and final span before termination."""

    row = tuple(tokens)
    terminal_end = termination.terminal_index
    content_end = len(row) if terminal_end is None else min(terminal_end, len(row))
    marker = _first_marker(row[:content_end], close_markers)
    if marker is None:
        state = (
            FinalParseState.EOS_BEFORE_CLOSE
            if termination.reason in {TerminationReason.EOS, TerminationReason.SECONDARY_EOS}
            else FinalParseState.MISSING_CLOSE
        )
        return FinalSpan(state, None, None, None, None, ())

    close_start, close_end = marker
    final_tokens = row[close_end:content_end]
    if termination.reason is TerminationReason.REPETITION_LOOP:
        state = FinalParseState.LOOP_AFTER_CLOSE
    elif not final_tokens:
        state = FinalParseState.CLOSE_WITH_EMPTY_FINAL
    elif termination.reason in {TerminationReason.EOS, TerminationReason.SECONDARY_EOS}:
        state = FinalParseState.VALID_FINAL
    else:
        state = FinalParseState.CAP_AFTER_PARTIAL_FINAL
    return FinalSpan(state, close_start, close_end, close_end, content_end, final_tokens)
