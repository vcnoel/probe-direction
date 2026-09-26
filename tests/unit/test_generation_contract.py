from __future__ import annotations

import json
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from eval_awareness_spectral.generation_contract import (
    FinalParseState,
    GenerationRecord,
    RepetitionDetector,
    RepetitionResult,
    SingleSequenceRepetitionStoppingCriteria,
    TerminalIds,
    TerminationReason,
    assess_generation,
    classify_termination,
    detect_repetition,
    normalize_terminal_ids,
    parse_final_span,
)


class _Config:
    def __init__(self, eos_token_id: int | list[int] | tuple[int, ...] | None) -> None:
        self.eos_token_id = eos_token_id


@pytest.mark.parametrize(
    ("configured", "fallback", "expected"),
    [
        (151645, None, (151645,)),
        ([151645, 151643, 151645], 151645, (151645, 151643)),
        ((151645, 151643), [151643, 99], (151645, 151643, 99)),
        (None, 151643, (151643,)),
        (None, None, ()),
    ],
)
def test_terminal_normalization_scalar_list_tuple_absent_and_fallback(
    configured: int | list[int] | tuple[int, ...] | None,
    fallback: int | list[int] | None,
    expected: tuple[int, ...],
) -> None:
    terminals = normalize_terminal_ids(_Config(configured), {"eos_token_id": fallback})
    assert terminals.ordered == expected
    assert terminals.primary == (expected[0] if expected else None)
    assert terminals.secondary == expected[1:]


def test_terminal_normalization_rejects_boolean_and_non_integer() -> None:
    with pytest.raises(TypeError, match="integer"):
        normalize_terminal_ids({"eos_token_id": True})
    with pytest.raises(TypeError, match="integer"):
        normalize_terminal_ids({"eos_token_id": [1, "2"]})


def test_assessment_emits_flat_structured_provenance() -> None:
    assessment = assess_generation(
        [4, 151643],
        generation_config={"eos_token_id": [151645, 151643]},
        tokenizer={"eos_token_id": 151645},
        generated_cap=8,
    )
    provenance = assessment.provenance_dict()
    assert provenance["termination_contract_version"] == 1
    assert provenance["termination_reason"] == "secondary_eos"
    assert provenance["termination_declared_ids"] == "151645,151643"
    assert provenance["termination_scoring_eligible"] is True
    assert provenance["generated_token_count"] == 2


def test_single_sequence_online_stopper_is_batch_safe() -> None:
    stopper = SingleSequenceRepetitionStoppingCriteria(prompt_length=2)
    assert not stopper([[100, 101, 7, 7]], scores=None)
    assert stopper([[100, 101, *([7] * 48)]], scores=None)
    assert stopper.result.detected
    with pytest.raises(ValueError, match="batch of one"):
        SingleSequenceRepetitionStoppingCriteria(0)([[1], [2]], scores=None)


def test_classifies_primary_secondary_cap_error_and_immutable_result() -> None:
    terminals = TerminalIds((151645, 151643))
    primary = classify_termination([4, 151645, 0], terminals, generated_cap=3)
    secondary = classify_termination([4, 151643], terminals, generated_cap=3)
    capped = classify_termination([4, 5, 6], terminals, generated_cap=3)
    errored = classify_termination([], terminals, generated_cap=3, error=True)

    assert (primary.reason, primary.terminal_token_id, primary.terminal_index) == (
        TerminationReason.EOS,
        151645,
        1,
    )
    assert secondary.reason is TerminationReason.SECONDARY_EOS
    assert secondary.complete and secondary.scoring_eligible and not secondary.degenerate
    assert capped.reason is TerminationReason.MAX_TOKENS
    assert not capped.complete and not capped.scoring_eligible
    assert errored.reason is TerminationReason.ERROR
    with pytest.raises(FrozenInstanceError):
        primary.complete = False  # type: ignore[misc]


def test_detects_single_token_phrase_and_paragraph_suffix_loops() -> None:
    loops = [
        [7] * 48,
        [3, 4] * 24,
        list(range(16)) * 4,
    ]
    results = [detect_repetition(tokens) for tokens in loops]
    assert all(result.detected for result in results)
    assert [result.period for result in results] == [1, 2, 16]
    assert all(result.repeated_mass >= 48 for result in results)


def test_online_repetition_matches_batch_decision_and_global_indices() -> None:
    tokens = list(range(50)) + [3, 4] * 24
    detector = RepetitionDetector()
    online = detector.extend(tokens)
    batch = detect_repetition(tokens)
    assert online.detected == batch.detected
    assert online.period == batch.period == 2
    assert (online.start_index, online.end_index) == (50, 98)


def test_actual_qwen_failure_suffix_is_caught_before_the_4096_token_cap() -> None:
    fixture_path = Path(__file__).parents[1] / "fixtures" / "qwen3_repetition_loop_compact.json"
    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
    prefix_count = int(fixture["prefix_token_count"])
    # Distinct negative IDs stand in for the deliberately omitted source prefix. The copied
    # suffix tokens and source positions remain exact and checksum-addressed in the fixture.
    prefix = [-(index + 1) for index in range(prefix_count)]
    tokens = prefix + fixture["token_ids"]

    result = detect_repetition(tokens)

    assert result.detected
    assert result.period == len(fixture["repeated_unit_token_ids"])
    assert result.end_index == fixture["source_slice_span"][1]
    assert result.end_index < fixture["source_generated_token_count"]


def test_repetition_detector_rejects_short_refrain_lists_and_code_like_controls() -> None:
    short_refrain = [10, 11, 12, 13] * 3
    repeated_chorus = list(range(8)) * 4
    numbered_list = [value for item in range(20) for value in (90, item, 91)]
    code_like = [value for item in range(16) for value in (1, item, 2, 3)]
    punctuation_rich = [value for item in range(20) for value in (45, item, 44)]
    ordinary_text = list(range(64))

    for control in (
        short_refrain,
        repeated_chorus,
        numbered_list,
        code_like,
        punctuation_rich,
        ordinary_text,
    ):
        assert not detect_repetition(control).detected


def test_loop_classification_is_always_degenerate_incomplete_and_ineligible() -> None:
    loop = detect_repetition([1, 2] * 24)
    result = classify_termination([1, 2] * 24, TerminalIds((9,)), generated_cap=48, loop=loop)
    assert result.reason is TerminationReason.REPETITION_LOOP
    assert result.degenerate and not result.complete and not result.scoring_eligible


def test_post_hoc_loop_detection_ignores_a_later_terminal_token() -> None:
    assessment = assess_generation(
        [1, 2] * 24 + [151645],
        generation_config={"eos_token_id": [151645, 151643]},
        generated_cap=64,
    )
    assert assessment.repetition.detected
    assert assessment.termination.reason is TerminationReason.REPETITION_LOOP
    assert not assessment.termination.scoring_eligible


@pytest.mark.parametrize(
    ("tokens", "termination", "markers", "expected_state", "expected_final"),
    [
        (
            [10, 151645],
            classify_termination([10, 151645], TerminalIds((151645,)), generated_cap=4),
            [(70,)],
            FinalParseState.EOS_BEFORE_CLOSE,
            (),
        ),
        (
            [10, 70, 151645],
            classify_termination([10, 70, 151645], TerminalIds((151645,)), generated_cap=4),
            [(70,)],
            FinalParseState.CLOSE_WITH_EMPTY_FINAL,
            (),
        ),
        (
            [10, 70, 21],
            classify_termination([10, 70, 21], TerminalIds((151645,)), generated_cap=3),
            [(70,)],
            FinalParseState.CAP_AFTER_PARTIAL_FINAL,
            (21,),
        ),
        (
            [10, 70, 21, 21],
            classify_termination(
                [10, 70, 21, 21],
                TerminalIds((151645,)),
                generated_cap=10,
                loop=RepetitionResult(True, 2, 4, 1, 2, 1.0),
            ),
            [(70,)],
            FinalParseState.LOOP_AFTER_CLOSE,
            (21, 21),
        ),
        (
            [10, 21],
            classify_termination([10, 21], TerminalIds((151645,)), generated_cap=2),
            [(70,)],
            FinalParseState.MISSING_CLOSE,
            (),
        ),
        (
            [10, 70, 21, 151645],
            classify_termination(
                [10, 70, 21, 151645], TerminalIds((151645,)), generated_cap=4
            ),
            [(70,)],
            FinalParseState.VALID_FINAL,
            (21,),
        ),
    ],
)
def test_all_final_parse_states(
    tokens: list[int],
    termination: object,
    markers: list[tuple[int, ...]],
    expected_state: FinalParseState,
    expected_final: tuple[int, ...],
) -> None:
    # Parameter typing is kept explicit without importing runtime model/tokenizer types.
    assert hasattr(termination, "reason")
    span = parse_final_span(tokens, markers, termination)  # type: ignore[arg-type]
    assert span.state is expected_state
    assert span.final_tokens == expected_final


def test_first_close_marker_supports_single_and_multi_token_variants() -> None:
    tokens = [5, 80, 81, 6, 70, 7, 151643]
    termination = classify_termination(tokens, TerminalIds((151645, 151643)), generated_cap=10)
    span = parse_final_span(tokens, [(70,), (80, 81)], termination)
    assert (span.close_start, span.close_end) == (1, 3)
    assert span.final_tokens == (6, 70, 7)
    assert span.state is FinalParseState.VALID_FINAL


def test_activation_alignment_stops_before_terminal_and_retains_provenance() -> None:
    record = GenerationRecord.from_unpadded_tokens(
        [100, 101, 20, 21, 151643, 0, 0],
        prompt_length=2,
        terminal_ids=TerminalIds((151645, 151643)),
        provenance={"row_id": "r-1", "model": "qwen"},
    )
    assert record.raw_tokens == (100, 101, 20, 21, 151643, 0, 0)
    assert record.generated_tokens == (20, 21)
    assert record.activation_indices == (2, 3)
    assert len(record.generated_tokens) == len(record.activation_indices)
    assert dict(record.provenance) == {"model": "qwen", "row_id": "r-1"}


@pytest.mark.parametrize(
    ("reason", "generated", "terminals", "expected_tokens"),
    [
        (TerminationReason.EOS, [20, 151645], TerminalIds((151645, 151643)), (20,)),
        (TerminationReason.SECONDARY_EOS, [20, 151643], TerminalIds((151645, 151643)), (20,)),
        (TerminationReason.REPETITION_LOOP, [20, 20, 20], TerminalIds((151645,)), (20, 20, 20)),
        (TerminationReason.MAX_TOKENS, [20, 21], TerminalIds((151645,)), (20, 21)),
        (TerminationReason.ERROR, [20], TerminalIds((151645,)), (20,)),
    ],
)
def test_activation_alignment_for_every_termination_type(
    reason: TerminationReason,
    generated: list[int],
    terminals: TerminalIds,
    expected_tokens: tuple[int, ...],
) -> None:
    record = GenerationRecord.from_unpadded_tokens(
        [100, 101, *generated],
        prompt_length=2,
        terminal_ids=terminals,
        provenance={"termination_reason": reason.value},
    )
    assert record.generated_tokens == expected_tokens
    assert len(record.generated_tokens) == len(record.activation_indices)
    assert record.activation_indices == tuple(range(2, 2 + len(expected_tokens)))


def test_batch_padding_invariance_and_per_row_terminal_ids() -> None:
    padded_rows = ([1, 20, 151645, 0, 0], [1, 30, 151643, 0, 0])
    unpadded_rows = (padded_rows[0][:3], padded_rows[1][:3])
    terminal_rows = (TerminalIds((151645,)), TerminalIds((151643, 151645)))

    records = [
        GenerationRecord.from_unpadded_tokens(row, prompt_length=1, terminal_ids=terminals)
        for row, terminals in zip(unpadded_rows, terminal_rows, strict=True)
    ]
    results = [
        classify_termination(row[1:], terminals, generated_cap=4)
        for row, terminals in zip(unpadded_rows, terminal_rows, strict=True)
    ]
    assert [record.generated_tokens for record in records] == [(20,), (30,)]
    assert [result.reason for result in results] == [
        TerminationReason.EOS,
        TerminationReason.EOS,
    ]
    assert [result.terminal_index for result in results] == [1, 1]
