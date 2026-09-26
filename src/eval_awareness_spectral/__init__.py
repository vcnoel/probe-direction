"""Public, CPU-safe API for the evaluation-cue causal research protocol.

The prompt-building helpers are available through lazy attributes so importing
the package never imports a model, tokenizer, dataset client, or GPU framework.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

from .config import ProtocolConfig, load_config
from .generation_contract import (
    FinalParseState,
    FinalSpan,
    GenerationAssessment,
    GenerationRecord,
    RepetitionDetector,
    RepetitionResult,
    SingleSequenceRepetitionStoppingCriteria,
    TerminalIds,
    TerminationReason,
    TerminationResult,
    assess_generation,
    detect_repetition,
    normalize_terminal_ids,
    parse_final_span,
)
from .schemas import (
    CacheConflictError,
    ClaimStatus,
    IncompleteRunError,
    ProtocolError,
    SpectralComparisonMode,
    ValidationError,
)

__version__ = "0.3.0"

__all__ = [
    "CacheConflictError",
    "ClaimStatus",
    "FinalParseState",
    "FinalSpan",
    "GenerationAssessment",
    "GenerationRecord",
    "IncompleteRunError",
    "ProtocolConfig",
    "ProtocolError",
    "RepetitionDetector",
    "RepetitionResult",
    "SingleSequenceRepetitionStoppingCriteria",
    "SpectralComparisonMode",
    "TerminalIds",
    "TerminationReason",
    "TerminationResult",
    "ValidationError",
    "assess_generation",
    "detect_repetition",
    "load_config",
    "normalize_terminal_ids",
    "parse_final_span",
]

_LEGACY_EXPORTS = {
    "Condition": ("prompts", "Condition"),
    "Task": ("prompts", "Task"),
    "build_prompts": ("prompts", "build_prompts"),
    "load_framings": ("prompts", "load_framings"),
    "load_tasks": ("prompts", "load_tasks"),
}


def __getattr__(name: str) -> Any:
    """Load historical empirical APIs only when a caller requests one."""

    if name not in _LEGACY_EXPORTS:
        raise AttributeError(name)
    module_name, attribute = _LEGACY_EXPORTS[name]
    module = import_module(f"{__name__}.{module_name}")
    return getattr(module, attribute)
