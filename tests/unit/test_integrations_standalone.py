from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from eval_awareness_spectral.data.manifests import build_prompt_record, write_manifest
from eval_awareness_spectral.provenance import sha256_file
from eval_awareness_spectral.schemas import Split


def test_importing_integrations_does_not_import_optional_modules() -> None:
    names = ("torch", "transformers", "sglang", "lm_eval")
    before = {name: name in sys.modules for name in names}
    import eval_awareness_spectral.integrations  # noqa: F401

    assert {name: name in sys.modules for name in names} == before


def test_local_numpy_path_exposes_representation_signal_and_provenance() -> None:
    from eval_awareness_spectral.integrations import adapt_numpy_activations

    values = np.arange(24, dtype=np.float64).reshape(2, 3, 4)
    batch = adapt_numpy_activations(
        values,
        sequence_ids=("a", "b"),
        layers=(7,),
        token_indices=(10, 11, 12),
    )

    np.testing.assert_array_equal(batch.signal("b", 7), values[1])
    rows = list(batch.rows())
    assert (rows[-1].sequence_id, rows[-1].layer, rows[-1].token_index) == ("b", 7, 12)
    decomposition = batch.ordinary_dc_ac("a", 7)
    np.testing.assert_allclose(decomposition.dc + decomposition.ac, values[0])


def test_local_manifest_loaders_validate_prompt_and_result_files(tmp_path: Path) -> None:
    from eval_awareness_spectral.integrations import load_prompt_manifest, load_results_manifest

    prompt_path = tmp_path / "prompts.jsonl"
    record = build_prompt_record(
        source_task_id="source",
        template_family="template",
        environment_factors={"cue": True},
        condition_valence="evaluation",
        content="content",
        rendered_text="rendered content",
        task_tokens=(1, 2),
        tokenizer_revision="tokenizer-v1",
        task_span=(0, 2),
        split=Split.TRAIN,
    )
    write_manifest(prompt_path, [record])
    assert load_prompt_manifest(prompt_path) == [record]

    dataset = tmp_path / "dataset.bin"
    split = tmp_path / "split.json"
    metric = tmp_path / "metric.json"
    for path in (dataset, split, metric):
        path.write_text(path.name, encoding="utf-8")
    manifest = {
        "schema_version": 1,
        "run_id": "run",
        "dataset": {"location": dataset.name, "revision": "v1", "checksum": sha256_file(dataset)},
        "models": [{"model_id": "model", "revision": "v1", "family": "test"}],
        "tokenizer_revision": "v1",
        "config_hash": "sha256:" + "0" * 64,
        "code_commit": "abcdef0",
        "environment_lock_hash": "sha256:" + "1" * 64,
        "seed": 0,
        "feature_definitions": ["activation"],
        "split_manifest": {
            "path": split.name,
            "checksum": sha256_file(split),
            "group_keys": ["source_task_id"],
            "confirmation_locked": True,
        },
        "metrics": [
            {
                "claim_id": "claim",
                "definition": "metric",
                "split": "confirmation",
                "sample_size": 1,
                "result_path": metric.name,
                "checksum": sha256_file(metric),
            }
        ],
        "generated_at": "2026-01-01T00:00:00Z",
    }
    result_path = tmp_path / "results.json"
    result_path.write_text(json.dumps(manifest), encoding="utf-8")
    assert load_results_manifest(result_path)["run_id"] == "run"
