"""The only sanctioned way to write a results file. Every writer must declare its conventions.

Four defects in this project came from two statistics differing by an undeclared convention and
being presented as comparable. Each compiled cleanly and passed every check in
paper/fill_numbers.py, because the check warned about files that declared nothing instead of
refusing them. Guarding one failure class at a time is unbounded; the class closes only if silence
is impossible.

So a results file cannot be written without a complete conventions record, and paper/fill_numbers.py
exits non-zero on any input lacking one. There is no warning path.

Three kinds, because not every analysis has a single convention:

  measured   The analysis scores activations. It must name the pooling, the layer set, the scoring
             array and the data root. This is the common case.
  crossed    The conventions ARE the facet under study, as in the implementation-latitude analysis.
             It must enumerate the levels it crosses, so a reader knows the span rather than a point.
  derived    The analysis reads other results files and computes on their numbers. It must name
             them, and it inherits their conventions transitively.

    from result_io import write_result
    write_result(path, payload, kind="measured", pooling="lf", layers="interior",
                 scoring_array="results/gstudy/*/sad_filtered_lf.npz", data_root="gstudy")
"""

from __future__ import annotations

import inspect
import json
from pathlib import Path

REQUIRED = {
    # model_population is a convention like any other. Enabling a fourth family in the shared config
    # silently widened two analyses from 7 models to 10 the moment they were next run, changing the
    # population their components describe. Recording it makes a population change fail the build the
    # same way a pooling mismatch does, instead of shipping.
    "measured": ("pooling", "layers", "scoring_array", "data_root", "model_population"),
    "crossed": ("crossed_levels", "scoring_array", "data_root", "model_population"),
    "derived": ("sources",),
}


def build_conventions(kind, **fields):
    """Validate and return a conventions block. Raises rather than defaulting."""
    if kind not in REQUIRED:
        raise ValueError(f"kind must be one of {sorted(REQUIRED)}, not {kind!r}")
    missing = [k for k in REQUIRED[kind] if fields.get(k) in (None, "", [], {})]
    if missing:
        raise ValueError(f"kind={kind!r} requires {', '.join(missing)}; "
                         "declare the convention rather than omitting it")
    unknown = set(fields) - set(REQUIRED[kind]) - {"statistic", "note"}
    if unknown:
        raise ValueError(f"unexpected convention field(s): {sorted(unknown)}")
    return {"kind": kind, **fields}


def script_digest(path):
    """Semantic hash of a Python file: parsed AST with docstrings stripped.

    Hashing raw bytes would invalidate a results file on a comment or docstring edit. Rebuilds are
    cheap, but false alarms are how a check earns being ignored, and that failure mode is worse than
    the one this guards against. Only a change that could alter the numbers should trigger it.
    """
    import ast
    import hashlib
    tree = ast.parse(Path(path).read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            body = node.body
            if (body and isinstance(body[0], ast.Expr)
                    and isinstance(body[0].value, ast.Constant)
                    and isinstance(body[0].value.value, str)):
                node.body = body[1:] or [ast.Pass()]
    return hashlib.sha256(ast.dump(tree).encode("utf-8")).hexdigest()[:12]


def payload_digest(path):
    """Hash of a results file's CONTENT, excluding its conventions block.

    Hashing whole bytes coupled metadata to staleness: stamping a producer hash into a source file
    changed its bytes and made every dependent look stale, which is a false alarm and would train
    anyone to ignore the check. Only the payload decides freshness.
    """
    import hashlib
    import json as _json
    d = _json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(d, dict):          # some results files are JSON arrays
        d.pop("conventions", None)
    return hashlib.sha256(_json.dumps(d, sort_keys=True, separators=(",", ":"))
                          .encode("utf-8")).hexdigest()[:12]


def source_hashes(names):
    """{name: payload digest} for each results file a derived analysis read."""
    from pathlib import Path as _P
    res = _P(__file__).resolve().parents[1] / "results"
    out = {}
    for n in names:
        f = res / n
        if not f.exists():
            raise ValueError(f"declared source {n} does not exist; cannot record its hash")
        out[n] = payload_digest(f)
    return out


def write_result(path, payload, *, kind, **fields):
    """Write payload with a validated conventions block attached under 'conventions'.

    For kind='derived' the sources' content hashes are recorded, so a source regenerated afterwards
    makes this file detectably stale. Declaring the same convention as your source is not enough:
    a derived file computed from an earlier version of that source declares everything correctly and
    is still wrong, which is how a 67% became a 50% in a shipped manuscript."""
    if not isinstance(payload, dict):
        raise TypeError("payload must be a dict")
    if "conventions" in payload:
        raise ValueError("payload already carries a conventions key; pass them as arguments")
    out = dict(payload)
    conv = build_conventions(kind, **fields)
    if kind == "derived":
        conv["source_hashes"] = source_hashes(fields["sources"])
    # Stamp the producing script automatically. Until now only backfill_conventions.py recorded
    # this, so a file written through the SANCTIONED path came out without the producer hash that
    # detects a result computed by an older version of its own script -- and the manuscript's
    # coverage assertion then refused to build on it. The writer knows its own caller; it should
    # never have been the backfiller's job.
    if "producer" not in conv:
        caller = Path(inspect.stack()[1].filename).resolve()
        try:
            rel = caller.relative_to(Path(__file__).resolve().parents[1]).as_posix()
        except ValueError:
            rel = caller.name
        conv["producer"] = rel
        conv["producer_sha12"] = script_digest(caller)[:12]
    out["conventions"] = conv
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(out, indent=2), encoding="utf-8")
    tmp.replace(p)
    return p


def check_conventions(d, name="<result>"):
    """-> list of problems. Empty means the record is complete."""
    c = d.get("conventions")
    if not isinstance(c, dict):
        return [f"{name}: no conventions block"]
    kind = c.get("kind")
    if kind not in REQUIRED:
        return [f"{name}: conventions.kind is {kind!r}, not one of {sorted(REQUIRED)}"]
    return [f"{name}: conventions.{k} missing or empty"
            for k in REQUIRED[kind] if c.get(k) in (None, "", [], {})]


def require_nonvacuous(count, what, minimum=1):
    """Refuse to report success on an empty collection.

    A check that examined nothing exits zero and reads as a pass. That happened twice while the
    convention checks were being written -- an inverted mapping matched no macros, and a glob could
    have matched no files -- and in both cases the output said the manuscript was clean. Any check
    that can report success must first assert it had something to inspect.
    """
    if count < minimum:
        raise SystemExit(
            f"vacuous check: examined {count} {what} (minimum {minimum}). This would have reported "
            "success without inspecting anything; fix the input selection rather than trusting it.")
    return count

def params_b(key=None):
    """Parameter counts in billions, from configs/gstudy_models.json, derived from safetensors byte
    totals rather than from model names. Nominal names understate several models -- Qwen2.5-7B holds
    7.62B, a log10 error of 0.037 at the leverage point of a four-point fit -- so no script should
    carry its own dict of round numbers.
    """
    import json as _json
    from pathlib import Path as _Path
    conf = _json.loads((_Path(__file__).resolve().parents[1] / "configs" / "gstudy_models.json")
                       .read_text(encoding="utf-8"))["models"]
    table = {k: v["params_b"] for k, v in conf.items()}
    missing = [k for k, v in conf.items() if "params_b_source" not in v and v.get("enabled")]
    if missing:
        raise SystemExit(f"parameter counts not derived from weights for {missing}; "
                         "run scripts/pin_models.py --write")
    return table if key is None else table[key]


def check_freshness(d, name="<result>"):
    """-> list of problems. A derived file whose recorded source hashes no longer match is stale.

    This is a different failure from a mismatched convention and is invisible to that check: the
    declarations agree, the numbers do not.
    """
    import hashlib
    from pathlib import Path as _P
    c = d.get("conventions") or {}
    if c.get("kind") != "derived":
        return []
    rec = c.get("source_hashes")
    if not rec:
        return [f"{name}: derived but records no source hashes, so staleness cannot be detected"]
    res = _P(__file__).resolve().parents[1] / "results"
    bad = []
    for src, want in rec.items():
        f = res / src
        if not f.exists():
            bad.append(f"{name}: source {src} is missing")
            continue
        have = payload_digest(f)
        if have != want:
            bad.append(f"{name}: STALE -- derived from {src}@{want} but that file is now @{have}; "
                       f"re-run the analysis")
    return bad


def check_producer(d, name="<result>"):
    """-> problems if the producing script has changed since this file was written."""
    import hashlib
    from pathlib import Path as _P
    c = d.get("conventions") or {}
    prod, want = c.get("producer"), c.get("producer_sha12")
    if not prod or not want:
        return []
    f = _P(__file__).resolve().parents[1] / prod
    if not f.exists():
        return [f"{name}: producer {prod} is missing"]
    have = script_digest(f)
    return [] if have == want else [
        f"{name}: producer {prod} changed semantically since this was written "
        f"({want} -> {have}); re-run it"]
