"""Fail if SAD item text appears where it must not (docs/behavioural-anchor-registration.md).

SAD's terms: question and answer text must never appear in plain text anywhere scrapable. This
builds every 12-word sequence of every sample body in the decrypted struct files that anything here
reads: test_oversight.yaml and deploy_oversight.yaml, which the stages_oversight task evaluates
(sad/stages/oversight/run.py, batch_names), and oversight.yaml, which the probe pipeline read for
its testing items. The other struct files belong to SAD experiments not run here and include
public text (a GPL licence in stg-pretrain.yaml) that would match unrelated files. It looks for
any of them in the files given: by default everything under results/ and every file git tracks,
plus any --extra path (for example a log directory before it is uploaded). Words are lowercased runs
of letters and digits, so JSON escaping and line breaks cannot hide a match. A body shorter than 12
words contributes itself whole if it has at least 8 words; shorter bodies are counted and reported,
not matched. The task's prompt templates (structs/sampletemplates, structs/templateextensions) are
matched too and reported separately, as a warning, since the probe code quotes SAD's instruction
line. A hit names the file and the item id or template key, never the text.

    python scripts/check_no_sad_text.py --sad-dir <clone> [--extra <dir> ...]
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STRUCTS = Path("sad") / "stages" / "oversight" / "structs" / "batch"
N = 12
ITEM_FILES = ("test_oversight.yaml", "deploy_oversight.yaml", "oversight.yaml")
TEXT_SUFFIXES = {".json", ".csv", ".tsv", ".txt", ".md", ".log", ".yaml", ".yml", ".tex", ".py",
                 ".sh", ".html", ".jsonl", ".env", ""}


def words(text: str):
    return re.findall(r"[a-z0-9]+", text.lower())


def _add(table, w, label):
    if len(w) >= N:
        for i in range(len(w) - N + 1):
            table.setdefault(tuple(w[i:i + N]), label)
        return True
    if len(w) >= 8:
        table.setdefault(tuple(w), label)
        return True
    return False


def _strings(node, path=""):
    if isinstance(node, str):
        yield path, node
    elif isinstance(node, dict):
        for k, v in node.items():
            yield from _strings(v, f"{path}.{k}" if path else str(k))
    elif isinstance(node, list):
        for i, v in enumerate(node):
            yield from _strings(v, f"{path}[{i}]")


def shingles(sad_dir: Path):
    """-> (item table, template table, number of short bodies, number of struct files)."""
    import yaml
    items, templates, short, n_files = {}, {}, 0, 0
    root = sad_dir / STRUCTS.parent
    for name in ITEM_FILES:
        y = yaml.safe_load((root / "batch" / name).read_text(encoding="utf-8", errors="replace"))
        n_files += 1
        for s in y["samples"]:
            short += not _add(items, words(s["body"]), s["id"])
    for sub in ("sampletemplates", "templateextensions"):
        for f in sorted((root / sub).glob("*.yaml")):
            y = yaml.safe_load(f.read_text(encoding="utf-8", errors="replace"))
            for key, text in _strings(y):
                _add(templates, words(text), f"{sub}/{f.name}:{key}")
    return items, templates, short, n_files


def _find(w, table, lengths):
    for n in lengths:
        for i in range(len(w) - n + 1):
            sid = table.get(tuple(w[i:i + n]))
            if sid is not None:
                return sid
    return None


def files_to_scan(extra):
    tracked = subprocess.run(["git", "-C", str(ROOT), "ls-files"], capture_output=True, text=True,
                             check=True).stdout.splitlines()
    paths = {ROOT / p for p in tracked}
    paths |= {p for p in (ROOT / "results").rglob("*") if p.is_file()}
    for e in extra:
        e = Path(e)
        paths |= {p for p in e.rglob("*") if p.is_file()} if e.is_dir() else ({e} if e.exists() else set())
    return sorted(p for p in paths if p.suffix.lower() in TEXT_SUFFIXES and p.is_file())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sad-dir", required=True, type=Path)
    ap.add_argument("--extra", nargs="*", default=[])
    a = ap.parse_args()
    table, templates, short, n_files = shingles(a.sad_dir)
    lengths = sorted({len(k) for k in table})
    tlengths = sorted({len(k) for k in templates})
    files = files_to_scan(a.extra)
    if not table or not files or n_files < len(ITEM_FILES):
        print("vacuous check: no shingles or no files", file=sys.stderr)
        return 2
    hits, warn = [], []
    for f in files:
        try:
            w = words(f.read_text(encoding="utf-8", errors="ignore"))
        except OSError:
            continue
        sid = _find(w, table, lengths)
        if sid is not None:
            hits.append((f, sid))
        tid = _find(w, templates, tlengths)
        if tid is not None:
            warn.append((f, tid))
    print(f"scanned {len(files)} files against {len(table)} item sequences from {n_files} struct "
          f"files ({short} bodies under 8 words not matched) and {len(templates)} template sequences")
    for f, tid in warn:
        print(f"  note: SAD template text in {f} ({tid})", file=sys.stderr)
    if hits:
        for f, sid in hits:
            print(f"  SAD ITEM TEXT FOUND in {f} (item id {sid})", file=sys.stderr)
        return 1
    print("no SAD item text found")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
