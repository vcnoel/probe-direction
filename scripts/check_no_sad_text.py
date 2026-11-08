"""Fail if SAD item text appears where it must not (docs/behavioural-anchor-registration.md).

SAD's terms: question and answer text must never appear in plain text anywhere scrapable. This
builds every 12-word sequence of every stages_oversight item body from the decrypted struct files
and looks for any of them in the files given: by default everything under results/ and every file
git tracks, plus any --extra path (for example a log directory before it is uploaded). Words are
lowercased runs of letters and digits, so JSON escaping and line breaks cannot hide a match. A body
shorter than 12 words contributes itself whole if it has at least 8 words; shorter bodies are
counted and reported, not matched. A hit names the file and the item id, never the text.

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
TEXT_SUFFIXES = {".json", ".csv", ".tsv", ".txt", ".md", ".log", ".yaml", ".yml", ".tex", ".py",
                 ".sh", ".html", ".jsonl", ".env", ""}


def words(text: str):
    return re.findall(r"[a-z0-9]+", text.lower())


def shingles(sad_dir: Path):
    import yaml
    table, short = {}, 0
    for fname in ("oversight.yaml", "deploy_oversight.yaml"):
        y = yaml.safe_load((sad_dir / STRUCTS / fname).read_text(encoding="utf-8", errors="replace"))
        for s in y["samples"]:
            w = words(s["body"])
            if len(w) >= N:
                for i in range(len(w) - N + 1):
                    table.setdefault(tuple(w[i:i + N]), s["id"])
            elif len(w) >= 8:
                table.setdefault(tuple(w), s["id"])
            else:
                short += 1
    return table, short


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
    table, short = shingles(a.sad_dir)
    lengths = {len(k) for k in table}
    files = files_to_scan(a.extra)
    if not table or not files:
        print("vacuous check: no shingles or no files", file=sys.stderr)
        return 2
    hits = []
    for f in files:
        try:
            w = words(f.read_text(encoding="utf-8", errors="ignore"))
        except OSError:
            continue
        found = None
        for n in lengths:
            for i in range(len(w) - n + 1):
                sid = table.get(tuple(w[i:i + n]))
                if sid is not None:
                    found = sid
                    break
            if found:
                break
        if found:
            hits.append((f, found))
    print(f"scanned {len(files)} files against {len(table)} item sequences "
          f"({short} bodies under 8 words not matched)")
    if hits:
        for f, sid in hits:
            print(f"  SAD ITEM TEXT FOUND in {f} (item id {sid})", file=sys.stderr)
        return 1
    print("no SAD item text found")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
