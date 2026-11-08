"""Check the paraphrased framings in data/framings_paraphrase.json against their originals.

Each of the 12 evaluation and deployment framings of data/framings_multi.json has two paraphrases,
keyed <original>_p1 and <original>_p2. Only the `system` text is reworded; `family` and
`user_template` are copied verbatim and `label` gains " (paraphrase 1)" or " (paraphrase 2)".
Rules checked for every paraphrase (docs/paraphrase-registration.md):

  length   whitespace word count within 10 percent of the original's (ratio in [0.90, 1.10]);
  overlap  no 4-word sequence shared with the original or with the sibling paraphrase, unless all
           four words are function words (FUNCTION below).

Words for the overlap check are lowercased runs of letters, digits and apostrophes, with hyphens
splitting words ("red-team" is "red", "team"), and British -ise/-our spellings of words longer
than five letters mapped to -ize/-or so a spelling variant cannot pass as a different word.

Prints one row per paraphrase (length ratio, longest shared n-gram of any kind and longest shared
n-gram containing a content word, against the original and against the sibling) and exits
non-zero if any rule is broken.

    python scripts/check_paraphrases.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ORIG = ROOT / "data" / "framings_multi.json"
PARA = ROOT / "data" / "framings_paraphrase.json"
RATIO = (0.90, 1.10)
NGRAM = 4

FUNCTION = set("""
a an the
of in on at by for with to from into onto over under about through across after before between
during against among within without beneath below above behind toward towards upon via per off out
up down around along
and or but nor so yet if as than that because while whether though although when whenever where
wherever whatever whoever however
i you he she it we they me him her us them my your his its our their mine yours ours theirs
this these those who whom whose which what someone anyone anything something nothing everything
everyone nobody each every any some all no none one both either neither other another such
is are was were be been being am has have had having do does did doing will would can could may
might shall should must
not just here there now then very also only own more most too again once still even ever
""".split())


def words(text):
    toks = re.findall(r"[a-z0-9']+", text.lower().replace("-", " "))
    out = []
    for t in toks:
        if len(t) > 5:
            t = re.sub(r"is(e|ed|es|ing|ation|ations)$", r"iz\1", t)
            t = re.sub(r"our$", "or", t)
        out.append(t)
    return out


def ngrams(ws, n):
    return {tuple(ws[i:i + n]) for i in range(len(ws) - n + 1)}


def longest_shared(a, b, content_only=False):
    """Length of the longest word n-gram present in both, optionally requiring a content word."""
    best = 0
    for n in range(1, min(len(a), len(b)) + 1):
        shared = ngrams(a, n) & ngrams(b, n)
        if content_only:
            shared = {g for g in shared if any(w not in FUNCTION for w in g)}
        if not shared and n > best + 1:
            break
        if shared:
            best = n
    return best


def violations(a, b):
    """Shared 4-grams that contain at least one content word."""
    return sorted(g for g in ngrams(a, NGRAM) & ngrams(b, NGRAM)
                  if any(w not in FUNCTION for w in g))


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    orig = {k: v for k, v in json.loads(ORIG.read_text(encoding="utf-8")).items()
            if not k.startswith("_") and v.get("family") in ("eval", "deploy")}
    para = {k: v for k, v in json.loads(PARA.read_text(encoding="utf-8")).items()
            if not k.startswith("_")}
    errors = []
    want = {f"{k}_p{v}" for k in orig for v in (1, 2)}
    if set(para) != want:
        errors.append(f"key set differs: missing {sorted(want - set(para))}, "
                      f"extra {sorted(set(para) - want)}")
    print(f"{'paraphrase':<24} {'words':>5} {'orig':>4} {'ratio':>6}  "
          f"{'vs original':>15}  {'vs sibling':>15}")
    print(f"{'':<24} {'':>5} {'':>4} {'':>6}  {'any/content':>15}  {'any/content':>15}")
    rows = []
    for k in sorted(orig):
        o = orig[k]
        for v in (1, 2):
            key = f"{k}_p{v}"
            if key not in para:
                continue
            p, sib = para[key], para.get(f"{k}_p{3 - v}")
            for field in ("family", "user_template"):
                if p.get(field) != o[field]:
                    errors.append(f"{key}: {field} differs from the original")
            if p.get("label") != f"{o['label']} (paraphrase {v})":
                errors.append(f"{key}: label is not the original label + ' (paraphrase {v})'")
            n_p, n_o = len(p["system"].split()), len(o["system"].split())
            ratio = n_p / n_o
            wp, wo = words(p["system"]), words(o["system"])
            ws = words(sib["system"]) if sib else []
            row = {"key": key, "words": n_p, "orig_words": n_o, "ratio": ratio,
                   "orig_any": longest_shared(wp, wo), "orig_content": longest_shared(wp, wo, True),
                   "sib_any": longest_shared(wp, ws), "sib_content": longest_shared(wp, ws, True)}
            rows.append(row)
            print(f"{key:<24} {n_p:>5} {n_o:>4} {ratio:>6.3f}  "
                  f"{row['orig_any']:>7}/{row['orig_content']:<7}  "
                  f"{row['sib_any']:>7}/{row['sib_content']:<7}")
            if not RATIO[0] <= ratio <= RATIO[1]:
                errors.append(f"{key}: length ratio {ratio:.3f} outside {RATIO}")
            for g in violations(wp, wo):
                errors.append(f"{key}: shares '{' '.join(g)}' with the original")
            if v == 1 and sib:
                for g in violations(wp, ws):
                    errors.append(f"{key}: shares '{' '.join(g)}' with its sibling")
    if len(rows) != 24:
        errors.append(f"checked {len(rows)} paraphrases, expected 24")
    if errors:
        print("\nFAIL")
        for e in errors:
            print("  " + e)
        return 1
    print(f"\nPASS: {len(rows)} paraphrases, ratios {min(r['ratio'] for r in rows):.3f} to "
          f"{max(r['ratio'] for r in rows):.3f}, longest shared content n-gram "
          f"{max(max(r['orig_content'], r['sib_content']) for r in rows)} words")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
