"""The model-by-wrapper share of the real directions against every content-free null, per ladder.

Post-submission descriptive check (docs/post-submission-checks.md), not registered: a direct read of
numbers already computed by scripts/matched_null.py. For each ladder, the share of variance of the
model x wrapper term (`mw_share`) and of the model term (`model_share`) for the real directions, and
for each null family the real value's mid-rank percentile, the null mean and its 5th to 95th
percentiles. Reported for the all-layer statistic (matched_null.json) and for the registered
interior band (matched_null_band.json).

    python scripts/mw_share_nulls.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from result_io import write_result  # noqa: E402

SOURCES = {"all_layers": "post_submission/matched_null.json",
           "interior_band": "post_submission/matched_null_band.json"}
OUT = ROOT / "results" / "post_submission" / "mw_share_nulls.json"
KEYS = ("mw_share", "model_share")


def main() -> None:
    out = {}
    for tag, src in SOURCES.items():
        d = json.loads((ROOT / "results" / src).read_text(encoding="utf-8"))
        out[tag] = {}
        for ladder, L in d["ladders"].items():
            rec = {}
            for q in KEYS:
                rec[q] = {"real": L["real"][q], "nulls": {
                    k: {"percentile": c[q]["percentile"], "null_mean": c[q]["null_mean"],
                        "null_p05_p95": c[q]["null_p05_p95"], "outside_95": c[q]["outside_95"],
                        "n": c[q]["n"]} for k, c in L["comparison"].items()}}
            out[tag][ladder] = rec
            m = rec["mw_share"]
            print(f"{tag:13s} {ladder:16s} M x W real {m['real']:.3f}  pct " + "  ".join(
                f"{k} {v['percentile']:.0f}" for k, v in m["nulls"].items()), flush=True)
    write_result(OUT, {"by_layer_set": out, "quantities": list(KEYS),
                       "check": "post-submission descriptive check, not registered "
                                "(docs/post-submission-checks.md)"},
                 kind="derived", sources=list(SOURCES.values()),
                 statistic="model x wrapper and model shares of the 3-way crossed G-study, real "
                           "directions placed in each content-free null by mid-rank percentile")
    print(f"wrote {OUT.relative_to(ROOT).as_posix()}")


if __name__ == "__main__":
    main()
