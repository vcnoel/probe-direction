"""Macros for the measured reliability of the wrapper-averaged statistic, from one result file.

Writes subset_numbers.tex next to the manuscript. Like paper_tools/make_ladders.py, a ladder whose
results are not on disk gets no macros, and the manuscript guards its use with \\ifqfive.

Provenance (file, field, unit): results/wrapper_subsets.json, written by
scripts/wrapper_subsets.py in the repository.
  Sub<tag>Ksign      ladders.<key>.k_sign_095          one ladder, k-subsets against the 36 average
  Sub<tag>Ktau       ladders.<key>.k_tau_080           same unit, mean Kendall tau of the ordering
  Sub<tag>DisKsign   ladders.<key>.disjoint_k_sign_095 one ladder, two disjoint k-subsets, k <= 18
  Sub<tag>DisKtau    ladders.<key>.disjoint_k_tau_080  same unit
  Sub<tag>SignOne    ladders.<key>.sign_agreement[0]   one wrapper against the 36 average
  Sub<tag>Rfull      ladders.<key>.r_full_average      correlation of the 36 average with log params
  Sub<tag>EqFix      ladders.<key>.eq1.k_080_items_fixed   Equation 1 with the item set held fixed
A null disjoint threshold renders as "no $k \\le 18$" and a null Equation 1 value as "none". The
flags Sub<tag>DisKtauReached and Sub<tag>EqFixFinite (1 or 0) let the prose word a null case as a
sentence of its own, so no sentence reads "reaches 0.80 at no $k \\le 18$".

    python paper_tools/make_subsets.py [--strict]
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

REPO = Path(os.environ.get("EAS_REPO", Path(__file__).resolve().parents[1]))
# Generated .tex and figure files go to paper_tools/out/ unless EAS_OUT names another directory.
HERE = Path(os.environ.get("EAS_OUT", REPO / "paper_tools" / "out"))
HERE.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(REPO / "scripts"))
from result_io import check_conventions, check_freshness, check_producer  # noqa: E402

TAGS = [("qwen3.5_nothink", "Qfive", r"\texttt{Qwen3.5}"),
        ("qwen3.5_think", "QfiveT", r"\texttt{Qwen3.5}, thinking on"),
        ("qwen3_nothink", "Qthree", r"\texttt{Qwen3}"),
        ("qwen3_think", "QthreeT", r"\texttt{Qwen3}, thinking on"),
        ("olmo2", "Olmo", r"\texttt{OLMo-2}"),
        ("qwen2.5_paper", "Qtf", r"\texttt{Qwen2.5}")]
WORDS = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six"}


def main() -> None:
    strict = "--strict" in sys.argv
    p = REPO / "results" / "wrapper_subsets.json"
    d = json.loads(p.read_text(encoding="utf-8"))
    probs = check_conventions(d, p.name) + check_freshness(d, p.name) + check_producer(d, p.name)
    if probs:
        msg = "\n  ".join(probs)
        if strict:
            raise SystemExit(f"refusing to build on a stale or undeclared input:\n  {msg}")
        print(f"  WARNING (non-strict build):\n  {msg}")
    lad = d["ladders"]
    m: dict[str, str] = {
        "SubNDraws": f"{d['n_draws']:,}".replace(",", "{,}"),
        "SubSignTarget": f"{100 * d['sign_target']:.0f}\\%",
        "SubTauTarget": f"{d['tau_target']:.2f}",
    }
    present = []
    for key, tag, name in TAGS:
        if key not in lad:
            continue
        r = lad[key]
        kmax = r["disjoint_k"][-1]
        m["SubDisKmax"] = str(kmax)

        def dis(v):
            return str(v) if v is not None else f"no $k \\le {kmax}$"
        e = r["eq1"]
        m.update({
            f"Sub{tag}Ksign": str(r["k_sign_095"]),
            f"Sub{tag}Ktau": str(r["k_tau_080"]),
            f"Sub{tag}DisKsign": dis(r["disjoint_k_sign_095"]),
            f"Sub{tag}DisKtau": dis(r["disjoint_k_tau_080"]),
            f"Sub{tag}SignOne": f"{r['sign_agreement'][0]:.2f}",
            f"Sub{tag}TauOne": f"{r['mean_tau'][0]:.2f}",
            f"Sub{tag}Rfull": f"{r['r_full_average']:+.2f}",
            f"Sub{tag}EqFix": "none" if e["k_080_items_fixed"] is None else str(e["k_080_items_fixed"]),
            # 1 or 0, read with \ifnum by the prose that words each null case as its own sentence.
            f"Sub{tag}DisKtauReached": "0" if r["disjoint_k_tau_080"] is None else "1",
            f"Sub{tag}EqFixFinite": "0" if e["k_080_items_fixed"] is None else "1",
        })
        assert e["matches_pod_gtheory_k_for_080"], f"{key}: Equation 1 here disagrees with pod_gtheory"
        present.append((key, tag, name, r))
    # Cross-ladder summaries over the renderings without the thinking switch, the ones the figure
    # draws. The recommendation quotes the worst ladder, so both ends are named.
    prim = [t for t in present if not t[0].endswith("_think")]
    for field, mac in (("k_sign_095", "Ksign"), ("k_tau_080", "Ktau")):
        lo = min(prim, key=lambda t: t[3][field])
        hi = max(prim, key=lambda t: t[3][field])
        m[f"Sub{mac}Min"], m[f"Sub{mac}MinName"] = str(lo[3][field]), lo[2]
        m[f"Sub{mac}Max"], m[f"Sub{mac}MaxName"] = str(hi[3][field]), hi[2]
    m["SubNPrim"] = WORDS[len(prim)]
    # Disjoint reading, the one without overlap, which the abstract and the checklist quote. The
    # friendliest ladder is named with its k; every other primary ladder is reported by its value
    # at the largest disjoint k (half the pool), where it has not reached the target.
    res = [t for t in prim if t[3]["disjoint_k_tau_080"] is not None]
    if res:
        lo = min(res, key=lambda t: t[3]["disjoint_k_tau_080"])
        m["SubDisKtauMin"], m["SubDisKtauMinName"] = str(lo[3]["disjoint_k_tau_080"]), lo[2]
    m["SubNDisUnresolved"] = WORDS.get(len(prim) - len(res), str(len(prim) - len(res)))
    for key, tag, _, r in present:
        m[f"Sub{tag}DisTauHalf"] = f"{r['disjoint_mean_tau'][-1]:.2f}"
        m[f"Sub{tag}DisSignHalf"] = f"{100 * r['disjoint_sign_agreement'][-1]:.0f}\%"
        m[f"Sub{tag}SignOnePct"] = f"{100 * r['sign_agreement'][0]:.0f}\%"
    # The abstract says the ladder both published studies use is not resolved by disjoint halves.
    assert lad["qwen2.5_paper"]["disjoint_k_tau_080"] is None, "Qwen2.5 now resolves; rewrite"
    # The prose says two disjoint halves of the Qwen2.5 pool agree on the sign less often than not.
    assert lad["qwen2.5_paper"]["disjoint_sign_agreement"][-1] < 0.5, "Qwen2.5 halves now agree"
    # The sentence on the averaged scaling result says every ladder's 36-wrapper average is
    # positive. A ladder whose average is negative stops the build and the sentence is rewritten.
    assert all(t[3]["r_full_average"] > 0 for t in present), "a 36-wrapper average is negative"
    # The prose names Qwen2.5 as the ladder whose averaged sign needs nearly the whole pool.
    assert m["SubKsignMaxName"] == r"\texttt{Qwen2.5}", "Qwen2.5 is no longer the worst ladder"
    # Equation 1 with the items fixed and the disjoint reading agree in order: a ladder the
    # disjoint reading leaves unresolved at k <= 18 needs more than 18 by Equation 1, and one it
    # resolves needs no more than twice the resolved k.
    for key, _, _, r in present:
        kf, kd = r["eq1"]["k_080_items_fixed"], r["disjoint_k_tau_080"]
        if kd is None:
            assert kf is None or kf > r["disjoint_k"][-1], f"{key}: Equation 1 and disjoint disagree"
        else:
            assert kf is not None and kf <= 2 * kd and kd <= 2 * kf, \
                f"{key}: Equation 1 and disjoint disagree"
    for k, v in list(m.items()):
        if v[:1] in "+-":
            m[k] = f"\\ensuremath{{{v}}}"
    head = ["% GENERATED by paper_tools/make_subsets.py -- do not edit.",
            "% input: results/wrapper_subsets.json; ladders present: "
            + ", ".join(k for k, *_ in present)]
    (HERE / "subset_numbers.tex").write_text(
        "\n".join(head) + "\n" + "\n".join(f"\\newcommand{{\\{k}}}{{{v}}}"
                                           for k, v in sorted(m.items())) + "\n",
        encoding="utf-8")
    print(f"wrote subset_numbers.tex ({len(m)} macros; {', '.join(k for k, *_ in present)})")


if __name__ == "__main__":
    main()
