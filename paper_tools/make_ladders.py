"""Macros and the main-text table for every ladder measured, from the result files.

Writes two files next to the manuscript:

  ladder_numbers.tex  one macro family per ladder, \\Lad<tag><field>, plus release months
  ladders_table.tex   the table of all ladders the body prints (tab:ladders)

A ladder whose results are not on disk gets no macros at all, rather than empty ones, and the
manuscript guards every use of it with \\ifdefined, so the PDF builds cleanly before it lands and
picks it up with no edit after. Adding the Qwen3.5 ladder is therefore: run scripts/pod_ladders.py
and scripts/ladder_curves.py in the repository, then make.

Provenance, one line per macro family (file, field, unit):
  Rep fields       results/pod_replication.json   ladders.<key>             one ladder, 36 wrappers
  Model, Erho, K   results/pod_gtheory.json       ladders.<key>.components  one ladder, 3-way crossed
  SB               results/pod_slope_reliability  ladders.<key>             one ladder, 20 splits
  Date             configs/release_dates.json     ladders.<family>          hub creation month

Erho is the coefficient for ONE wrapper scored on the FULL item set (k = 1, n = 4 item blocks), the
design every published study uses. pod_gtheory.json also stores erho_single at n = 1 (a quarter of
the items); the manuscript does not print that one, so no sentence can mix the two.

    python paper_tools/make_ladders.py [--strict]
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
from gstudy_gtheory import coefficients  # noqa: E402
from result_io import check_conventions, check_producer  # noqa: E402

# Ladder key in the result files -> (macro tag, display name, family key for the release month,
# rendering note). Order is the order of the table: newest generation first.
LADDERS = [
    ("qwen3.5_nothink", "Qfive", r"\texttt{Qwen3.5}", "qwen3.5", ""),
    ("qwen3.5_think", "QfiveT", r"\texttt{Qwen3.5}", "qwen3.5", "thinking on"),
    ("qwen3_nothink", "Qthree", r"\texttt{Qwen3}", "qwen3", ""),
    ("qwen3_think", "QthreeT", r"\texttt{Qwen3}", "qwen3", "thinking on"),
    ("olmo2", "Olmo", r"\texttt{OLMo-2}", "olmo2", ""),
    ("qwen2.5_paper", "Qtf", r"\texttt{Qwen2.5}", "qwen2.5", ""),
]
DATE_TAGS = {"qwen2.5": "Qtf", "qwen3": "Qthree", "olmo2": "Olmo", "qwen3.5": "Qfive",
             "gemma2": "Gemma", "llama3.2": "Llama", "smollm2": "Smol"}
MONTHS = "Jan Feb Mar Apr May Jun Jul Aug Sep Oct Nov Dec".split()


def month(ym: str) -> str:
    y, m = ym.split("-")
    return f"{MONTHS[int(m) - 1]} {y}"


def load(p: Path, strict: bool):
    d = json.loads(p.read_text(encoding="utf-8"))
    probs = check_conventions(d, p.name) + check_producer(d, p.name)
    if probs:
        msg = "\n  ".join(probs)
        if strict:
            raise SystemExit(f"refusing to build on a stale or undeclared input:\n  {msg}")
        print(f"  WARNING (non-strict build):\n  {msg}")
    return d


def fmt_b(x: float) -> str:
    return f"{x:.1f}" if x < 10 else f"{x:.0f}"


def main() -> None:
    strict = "--strict" in sys.argv
    res = REPO / "results"
    rep = load(res / "pod_replication.json", strict)["ladders"]
    gt = load(res / "pod_gtheory.json", strict)["ladders"]
    rel = load(res / "pod_slope_reliability.json", strict)["ladders"]
    dates = json.loads((REPO / "configs" / "release_dates.json").read_text(encoding="utf-8"))
    m: dict[str, str] = {}
    for fam, tag in DATE_TAGS.items():
        if fam in dates["ladders"]:
            d = dates["ladders"][fam]
            a, b = month(d["first_month"]), month(d["last_month"])
            m[f"Date{tag}"] = a if a == b else f"{a} to {b}"
    present = []
    for key, tag, name, fam, note in LADDERS:
        if key not in rep or key not in gt or key not in rel:
            print(f"  {key}: no results, no macros")
            continue
        q, g, s = rep[key], gt[key], rel[key]
        erho, phi = coefficients(g["components"], k=1, n=g["n_blocks"])
        p = q["params_b"]
        m.update({
            f"Lad{tag}N": str(len(q["models"])),
            f"Lad{tag}Span": f"{fmt_b(min(p))} to {fmt_b(max(p))}",
            f"Lad{tag}Dec": f"{q['decades']:.2f}",
            f"Lad{tag}Rlo": f"{q['r_min']:+.2f}",
            f"Lad{tag}Rhi": f"{q['r_max']:+.2f}",
            f"Lad{tag}Pos": str(q["n_positive"]),
            f"Lad{tag}Neg": str(q["n_negative"]),
            f"Lad{tag}MargExp": f"{q['expected_spanning_from_marginal']:.1f}",
            f"Lad{tag}EvalFix": str(q["n_eval_framings_spanning"]),
            f"Lad{tag}DepFix": str(q["n_deploy_framings_spanning"]),
            f"Lad{tag}Model": f"{100 * g['shares']['s_m']:.1f}\\%",
            f"Lad{tag}MW": f"{100 * g['shares']['s_mt']:.1f}\\%",
            f"Lad{tag}MI": f"{100 * g['shares']['s_mh']:.1f}\\%",
            f"Lad{tag}MWMH": f"{g['shares']['s_mt'] / g['shares']['s_mh']:.1f}",
            f"Lad{tag}Erho": f"{erho:.3f}",
            f"Lad{tag}Phi": f"{phi:.3f}",
            f"Lad{tag}K": "none" if g["k_for_080_unreachable"] else str(g["k_for_080"]),
            f"Lad{tag}SB": f"{s['spearman_brown']:.2f}",
        })
        present.append((key, tag, name, fam, note, erho))
    # The three Qwen3.5 predictions stated before extraction (pod/PREDICTIONS.md sec. 4), scored
    # by the rule written there, on the default-free rendering the table leads with.
    if "qwen3.5_nothink" in rep and "qwen3.5_nothink" in gt:
        q, g = rep["qwen3.5_nothink"], gt["qwen3.5_nothink"]
        sh = g["shares"]
        held = [q["r_min"] < 0 < q["r_max"],
                sh["s_m"] < sh["s_mt"] and sh["s_mt"] > sh["s_mh"],
                coefficients(g["components"], k=1, n=g["n_blocks"])[0] < 0.80]
        words = {0: "none", 1: "one", 2: "two", 3: "all three"}
        m["LadQfivePredHeld"] = words[sum(held)]
    # Cross-ladder summaries the prose quotes, over the renderings the table prints.
    er = [e for *_, e in present]
    lo = min(present, key=lambda t: t[5])
    hi = max(present, key=lambda t: t[5])
    m["LadErhoLo"], m["LadErhoHi"] = f"{lo[5]:.3f}", f"{hi[5]:.3f}"
    m["LadErhoLoName"], m["LadErhoHiName"] = lo[2], hi[2]
    fams = []
    for _, _, name, _, _, _ in present:
        if name not in fams:
            fams.append(name)
    words = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six"}
    m["LadNLadders"] = words[len(fams)]
    m["LadNRows"] = words[len(present)]
    m["LadAllSpan"] = "every" if all(rep[k]["r_min"] < 0 < rep[k]["r_max"] for k, *_ in present) \
        else "not every"
    assert max(er) < 0.80, "a ladder reached the conventional 0.80; the prose says none does"
    # Every claim the prose makes about "every ladder" is asserted here, so a ladder that breaks
    # one stops the build and the sentence is rewritten rather than silently falsified.
    keys = [k for k, *_ in present]
    for k in keys:
        q, sh = rep[k], gt[k]["shares"]
        assert q["r_min"] < 0 < q["r_max"], f"{k}: the wrapper no longer decides the sign"
        # OLMo-2 is the one ladder where the model share reaches model x wrapper, and the prose
        # names it as the exception. Any other ladder doing so stops the build.
        # Qwen3.5 with thinking on is the second: sec:gtheory reports it in its own sentence as the
        # rendering on the other side of the prediction scored on the rendering without thinking.
        assert sh["s_m"] < sh["s_mt"] or k in ("olmo2", "qwen3.5_think"), \
            f"{k}: model share exceeds model x wrapper"
        assert sh["s_mt"] > sh["s_mh"], f"{k}: model x item exceeds model x wrapper"
        assert q["n_eval_framings_spanning"] >= 1 and q["n_deploy_framings_spanning"] >= 1, \
            f"{k}: one arm never spans both signs; rewrite 'neither arm is privileged'"
    m["LadEvalFixMin"] = str(min(rep[k]["n_eval_framings_spanning"] for k in keys))
    m["LadEvalFixMax"] = str(max(rep[k]["n_eval_framings_spanning"] for k in keys))
    m["LadSBMin"] = f"{min(rel[k]['spearman_brown'] for k in keys):.2f}"
    others = [k for k in keys if k != "qwen2.5_paper"]
    assert all(rep[k]["n_positive"] > rep[k]["n_negative"] for k in others), \
        "a ladder other than Qwen2.5 is mostly negative; rewrite the sentence on skew"
    m["LadNegMin"] = str(min(rep[k]["n_negative"] for k in others))
    m["LadNegMax"] = str(max(rep[k]["n_negative"] for k in others))
    ks = [gt[k]["k_for_080"] for k in keys if not gt[k]["k_for_080_unreachable"]]
    m["LadKMin"] = str(min(ks))
    m["LadModelMax"] = f"{100 * max(gt[k]['shares']['s_m'] for k in keys):.1f}\%"
    assert any(gt[k]["k_for_080_unreachable"] for k in keys), "every ladder now reaches 0.80"
    # sec:dstudy names Qwen2.5 as the one ladder that never reaches 0.80 and prints a wrapper count
    # for every other rendering; a second unreachable ladder would print "none wrappers".
    unreach = sorted(k for k in keys if gt[k]["k_for_080_unreachable"])
    assert unreach == ["qwen2.5_paper"], f"unreachable on {unreach}; rewrite the sec:dstudy sentence"
    # sec:gtheory names OLMo-2 as "the one ladder where the two are level" (model share above three
    # quarters of model x wrapper). Another ladder that close makes the sentence false.
    # Qwen3.5 with thinking on is past level and has its own sentence there, so it is exempt.
    level = sorted(k for k in keys if gt[k]["shares"]["s_m"] > 0.75 * gt[k]["shares"]["s_mt"]
                   and k != "qwen3.5_think")
    assert level == ["olmo2"], f"level ladders now {level}; rewrite 'the one ladder where level'"

    # --- how extreme the wrapper set is, against the published constructions (Qwen2.5 only) ---
    # Unit: the four Qwen2.5 checkpoints. Our 36 wrappers per checkpoint (wrapper_curves.json,
    # matrix, folded max over all layers) against the two released studies' own values for the
    # same checkpoint (released_label_permuted.json per_model.published; the per-layer files
    # of the other study refolded in reanalysis_scaling_law.json, reported_max_distance).
    import numpy as np
    wc = json.loads((res / "wrapper_curves.json").read_text(encoding="utf-8"))
    x = np.array(wc["matrix"])
    rlp = json.loads((res / "released_label_permuted.json").read_text(encoding="utf-8"))["per_model"]
    chd = {r["model"].lower(): r["reported_max_distance"] for r in json.loads(
        (res / "reanalysis_scaling_law.json").read_text(encoding="utf-8"))}
    ranks, inside, above = [], 0, 0
    for i, mod in enumerate(wc["models"]):
        row = x[i]
        for v in (rlp[mod]["published"], chd[mod]):
            ranks.append(float((row < v).mean()))
            inside += int(row.min() <= v <= row.max())
            above += int(v > row.max())
    m["PubNVals"] = str(len(ranks))
    m["PubNInside"] = str(inside)
    m["PubNAbove"] = str(above)
    m["PubRankMed"] = f"{100 * float(np.median(ranks)):.0f}"
    pubr = json.loads((res / "slope_distribution_deploy_peak_lf.json").read_text(
        encoding="utf-8"))["published"]
    m["PubChaudQtf"] = f"{pubr['Chaudhary et al., as released']:+.2f}"
    m["PubManekQtf"] = f"{pubr['Manek et al.']:+.2f}"
    # The neutral arm: evaluation framings against six framings that say nothing about deployment.
    # Unit: the four Qwen2.5 checkpoints, 36 evaluation x neutral pairs, lf scoring.
    neu = json.loads((res / "slope_distribution_neutral_peak_lf.json").read_text(encoding="utf-8"))
    rn = np.array(neu["correlations"])
    m["NeuPos"], m["NeuNeg"] = str(int((rn > 0).sum())), str(int((rn < 0).sum()))
    m["NeuRlo"], m["NeuRhi"] = f"{rn.min():+.2f}", f"{rn.max():+.2f}"
    ng = json.loads((res / "gstudy_gtheory_neutral_peak_lf.json").read_text(encoding="utf-8"))
    m["NeuModel"] = f"{100 * ng['share']['s_m']:.1f}\\%"
    m["NeuMW"] = f"{100 * ng['share']['s_mt']:.1f}\\%"
    m["NeuMI"] = f"{100 * ng['share']['s_mh']:.1f}\\%"
    m["NeuErho"] = f"{ng['g_k1']:.3f}"
    # The withdrawn layer rule (app:layers, sec:dstudy). Unit: the three held-out models (the
    # non-Qwen2.5 families of the geometry set that have a gstudy_v2 extraction), 36 wrappers x 4
    # item blocks. File results/validate_layer_rule_validated.json, fields holdout_diff_ci and
    # held_out_g, written by scripts/validate_layer_rule.py (in the repository history at
    # 47e2719, removed from the tree at 6d124b3). The contrast is the INTERIOR BAND (relative depth
    # 0.25 to 0.75 of the hidden_states entries) minus TRUNK (every entry but the first and last),
    # E rho^2 for one wrapper, bootstrap over wrappers and item blocks.
    lr = json.loads((res / "validate_layer_rule_validated.json").read_text(encoding="utf-8"))
    held_keys = [k for k in lr["models"] if lr["families"][k] != "qwen2.5"]
    words = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven"}
    m["LayNHeld"] = words[len(held_keys)]
    m["LayCIlo"], m["LayCIhi"] = (f"{v:+.3f}" for v in lr["holdout_diff_ci"])
    m["LayHeldBand"] = f"{lr['held_out_g']['interior']:.3f}"
    m["LayHeldTrunk"] = f"{lr['held_out_g']['trunk']:.3f}"
    assert lr["holdout_diff_spans_zero"], "the layer-rule interval no longer spans zero; rewrite"

    for k, v in list(m.items()):
        if v[:1] in "+-":
            m[k] = f"\\ensuremath{{{v}}}"

    head = ["% GENERATED by paper_tools/make_ladders.py -- do not edit.",
            "% inputs: results/pod_replication.json, pod_gtheory.json, pod_slope_reliability.json,",
            "%         configs/release_dates.json; ladders present: "
            + ", ".join(k for k, *_ in present)]
    (HERE / "ladder_numbers.tex").write_text(
        "\n".join(head) + "\n" + "\n".join(f"\\newcommand{{\\{k}}}{{{v}}}"
                                           for k, v in sorted(m.items())) + "\n",
        encoding="utf-8")

    rows = []
    for key, tag, name, fam, note, _ in present:
        # The rendering row sits under its ladder, so it names only the rendering, not the family.
        # It repeats the date, as it repeats rungs and parameters: same checkpoints, and no cell of
        # the table is left blank. The date is the newest rung's month; the setup paragraph gives
        # the full range where a ladder's rungs were released months apart.
        label = r"\quad " + note if note else name
        date = month(dates["ladders"][fam]["last_month"])
        L = "\\Lad" + tag
        # A ladder whose coefficient stays below 0.80 at every wrapper count prints "never", a
        # word that cannot be read as zero wrappers or as a missing value.
        kcell = "never" if gt[key]["k_for_080_unreachable"] else f"{L}K"
        rows.append(
            f"{label} & {date} & {L}N & {L}Span & {L}Rlo{{}} to {L}Rhi"
            f" & {L}Pos & {L}EvalFix/{L}DepFix & {L}Model & {L}Erho & {kcell} \\\\")
    table = "\n".join([
        "% GENERATED by paper_tools/make_ladders.py -- do not edit.",
        r"\begin{tabular}{@{}llcccccccc@{}}",
        r"\toprule",
        r"Ladder & newest & rungs & params (B) & $r$ range & $r>0$ & spanning"
        r" & model & $\Erho$ & $k_{0.80}$ \\",
        r"\midrule",
        *rows,
        r"\bottomrule",
        r"\end{tabular}",
    ])
    (HERE / "ladders_table.tex").write_text(table + "\n", encoding="utf-8")
    print(f"wrote ladder_numbers.tex ({len(m)} macros) and ladders_table.tex "
          f"({len(rows)} rows: {', '.join(k for k, *_ in present)})")


if __name__ == "__main__":
    main()

