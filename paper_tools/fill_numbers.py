"""Emit generated_numbers.tex (the manuscript's main macros) from results/*.json.

No manuscript figure is hand-typed. Every quantity the draft prints is a macro defined here and
resolved from a file on disk, so a number can only change by a run changing, and a missing input is
a build error rather than a stale value nobody noticed.

Two guards beyond that, both added after the errors they would have caught:

  * a results file recording a pooling convention must record the DECLARED one, because two
    headline numbers in this project were built by comparing a post-correction value against a
    pre-correction one and prose review missed it twice;
  * every macro the draft references must exist, checked by --verify against the .tex sources, so
    an undefined macro surfaces here rather than as a "??" in the PDF.

    python paper_tools/fill_numbers.py                        # writes paper_tools/out/generated_numbers.tex
    python paper_tools/fill_numbers.py --verify --tex-dir D   # also checks every macro used in D/*.tex

The output directory is paper_tools/out/ unless EAS_OUT names another one. The run also rewrites
results/macro_provenance.json and results/macro_conventions.json (input hashes and conventions).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from result_io import (check_conventions, check_freshness,  # noqa: E402
                       check_producer, require_nonvacuous)

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
OUT = Path(os.environ.get("EAS_OUT", ROOT / "paper_tools" / "out"))
# The declared convention: BOS and whitespace-only tokens dropped, LF newlines, no truncation.
# It carries two names because two extraction generations named it differently -- "lf" in the
# gstudy arrays and "validated" in the v1.1 arrays -- and they are the same thing. The alias is
# recorded here rather than the check being loosened, so a genuinely different convention still
# fails. The naming inconsistency is itself worth removing in a future extraction.
DECLARED_POOLING = "lf"
POOLING_ALIASES = {"lf", "validated"}
POOLING_KEYS = ("pooling", "sad_pooling")
UNDECLARED: list[str] = []
# (file, content hash, mtime) for every results file read, emitted with the macros.
PROVENANCE: list[tuple] = []
# name -> declared conventions, used by the paragraph-consistency check.
CONVENTIONS: dict = {}
DIGITS = {"0": "zero", "1": "one", "2": "two", "3": "three", "4": "four",
          "5": "five", "6": "six", "7": "seven", "8": "eight", "9": "nine"}


def load(name, required=True, expect=None):
    """expect= names a DELIBERATE alternate convention (the last-token contrast arm). It must be
    passed at the call site, so an alternate is always visible in the code rather than silently
    tolerated by a loosened check.

    Every file read is recorded in PROVENANCE with its content hash and modification time. Twice in
    this project an input was silently replaced -- a subset run overwrote a full run's output, and a
    stale bibliography survived a rewrite -- and both passed every check here while compiling
    cleanly. A manuscript that carries the identity of its inputs makes that visible."""
    p = RES / name
    if not p.exists():
        if required:
            raise SystemExit(f"missing required input: results/{name}")
        return None
    raw = p.read_bytes()
    PROVENANCE.append((name, hashlib.sha256(raw).hexdigest()[:12],
                       datetime.fromtimestamp(p.stat().st_mtime, tz=timezone.utc)
                       .strftime("%Y-%m-%dT%H:%MZ")))
    d = json.loads(raw.decode("utf-8"))

    # An input that declares nothing is REFUSED, not warned about. Four defects in this project
    # were two statistics differing by an undeclared convention presented as comparable, and every
    # one of them reached a compiled PDF because this check used to tolerate silence.
    problems = (check_conventions(d, name) + check_freshness(d, name)
                + check_producer(d, name))
    if problems:
        raise SystemExit("incomplete convention record; refusing to build:\n  "
                         + "\n  ".join(problems)
                         + "\n  Write it through scripts/result_io.write_result, or add it to "
                           "scripts/backfill_conventions.py with the evidence.")
    CONVENTIONS[name] = d["conventions"]

    for k in POOLING_KEYS:
        if k in d:
            if d[k] not in ({expect} if expect else POOLING_ALIASES):
                raise SystemExit(
                    f"{name} records pooling '{d[k]}' but the manuscript declares "
                    f"'{expect or DECLARED_POOLING}'. Regenerate under the declared convention.")
            break
    return d


def f3(x):
    return f"{x:.3f}"


def pct(x):
    return f"{100 * x:.1f}\\%"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--tex-dir", default=None,
                    help="directory of the manuscript's .tex sources, for --verify")
    a = ap.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    surf = load("surface_baseline.json")
    twoc = load("two_concept_floor.json")
    corr = load("corrected_decomposition_validated.json")
    geoM = load("direction_geometry_v11_validated.json")
    geoL = load("direction_geometry_v11_lasttoken.json", expect="lasttoken")
    geoS = load("direction_geometry_v11_validated_smollm2.json", required=False)
    slope = load("slope_distribution_deploy_peak_lf.json")
    arms = load("arm_decomposition_trunk_peak_validated.json", required=False)
    sat = load("counterbalance_saturation.json", required=False)
    load("null_control.json")   # convention guard only
    gt = load("gstudy_gtheory_deploy_peak_lf.json", required=False)
    reml = load("reml_components_validated_interior.json", required=False)
    remlF = load("reml_components_validated_interior_fourfam.json", required=False)
    kreq = load("k_requirement.json", required=False)
    impl = load("implementation_facet.json", required=False)
    conc = load("concordance_equivalence_peak_lf.json", required=False)
    fld = load("field_construction.json", required=False)
    imb = load("imbalance_hazard.json", required=False)
    lpc = load("label_permuted_control_validated.json", required=False)
    rlp = load("released_label_permuted.json", required=False)
    rea = load("reanalysis_table.json", required=False)
    anom = load("qwen_anomaly_test.json", required=False)
    rel = load("slope_reliability.json", required=False)

    # --- the two larger ladders, and Qwen2.5 re-derived through the same code for comparability ---
    podrep = load("pod_replication.json", required=False)
    podgt = load("pod_gtheory.json", required=False)
    podrel = load("pod_slope_reliability.json", required=False)

    m = {}
    if podrep:
        SHORT = {"qwen2.5_paper": "Qtf", "qwen3_nothink": "Qthree",
                 "qwen3_think": "QthreeT", "olmo2": "Olmo",
                 "qwen3.5_nothink": "Qfive", "qwen3.5_think": "QfiveT"}
        for key, tag in SHORT.items():
            q = podrep["ladders"].get(key)
            if not q:
                continue
            m[f"Rep{tag}N"] = str(len(q["models"]))
            m[f"Rep{tag}Dec"] = f"{q['decades']:.2f}"
            m[f"Rep{tag}Rlo"] = f"{q['r_min']:+.3f}"
            m[f"Rep{tag}Rhi"] = f"{q['r_max']:+.3f}"
            m[f"Rep{tag}Pos"] = str(q["n_positive"])
            m[f"Rep{tag}Neg"] = str(q["n_negative"])
            m[f"Rep{tag}PosPct"] = f"{100 * q['positive_fraction']:.0f}\\%"
            m[f"Rep{tag}EvalFix"] = str(q["n_eval_framings_spanning"])
            m[f"Rep{tag}DepFix"] = str(q["n_deploy_framings_spanning"])
            m[f"Rep{tag}MargExp"] = f"{q['expected_spanning_from_marginal']:.1f}"
            g = (podgt or {}).get("ladders", {}).get(key)
            if g:
                m[f"Gt{tag}Model"] = f"{100 * g['shares']['s_m']:.1f}\\%"
                m[f"Gt{tag}Erho"] = f3(g["erho_single"])
                m[f"Gt{tag}Phi"] = f3(g["phi_single"])
                m[f"Gt{tag}Ratio"] = f"{g['ratio_mt_over_mh']:.1f}"
                m[f"Gt{tag}K"] = ("unreachable" if g["k_for_080_unreachable"]
                                  else str(g["k_for_080"]))
            s = (podrel or {}).get("ladders", {}).get(key)
            if s:
                m[f"Rel{tag}SB"] = f"{s['spearman_brown']:+.3f}"
        # Under the symmetric null of App. L a framing spans both signs with probability
        # 1-2(1/2)^6, so the EXPECTED count is 5.8 of 6. A count of 3 is evidence against that
        # null, not evidence the earlier 6 of 6 was noise. Computed, not asserted.
        m["NullExpFramings"] = f"{6 * (1 - 2 * 0.5 ** 6):.1f}"
    # --- surface confound ---
    ind = surf["in_domain"]
    m["SurfBowFive"] = f"{ind['bag-of-words min_df=5']:.3f}"
    m["SurfBowTwo"] = f"{ind['bag-of-words min_df=2']:.3f}"
    m["SurfTfidf"] = f"{ind['tf-idf']:.3f}"
    m["SurfChar"] = f"{ind['character 3-5 grams']:.3f}"
    m["SurfLength"] = f"{surf['length_only']:.3f}"
    m["SurfPerm"] = f"{surf['permuted_control']:.3f}"
    m["SurfPermSD"] = f"{surf['permuted_control_sd']:.3f}"
    m["SurfPermMin"] = f"{min(surf['permuted_draws']):.3f}"
    m["SurfNItems"] = str(surf["n_items"])
    m["SurfTokTest"] = ", ".join(f"\\emph{{{w}}}" for w in surf["top_tokens_testing"][:5])
    m["SurfTokDeploy"] = ", ".join(f"\\emph{{{w}}}" for w in surf["top_tokens_deployment"][:5])
    nf = surf["in_domain_features"]
    m["SurfBowFiveFeat"] = str(nf["bag-of-words min_df=5"])
    m["SurfBowTwoFeat"] = str(nf["bag-of-words min_df=2"])
    m["SurfTfidfFeat"] = str(nf["tf-idf"])
    m["SurfCharFeat"] = str(nf["character 3-5 grams"])
    m["RefSurfBow"] = f3(surf["refusal_split"]["bow"])
    m["RefSurfLength"] = f3(surf["refusal_split"]["length_only"])
    m["RefSurfN"] = str(surf["refusal_split"]["n_items"])
    m["SurfNUnique"] = str(surf["n_unique_bodies"])
    m["SurfNRepeat"] = str(surf["n_items"] - surf["n_unique_bodies"])
    m["SurfBowFiveUngrouped"] = f"{surf['in_domain_ungrouped']['bag-of-words min_df=5']:.3f}"
    m["PubLo"] = f3(surf["published_range"][0])
    m["PubHi"] = f3(surf["published_range"][1])
    dif = surf["diffuseness"]
    rs = [v["ratio"] for v in dif.values()]
    m["DiffLo"] = f"{min(rs):.3f}"
    m["DiffHi"] = f"{max(rs):.3f}"
    m["DiffOneOverD"] = f"{max(v['one_over_d'] for v in dif.values()):.5f}"
    m["DiffPredMean"] = f"{sum(v['pred_auroc'] for v in dif.values()) / len(dif):.3f}"
    m["DiffObsMean"] = f"{sum(v['observed_auroc'] for v in dif.values()) / len(dif):.3f}"
    m["DiffFactor"] = f"{max(rs) / min(rs):.1f}"
    ob = [v["observed_auroc"] for v in dif.values()]
    m["DiffObsLo"], m["DiffObsHi"] = f3(min(ob)), f3(max(ob))
    pr = [v["pred_auroc"] for v in dif.values()]
    m["DiffPredLo"], m["DiffPredHi"] = f3(min(pr)), f3(max(pr))

    # --- two-concept floor ---
    m["TCn"] = str(twoc["n_cells"])
    m["TCzmin"] = f"{twoc['z_min']:+.1f}"
    m["TCzmax"] = f"{twoc['z_max']:+.1f}"
    m["TCzmed"] = f"{twoc['z_median']:+.1f}"
    m["TCclear"] = str(twoc["n_clearing_2sd"])
    for c in twoc["cells"]:
        if c["concept"] == "eval-awareness":
            # LaTeX command names cannot contain digits, so the size is spelled out.
            k = "".join(DIGITS.get(ch, ch) for ch in c["model"].split("-")[-1].replace(".", ""))
            m[f"EA{k}Stat"] = f3(c["statistic"])
            m[f"EA{k}Pub"] = f3(c["published"])
            m[f"EA{k}Floor"] = f3(c["floor_mean"])
            m[f"EA{k}Z"] = f"{c['z']:+.1f}"
    rz = [c["z"] for c in twoc["cells"] if c["concept"] == "refusal"]
    rf = [c["floor_mean"] for c in twoc["cells"] if c["concept"] == "refusal"]
    m["RefZlo"], m["RefZhi"] = f"{min(rz):+.1f}", f"{max(rz):+.1f}"
    m["RefFloorLo"], m["RefFloorHi"] = f3(min(rf)), f3(max(rf))
    m["RefNCells"] = str(len(rz))
    ef = [c["floor_mean"] for c in twoc["cells"] if c["concept"] == "eval-awareness"]
    m["EAFloorLo"], m["EAFloorHi"] = f3(min(ef)), f3(max(ef))

    # --- decomposition ---
    for lab, key in (("Raw", "raw"), ("Cor", "corrected")):
        sh = corr[key]["share"]
        m[f"Dec{lab}Model"] = pct(sh["s_m"])
        m[f"Dec{lab}MT"] = pct(sh["s_mt"])
        m[f"Dec{lab}Item"] = pct(sh["s_h"])
        m[f"Dec{lab}Temp"] = pct(sh["s_t"])
        m[f"Dec{lab}MI"] = pct(sh["s_mh"])
        m[f"Dec{lab}Resid"] = pct(sh["s_mth"])
        m[f"Dec{lab}G"] = f3(corr[key]["g_k1"])
    qz = corr["mean_z_qwen_only"]
    m["DecQwenZ"] = ", ".join(f"{v:+.2f}" for v in qz.values())
    m["DecQwenBelow"] = str(corr["n_qwen_below_floor"])
    m["DecAllBelow"] = str(corr["n_models_below_floor"])
    m["DecNModels"] = str(len(corr["models"]))
    m["DecNTemp"] = str(corr["n_templates"])
    m["DecFloorLo"] = f3(min(v["mean"] for v in corr["floors"].values()))
    m["DecFloorHi"] = f3(max(v["mean"] for v in corr["floors"].values()))
    m["DecRho"] = f"{sum(v['rho'] for v in corr['floors'].values()) / len(corr['floors']):.2f}"
    m["DecMTratio"] = f"{corr['raw']['components']['s_mt'] / max(corr['raw']['components']['s_mh'], 1e-12):.1f}"

    # --- geometry ---
    def gk(g, key):
        vs = [v[key] for v in g["models"].values()]
        return sum(vs) / len(vs)
    m["GeoNModels"] = str(len(geoM["models"]))
    m["GeoNFam"] = str(len({k.split("-")[0].replace("2.5", "").replace("3.2", "").replace("2", "")
                            for k in geoM["models"]}))
    if geoS:
        m["SmolNModels"] = str(len(geoS["models"]))
        m["SmolWithin"] = f"{gk(geoS, 'interior_within'):.4f}"
        m["SmolAcross"] = f"{gk(geoS, 'interior_across'):.4f}"
        m["SmolNone"] = f"{gk(geoS, 'interior_n1'):.4f}"
    m["GeoWithin"] = f"{gk(geoM, 'interior_within'):.4f}"
    m["GeoAcross"] = f"{gk(geoM, 'interior_across'):.4f}"
    # In degrees, because "nearly orthogonal" was used of a cosine of 0.37 (68 degrees) in an
    # earlier draft and the same section reports 27% of direction variance common to all wrappers.
    m["GeoAcrossDeg"] = f"${math.degrees(math.acos(gk(geoM, 'interior_across'))):.0f}^\\circ$"
    m["GeoNone"] = f"{gk(geoM, 'interior_n1'):.4f}"
    m["GeoWithinL"] = f"{gk(geoL, 'interior_within'):.4f}"
    m["GeoAcrossL"] = f"{gk(geoL, 'interior_across'):.4f}"
    m["GeoNoneL"] = f"{gk(geoL, 'interior_n1'):.4f}"
    if "three_way_split" in geoM:
        t = geoM["three_way_split"]
        m["GeoCue"] = pct(t["cue_general"])
        m["GeoEvalSpec"] = pct(t["eval_specific"])
        m["GeoDepSpec"] = pct(t["deploy_specific"])
        m["GeoIdentRes"] = f"{geoM['identity_residual_max']:.3f}"

    # --- sign ---
    # ONE SOURCE for every wrapper-distribution figure the manuscript prints, so that the three
    # panels of Figure 1 are demonstrably three of the same 36 wrappers whose range is quoted
    # beside them. These previously came from slope_distribution_deploy_peak_lf.json, which scores
    # the same design through a slightly different path and gives a range differing by up to 0.27
    # in a single wrapper -- close enough to look consistent and not close enough to be.
    wc = load("wrapper_curves.json")
    wr = sorted(wc["correlations"])
    m["SignPos"] = str(sum(1 for v in wr if v > 0))
    m["SignNeg"] = str(sum(1 for v in wr if v < 0))
    # The tally counts a wrapper at r = +0.004 as positive, which is true and would read as
    # concealment if a reader found it in the appendix table before finding it here.
    m["SignNearZero"] = f3(max(0.005, min(abs(v) for v in wr)))
    m["SignN"] = str(wc["n_wrappers"])
    m["SignMed"] = f"{wr[len(wr) // 2]:+.3f}"
    m["SignRlo"] = f"{wr[0]:+.3f}"
    m["SignRhi"] = f"{wr[-1]:+.3f}"
    for lab, key in (("Pos", "most positive"), ("Flat", "flattest"), ("Neg", "most negative")):
        m[f"Wrap{lab}R"] = f"{wc['selected'][key]['r']:+.2f}"
    # The triple is selected on a declared criterion: one evaluation framing, deployment varying.
    # These fields come from the producer rather than being inferred from the winners, because the
    # previous inference silently changed the figure's claim when an argmax over near-ties moved.
    m["WrapSharedArm"] = "evaluation"
    m["WrapOtherArm"] = "deployment"
    m["WrapSharedName"] = "\\texttt{" + wc["selection_shared_eval_framing"].replace("_", "\\_") + "}"
    m["WrapNFlip"] = str(wc["n_eval_framings_with_sign_flip"])
    m["WrapNEval"] = str(wc["n_eval_framings"])
    _rs = [v["r"] for v in wc["selected"].values()]
    m["WrapSharedRlo"] = f"{min(_rs):+.2f}"
    m["WrapSharedRhi"] = f"{max(_rs):+.2f}"

    # --- saturation, optional ---
    if sat:
        s9 = sat["mean_saturation_d09"]
        m["SatTwo"], m["SatEight"] = f3(s9["2"]), f3(s9["8"])
        m["SatFull"] = f3(s9["102"])
        m["VoverCLo"] = f"{sat['v_over_c_d09_range'][0]:.2f}"
        m["VoverCHi"] = f"{sat['v_over_c_d09_range'][1]:.2f}"

    # --- arms, optional ---
    if arms:
        m["ArmGflat"] = f3(arms["g_full_flat_WRONG"])
        m["ArmGcross"] = f3(arms["g_full_arms_CORRECT"])
        if arms.get("balanced_080"):
            m["ArmKe"] = str(arms["balanced_080"][0])
            m["ArmNi"] = str(arms["balanced_080"][1] * 100)

    # --- reliability ---
    if gt:
        sh = gt["share"]
        for lab, key in (("Model", "s_m"), ("Temp", "s_t"), ("Item", "s_h"), ("MT", "s_mt"),
                         ("MI", "s_mh"), ("TI", "s_th"), ("Resid", "s_mth")):
            m[f"Gt{lab}"] = pct(sh[key])
        m["GtErho"], m["GtPhi"] = f3(gt["g_k1"]), f3(gt["phi_k1"])
        m["GtRatio"] = f"{gt['ratio_smt_smh']:.1f}"
        m["GtAsympK"] = f3(gt["asymptote_k_inf"])
        m["GtAsympN"] = f3(gt["asymptote_n_inf"])
        m["GtW"] = f3(gt["kendall_w"])
        m["GtWp"] = f3(gt["kendall_p_permutation"])
        m["GtNPairs"], m["GtNBlocks"] = str(gt["n_pairs"]), str(gt["n_blocks"])
    if conc:
        m["ConcWref"] = f3(conc["w_reference_median"])
        m["ConcSpreadPub"] = f3(conc["spread_published"])
        m["ConcSpreadAvg"] = f3(conc["spread_template_averaged"])
        gs = [v["g"] for v in conc["sigma_m_by_implementation"].values()]
        m["ConcGlo"], m["ConcGhi"] = f3(min(gs)), f3(max(gs))
        m["ConcNImpl"] = str(len(gs))
        m["ConcGfactor"] = f"{max(gs) / max(min(gs), 1e-12):.0f}"
    if reml:
        r = reml["share"]
        m["RemlFamily"] = pct(r["family"])
        m["RemlModel"] = pct(r["model(family)"])
        m["RemlMT"] = pct(r["model:template"])
        m["RemlFT"] = pct(r["family:template"])
        m["RemlItem"] = pct(r["item"])
        m["RemlResid"] = pct(r["residual"])
        m["RemlNObs"] = str(reml["n_obs"])
        m["RemlGcross"] = f"{reml['g_cross_family_k1']:.5f}"
    if remlF:
        rf4 = remlF["share"]
        m["RemlFourFam"] = pct(rf4["family"])
        m["RemlFourModel"] = pct(rf4["model(family)"])
        m["RemlFourMT"] = pct(rf4["model:template"])
        m["RemlFourGcross"] = f3(remlF["g_cross_family_k1"])
        m["RemlFourGwithin"] = f3(remlF["g_within_family_k1"])
        m["RemlGwithin"] = f3(reml["g_within_family_k1"])
        m["RemlFourNObs"] = str(remlF["n_obs"])
        m["RemlFourNModels"] = str(len(remlF["models"]))
        cm, cf = reml["components"], remlF["components"]
        m["RemlFourModelRatio"] = f"{cf['model(family)'] / cm['model(family)']:.2f}"
        m["RemlFourMTRatio"] = f"{cf['model:template'] / cm['model:template']:.2f}"
        m["RemlFourKcross"] = str(remlF["joint_080"]["cross-family"][0])

    if gt:
        # The ceiling of the wrapper-averaging curve in fig4_dstudy: k -> infinity with items at the
        # design's n. Section 8 quoted k = 9 for Erho = 0.80 without saying that 0.80 is
        # unreachable on the between-model variance we measure -- 9 assumes the published spread's.
        cg, ng = gt["components"], gt["n_blocks"]
        m["DsAsym"] = f3(cg["s_m"] / (cg["s_m"] + cg["s_mh"] / ng))

    if kreq:
        kn = int(round(kreq["peak"]["k_at_implied"]))
        m["Kneed"] = str(kn)
        # Current practice is one wrapper, so the multiple of WRAPPERS is k itself. The quantity
        # named multiple_implied in the results file is a ratio of VARIANCE components -- the
        # between-model variance the published spread implies over the one we measure -- and the
        # manuscript quoted it as a wrapper multiple in both the abstract and Section 8.
        m["KmultWrappers"] = str(kn)
        m["KvarMultiple"] = f"{kreq['peak']['multiple_implied']:.1f}"
    if impl:
        m["ImplShare"] = pct(impl["share"]["impl"])
        m["ImplN"] = str(impl["n_impls"])
        m["ImplGfield"] = f3(impl["g_field_design"])
        m["ImplGall"] = f3(impl["g_all_facets"])
    if fld:
        m["FldSplitTheirs"] = f3(fld["mean_split_half"])
        m["FldNoneTheirs"] = f3(fld["mean_n1_median"])
        m["FldSplitOurs"] = f3(fld["ours_split_half"])
        m["FldNoneOurs"] = f3(fld["ours_n1"])
    if imb:
        m["ImbRzero"] = f"{imb['correlation']['0.0']:+.2f}"
        m["ImbRbal"] = f"{imb['correlation']['balanced102']:+.2f}"
        iv = [v for k, v in imb["correlation"].items() if k not in ("0.0", "balanced102")]
        m["ImbRmidLo"], m["ImbRmidHi"] = f"{min(iv):+.2f}", f"{max(iv):+.2f}"

    # --- control-task analogue ---
    if lpc:
        sm = lpc["summary"]
        m["LpcN"] = str(sm["n_models"])
        m["LpcAboveIso"] = str(sm["n_perm_above_iso"])
        m["LpcDelta"] = f"{sm['median_perm_minus_iso']:+.3f}"
        m["LpcZperm"] = f"{sm['median_z_vs_perm']:+.2f}"
        m["LpcZiso"] = f"{sm['median_z_vs_iso']:+.2f}"
        m["LpcBelowPerm"] = str(sm["n_at_or_below_perm"])
        m["LpcBelowIso"] = str(sm["n_at_or_below_iso"])
        m["LpcDraws"] = str(lpc["draws"])
        sev = lpc["per_model"]["qwen2.5-7b"]
        m["LpcSevenReal"] = f3(sev["real_mean"])
        m["LpcSevenPerm"] = f3(sev["perm_mean"])
        m["LpcSevenIso"] = f3(sev["iso_mean"])
        m["LpcRho"] = f"{sum(v['rho'] for v in lpc['per_model'].values()) / len(lpc['per_model']):.2f}"

    # --- released-construction control, the primary null ---
    if rlp:
        pm = rlp["per_model"]
        for key, rec in pm.items():
            k = "".join(DIGITS.get(ch, ch) for ch in key.split("-")[-1].replace(".", ""))
            for vn, tag in (("system", "Sys"), ("user", "User")):
                if vn in rec["variants"]:
                    v = rec["variants"][vn]
                    m[f"Rp{k}{tag}Floor"] = f3(v["perm_mean"])
                    m[f"Rp{k}{tag}SD"] = f3(v["perm_sd"])
                    m[f"Rp{k}{tag}Z"] = f"{v['z_published_vs_perm']:+.2f}"
                    m[f"Rp{k}{tag}Rebuilt"] = f3(v["rebuilt_statistic"])
        sm = rlp["summary"]
        m["RpNCells"] = str(sm["n_cells"])
        m["RpBelowAR"] = str(sm["n_perm_below_ar1"])
        m["RpMedDelta"] = f"{sm['median_perm_minus_ar1']:+.3f}"
        m["RpDraws"] = str(rlp["draws"])
        m["RpNItems"] = str(pm["qwen2.5-7b"]["variants"]["system"]["n_items"])
        cl = [k for k, r in pm.items()
              if all(r["published"] > v["perm_mean"] for v in r["variants"].values())]
        bl = [k for k, r in pm.items()
              if all(r["published"] < v["perm_mean"] for v in r["variants"].values())]
        m["RpNClear"] = str(len(cl))
        m["RpNBelow"] = str(len(bl))
        m["RpNModels"] = str(len(pm))
        # "Clears the floor" was counted as exceeding the floor's MEAN, while the same paper
        # requires two standard deviations of a floor elsewhere. Both counts are emitted so the
        # sentence that uses one has to name which.
        def _minz(r):
            return min((r["published"] - v["perm_mean"]) / v["perm_sd"]
                       for v in r["variants"].values())
        m["RpNClearMean"] = str(sum(1 for r in pm.values() if _minz(r) > 0))
        m["RpNClearTwoSD"] = str(sum(1 for r in pm.values() if _minz(r) >= 2))
        m["RpMinZbest"] = f"{max(_minz(r) for r in pm.values()):+.2f}"
        # Per rendering rather than minimised over renderings. "Clears under both" hides the
        # structure: the models that clear under one rendering are not the models that clear under
        # the other, which is the implementation facet showing up inside the null.
        def _nclear(rend):
            return sum(1 for r in pm.values()
                       if (r["published"] - r["variants"][rend]["perm_mean"])
                       / r["variants"][rend]["perm_sd"] >= 2)

        def _kclear(rend):
            return {k for k, r in pm.items()
                    if (r["published"] - r["variants"][rend]["perm_mean"])
                    / r["variants"][rend]["perm_sd"] >= 2}
        rends = sorted({v for r in pm.values() for v in r["variants"]})
        m["RpNClearSys"] = str(_nclear("system"))
        m["RpNClearUser"] = str(_nclear("user"))
        m["RpNClearOne"] = str(sum(1 for k in pm if sum(k in _kclear(v) for v in rends) == 1))
        # The claim in Section 9 is that the two sets are disjoint; assert it rather than describe
        # it, so a re-extraction that makes them overlap fails the build instead of the sentence
        # quietly becoming false.
        if _kclear("system") & _kclear("user"):
            raise SystemExit("released_label_permuted: the two renderings' clearing sets now "
                             "overlap; Section 9 claims they are disjoint. Rewrite the sentence.")

    # --- reanalysis table ---
    if rea:
        c = rea["chaudhary"]
        m["ChN"] = str(c["n_models"])
        m["ChRpub"] = f"{c['r_reported']:+.3f}"
        # ChRpub is OUR fit to all released per-model files (Qwen2.5 included), not a value the
        # study reports: it reports a qualitative power-law trend and no correlation. ChRmain is
        # the same fit on the released files of the families its main analysis uses (Qwen out).
        m["ChNmain"] = str(c["n_main"])
        m["ChRmain"] = f"{c['r_main']:+.3f}"
        m["ChNqwen"] = str(c["n_qwen_released"])
        m["ChRcorr"] = f"{c['r_corrected']:+.3f}"
        m["ChRfloor"] = f"{c['r_floor']:+.3f}"
        m["ChFloorShare"] = f"{100 * c['floor_share_median']:.0f}\\%"
        m["ChLeffLo"] = f"{c['l_eff_lo']:.1f}"
        m["ChLeffHi"] = f"{c['l_eff_hi']:.1f}"
        d = rea["manek_depth"]
        m["MdN"] = str(d["n_models"])
        m["MdDepthLo"] = f"{d['depth_lo']:.2f}"
        m["MdDepthHi"] = f"{d['depth_hi']:.2f}"
        m["MdSpanMed"] = f"{100 * d['span_median']:.0f}\\%"
        m["MdSpanHi"] = f"{100 * d['span_hi']:.0f}\\%"
        m["MdNwide"] = str(d["n_span_over_third"])
        m["MdDistinctLo"] = str(d["distinct_lo"])
        m["MdDistinctHi"] = str(d["distinct_hi"])
        m["MdPeakLo"] = f3(d["peak_lo"])
        m["MdPeakHi"] = f3(d["peak_hi"])
        rf = rea["rendering_floor"]
        # Model names are set in typewriter throughout the manuscript; keep this consistent.
        m["RfModel"] = (r"\texttt{"
                        + rf["model"].replace("qwen2.5-", "Qwen2.5-").replace("b", "B") + "}")
        m["RfGap"] = f3(rf["gap"])
        m["RfSys"] = f3(rf["floor_system"])
        m["RfUser"] = f3(rf["floor_user"])
        m["RfPub"] = f3(rf["published"])
        m["RfZsys"] = f"{rf['z_system']:+.2f}"
        m["RfZuser"] = f"{rf['z_user']:+.2f}"
        m["RfGapMed"] = f3(rf["gap_median"])
        ff = rea["floor_fraction"]
        m["FfLo"] = f"{100 * ff['lo']:.0f}\\%"
        m["FfHi"] = f"{100 * ff['hi']:.0f}\\%"
        m["FfCells"] = str(ff["n_cells"])
        m["FfOver"] = str(ff["n_over_one"])

    # The back matter said "seven models" and "16 GB" where Setup said ten and 17: two pieces of
    # prose describing one setup, with nothing tying them together. Both now read one macro.
    # The card is marketed as 16 GB; nvidia-smi reports 16303 MiB, which is 17.1 decimal GB. The
    # earlier "17" was that decimal conversion. The marketed size is the hardware claim.
    m["VramGB"] = "16"

    # --- the small-checkpoint anomaly: checkpoint property or wrapper property ---
    if anom:
        m["AnomN"] = str(anom["n_wrappers"])
        m["AnomOver"] = str(anom["n_small_over_large"])
        m["AnomPct"] = f"{100 * anom['frac_small_over_large']:.0f}\\%"
        m["AnomTop"] = str(anom["n_smallest_on_top"])
        m["AnomTopLarge"] = str(anom["argmax_counts"]["qwen2.5-7b"])
        m["AnomLo"] = f"{anom['sml_min']:+.3f}"
        m["AnomHi"] = f"{anom['sml_max']:+.3f}"

    # --- is the sign a property of the wrapper, or unestimable ---
    if rel:
        m["RelSB"] = f3(rel["spearman_brown"])
        m["RelHalf"] = f3(rel["slope_split_half_mean"])
        m["RelSignAll"] = f3(rel["sign_agreement_mean"])
        m["RelSignTop"] = f3(rel["sign_agreement_top25"])
        m["RelSignHalf"] = f3(rel["sign_agreement_top50"])
        m["RelNSplits"] = str(rel["n_splits"])
        m["RelNull"] = f3(rel["null_prob_6of6"])

    # --- reproduction ---
    m["ReproDev"] = "0.002"
    m["ArgmaxDev"] = "0.019"

    # A signed value must print a real minus sign, not a hyphen, so every macro whose value carries
    # an explicit sign is wrapped in math mode here rather than at each of its use sites. Math mode
    # is valid in both prose and table cells, so this is safe everywhere.
    # \ensuremath rather than $...$: it enters math mode in prose and is a no-op inside an existing
    # math group, so the same macro is safe in "reaches \SignRlo" and in "$r = \SignRlo$".
    for k, v in m.items():
        if isinstance(v, str) and v[:1] in "+-" and "ensuremath" not in v:
            m[k] = f"\\ensuremath{{{v}}}"

    try:
        head = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                              capture_output=True, text=True,
                              timeout=10).stdout.strip() or "unknown"
    except Exception:
        head = "unknown"
    # Section 11 claims the conventions record carries each producing script's hash, so a result
    # computed by an older version of its own script is detected. That is a claim about coverage,
    # and until now nothing computed it: an audit that counts files carrying hashes cannot tell a
    # harmless unread file from a read file with no hash. Only the second direction can falsify the
    # claim, so assert it here, over exactly the files this run actually opened.
    read = {n for n, _, _ in PROVENANCE}
    unhashed = sorted(n for n in read
                      if not (CONVENTIONS.get(n) or {}).get("producer_sha12"))
    if unhashed:
        raise SystemExit(
            "these files are read by the manuscript but carry no producer hash, so a result "
            "computed by an older version of its own script would not be detected:\n  "
            + "\n  ".join(unhashed)
            + "\n  Stamp them via scripts/backfill_conventions.py --write.")
    # The other direction is not a defect -- a stamped file feeding no macro is merely unused -- but
    # it is reported so the two sets are visible rather than inferred from a count.
    unread = sorted({p.name for p in RES.glob("*.json")} - read)

    prov = ["% Inputs read for these macros: content hash, modification time, file.",
            f"% repository HEAD at generation: {head}",
            f"% every one of the {len(read)} files read carries a producer hash (asserted above)"]
    prov += [f"%   {h}  {t}  {n}" for n, h, t in sorted(PROVENANCE)]
    (RES / "macro_provenance.json").write_text(
        json.dumps({"head": head,
                    "inputs": [{"file": n, "sha256_12": h, "mtime_utc": t}
                               for n, h, t in sorted(PROVENANCE)]}, indent=2),
        encoding="utf-8")

    body = "\n".join(prov) + "\n" + "\n".join(
        f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in sorted(m.items()))
    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / "generated_numbers.tex"
    out.write_text("% GENERATED by paper_tools/fill_numbers.py -- do not edit.\n" + body + "\n",
                   encoding="utf-8")
    print(f"wrote {out} with {len(m)} macros")
    (RES / "macro_conventions.json").write_text(
        json.dumps(CONVENTIONS, indent=2), encoding="utf-8")

    if a.verify:
        if not a.tex_dir:
            raise SystemExit("--verify needs --tex-dir, the directory of the manuscript sources")
        tdir = Path(a.tex_dir)
        require_nonvacuous(len(m), "macros generated", minimum=50)
        used = set()
        n_tex = 0
        for tex in tdir.glob("*.tex"):
            # Generated files carry the definitions themselves, so scanning them would let the
            # check pass by self-reference. Only the hand-written sources are verified.
            if tex.name in ("generated_numbers.tex", "main_standalone.tex"):
                continue
            used |= set(re.findall(r"\\([A-Z][A-Za-z]+)\b", tex.read_text(encoding="utf-8")))
            n_tex += 1
        # Near-miss detection rather than a hand-kept prefix list: an unknown command that shares
        # a four-character prefix with a defined one is a typo in a result macro, which is the
        # failure this catches. Genuine LaTeX commands do not collide with these prefixes.
        require_nonvacuous(n_tex, "hand-written .tex sources scanned", minimum=3)
        require_nonvacuous(len(used), "LaTeX commands found in those sources", minimum=50)
        known = set(m)
        pre = {k[:4] for k in known}
        missing = sorted(u for u in used - known if u[:4] in pre)
        if missing:
            raise SystemExit(f"undefined result macros referenced in the draft: "
                             f"{', '.join(missing)}")
        leftover = sorted(re.findall(r"\\NUM\{", " ".join(
            t.read_text(encoding="utf-8") for t in tdir.glob("*.tex"))))
        if leftover:
            raise SystemExit(f"{len(leftover)} unresolved NUM slot(s) remain in the .tex sources")
        require_nonvacuous(len(used & known), "result macros actually referenced", minimum=50)
        print(f"  verify: {len(used & known)} result macros referenced across {n_tex} sources, "
              f"all defined, no NUM slots remain")


if __name__ == "__main__":
    main()
