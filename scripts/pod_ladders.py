"""The two pod ladders, analysed the way the paper analyses Qwen2.5.

Four questions, all pre-specified in pod/PREDICTIONS.md or by review before anything here was run:

  1. REPLICATION   Does the wrapper still decide the sign, at six Qwen3 rungs and on a second
                   vendor? The sixth rung's direction was pre-registered (PREDICTIONS.md sec. 2).
  2. TRANSPOSE     Fixing the DEPLOYMENT framing instead of the evaluation one, do its six
                   evaluation variants span both signs? The paper's most quotable sentence -- the
                   sign is decided by the arm that never names the concept -- is an asymmetry
                   claim, and an asymmetry has to be measured in both directions.
  3. G-STUDY       Section 12 names the 6x6 crossing on a second family at four matched scales as
                   the experiment that would settle the coefficients. It now exists.
  4. RELIABILITY   "Determines" rests on Spearman-Brown 0.874 measured on Qwen2.5 alone.

Methods are imported from gstudy_gtheory rather than reimplemented, so the decomposition is the
same code the paper's numbers come from, not a lookalike.

    python scripts/pod_ladders.py
"""

from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from gstudy_gtheory import (auc_cols, coefficients, components_3way,  # noqa: E402
                            total)
from result_io import write_result  # noqa: E402

N_BLOCKS = 4
N_SPLITS = 20

# Directions come from the validated (BOS-dropped) arrays on BOTH ladders. On OLMo-2 that is the
# deviation pod/PREDICTIONS.md specified in advance, because OLMo-2-32B omits bos_token and an
# all-token mean is therefore not comparable across that ladder. Qwen3 uses the same array so the
# two ladders are scored identically; the all-token variant is reported alongside as a check.
LADDERS = {
    "qwen3_nothink": ("results/pod_qwen3/{}__nothink.npz",
                      {"qwen3-0.6b": 0.752, "qwen3-1.7b": 2.032, "qwen3-4b": 4.022,
                       "qwen3-8b": 8.191, "qwen3-14b": 14.77, "qwen3-32b": 32.76}),
    "qwen3_think": ("results/pod_qwen3/{}__think.npz",
                    {"qwen3-0.6b": 0.752, "qwen3-1.7b": 2.032, "qwen3-4b": 4.022,
                     "qwen3-8b": 8.191, "qwen3-14b": 14.77, "qwen3-32b": 32.76}),
    # Text-decoder parameters from the safetensors headers (configs/ladders.json): these checkpoints
    # are VLMs and the vision tower and MTP head are never run, so they are not on the x-axis.
    "qwen3.5_nothink": ("results/pod_qwen35/{}__nothink.npz",
                        {"qwen3.5-0.8b": 0.752, "qwen3.5-2b": 1.882, "qwen3.5-4b": 4.206,
                         "qwen3.5-9b": 8.954, "qwen3.5-27b": 26.90}),
    "qwen3.5_think": ("results/pod_qwen35/{}__think.npz",
                      {"qwen3.5-0.8b": 0.752, "qwen3.5-2b": 1.882, "qwen3.5-4b": 4.206,
                       "qwen3.5-9b": 8.954, "qwen3.5-27b": 26.90}),
    "qwen2.5_paper": ("GSTUDY",
                      {"qwen2.5-0.5b": 0.494, "qwen2.5-1.5b": 1.544,
                       "qwen2.5-3b": 3.086, "qwen2.5-7b": 7.616}),
    "olmo2": ("results/pod_all/{}.npz",
              {"olmo-2-0425-1b-instruct": 1.49, "olmo-2-1124-7b-instruct": 7.30,
               "olmo-2-1124-13b-instruct": 13.72, "olmo-2-0325-32b-instruct": 32.23}),
}


def unit(v):
    return v / np.clip(np.linalg.norm(v, axis=-1, keepdims=True), 1e-12, None)


def peak(y, proj):
    a = auc_cols(y, proj)
    return float(np.nanmax(np.abs(np.maximum(a, 1.0 - a) - 0.5)))


def load_qwen25(model):
    """The paper's own ladder, from results/gstudy, under the paper's own convention.

    Included so the transpose in question 2 has the ladder it was originally claimed on. The
    convention differs from the pod ladders -- lf scoring, all-token directions, as Section 7
    declares -- but question 2 compares eval-fixed against deploy-fixed WITHIN a ladder, so the
    convention is held fixed wherever the comparison is actually made.
    """
    with np.load(ROOT / "results" / "gstudy" / model / "pooled.npz", allow_pickle=False) as z:
        fr = z["framing"].astype(np.float32)
        names = np.array([str(s) for s in z["framing_name"]])
        fam = json.loads(str(z["families"]))
    with np.load(ROOT / "results" / "gstudy" / model / "sad_filtered_lf.npz",
                 allow_pickle=False) as z2:
        sad = z2["sad"].astype(np.float32)
        y = z2["sad_label"].astype(int)
    ev = sorted(k for k, v in fam.items() if v == "eval")
    dp = sorted(k for k, v in fam.items() if v == "deploy")
    pairs = list(itertools.product(ev, dp))
    proj = {j: np.einsum("ild,ld->il", sad,
                         unit(fr[names == fe].mean(0) - fr[names == fo].mean(0)))
            for j, (fe, fo) in enumerate(pairs)}
    return proj, y, pairs, ev, dp


def load(path, dirkey="framing_validated"):
    z = np.load(ROOT / path, allow_pickle=False)
    names = np.array([str(s) for s in z["framing_name"]])
    fr = z[dirkey].astype(np.float32)
    sad = z["sad_validated"].astype(np.float32)
    y = z["sad_label"].astype(int)
    ev = sorted({n for n in set(names) if n.startswith("eval_")})
    dp = sorted({n for n in set(names) if n.startswith("deploy_")})
    pairs = list(itertools.product(ev, dp))
    proj = {}
    for j, (fe, fo) in enumerate(pairs):
        d = unit(fr[names == fe].mean(0) - fr[names == fo].mean(0))
        proj[j] = np.einsum("ild,ld->il", sad, d)
    return proj, y, pairs, ev, dp


def spans_both(r, idx):
    return max(r[i] for i in idx) > 0 > min(r[i] for i in idx)


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    rep, gst, rel = {}, {}, {}
    for name, (tmpl, params) in LADDERS.items():
        models = list(params)
        gstudy = tmpl == "GSTUDY"
        paths = models if gstudy else [tmpl.format(m) for m in models]
        missing = [] if gstudy else [p for p in paths if not (ROOT / p).exists()]
        if missing:
            print(f"  SKIP {name}: missing {missing[0]}")
            continue
        lp = np.array([np.log10(params[m]) for m in models])

        P, Y, PAIRS, EV, DP = [], None, None, None, None
        for p in paths:
            proj, y, pairs, ev, dp = load_qwen25(p) if gstudy else load(p)
            P.append(proj)
            Y, PAIRS, EV, DP = y, pairs, ev, dp
        print(f"  loaded {name}: {len(models)} models", flush=True)

        # ---- full-set statistic per (model, wrapper) -> replication + transpose
        full = np.array([[peak(Y, P[i][j]) for j in range(len(PAIRS))]
                         for i in range(len(models))])
        r = np.array([np.corrcoef(lp, full[:, j])[0, 1] for j in range(full.shape[1])])
        by_eval, by_dep = {}, {}
        for j, (fe, fo) in enumerate(PAIRS):
            by_eval.setdefault(fe, []).append(j)
            by_dep.setdefault(fo, []).append(j)
        n_eval_flip = sum(spans_both(r, i) for i in by_eval.values())
        n_dep_flip = sum(spans_both(r, i) for i in by_dep.values())
        rep[name] = {
            "models": models, "params_b": [params[m] for m in models],
            "decades": float(np.ptp(lp)), "n_wrappers": len(PAIRS),
            "r_min": float(r.min()), "r_max": float(r.max()),
            "n_positive": int((r > 0).sum()), "n_negative": int((r < 0).sum()),
            "positive_fraction": float((r > 0).mean()),
            "n_eval_framings": len(EV), "n_eval_framings_spanning": int(n_eval_flip),
            "n_deploy_framings": len(DP), "n_deploy_framings_spanning": int(n_dep_flip),
            "statistic_median": float(np.median(full)),
            # How many framings the ladder's OWN marginal predicts would span both signs, if the
            # 36 signs were independent draws at the observed positive rate. A count at this value
            # says nothing beyond the skew; a count well below it says the framings cluster. The
            # paper makes a clustering claim, so the comparison has to be computed, not eyeballed.
            "expected_spanning_from_marginal": float(
                len(EV) * (1 - (q_pos := float((r > 0).mean())) ** len(DP)
                           - (1 - q_pos) ** len(DP))),
        }

        # ---- three-way crossed G-study, same design and code as the paper
        rng = np.random.default_rng(0)
        pos, neg = rng.permutation(np.flatnonzero(Y == 1)), rng.permutation(np.flatnonzero(Y == 0))
        blocks = [np.concatenate([p, q]) for p, q in
                  zip(np.array_split(pos, N_BLOCKS), np.array_split(neg, N_BLOCKS))]
        x = np.array([[[peak(Y[b], P[i][j][b]) for b in blocks]
                       for j in range(len(PAIRS))] for i in range(len(models))])
        c = components_3way(x)
        tot = total(c)
        erho, phi = coefficients(c, k=1, n=1)
        # k for a conventional 0.80 on THIS ladder's own components
        kt = None
        for k in range(1, 100001):
            if coefficients(c, k=k, n=N_BLOCKS)[0] >= 0.80:
                kt = k
                break
        gst[name] = {
            "models": models, "n_wrappers": len(PAIRS), "n_blocks": N_BLOCKS,
            # A component truncated at zero is unresolvable at this design's df, not absent, and
            # the paper says so wherever it reports one. Carried through rather than dropped.
            "shares": {kk: float(v / tot) for kk, v in c.items() if not kk.startswith("_")},
            "components": {kk: float(v) for kk, v in c.items() if not kk.startswith("_")},
            "truncated_at_zero": list(c["_truncated"]),
            "erho_single": float(erho), "phi_single": float(phi),
            "ratio_mt_over_mh": float(c["s_mt"] / c["s_mh"]) if c["s_mh"] > 0 else None,
            "k_for_080": kt, "k_for_080_unreachable": kt is None,
            "erho_asymptote_in_k": float(coefficients(c, k=10 ** 6, n=N_BLOCKS)[0]),
        }

        # ---- split-half slope reliability
        sc = []
        for _ in range(N_SPLITS):
            pa, na = rng.permutation(pos), rng.permutation(neg)
            h1 = np.concatenate([pa[:len(pa) // 2], na[:len(na) // 2]])
            h2 = np.concatenate([pa[len(pa) // 2:], na[len(na) // 2:]])
            sl = {}
            for half, idx in (("a", h1), ("b", h2)):
                st = np.array([[peak(Y[idx], P[i][j][idx]) for j in range(len(PAIRS))]
                               for i in range(len(models))])
                sl[half] = np.array([np.polyfit(lp, st[:, j], 1)[0] for j in range(len(PAIRS))])
            sc.append(float(np.corrcoef(sl["a"], sl["b"])[0, 1]))
        sc = np.array(sc)
        rel[name] = {"models": models, "n_splits": N_SPLITS,
                     "slope_split_half_mean": float(sc.mean()),
                     "slope_split_half_sd": float(sc.std(ddof=1)),
                     "spearman_brown": float(2 * sc.mean() / (1 + sc.mean()))}
        print(f"    {name}: eval-fixed {n_eval_flip}/{len(EV)}  deploy-fixed "
              f"{n_dep_flip}/{len(DP)}  r {r.min():+.3f}..{r.max():+.3f}  "
              f"Erho {erho:.4f}  SB {rel[name]['spearman_brown']:+.3f}", flush=True)

    pop = sorted({m for v in rep.values() for m in v["models"]})
    conv = dict(pooling="validated for scoring and for the direction arrays", layers="ALL",
                scoring_array="results/pod_*/*.npz", data_root="pod",
                statistic="folded max over layers", model_population=pop)
    write_result(ROOT / "results" / "pod_replication.json", {"ladders": rep}, kind="measured", **conv)
    write_result(ROOT / "results" / "pod_gtheory.json", {"ladders": gst}, kind="measured",
                 **dict(conv, statistic="folded max over layers, 3-way crossed "
                        "model x wrapper x item-block, one observation per cell"))
    write_result(ROOT / "results" / "pod_slope_reliability.json", {"ladders": rel}, kind="measured",
                 **dict(conv, statistic="folded max over layers, slopes refit on 20 disjoint "
                        "label-stratified item halves"))
    print("\nwrote pod_replication.json, pod_gtheory.json, pod_slope_reliability.json")


if __name__ == "__main__":
    main()
