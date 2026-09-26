"""Reanalysis of published layerwise evaluation-awareness probe results.

Operates on the released per-item probe scores from the shared
`evaluation-awareness-probing` codebase (Nguyen's repo; Manek's fork adds the layerwise
result CSVs for 11 models). Each CSV is one layer of one model:
`question_id, question, type, max_score, min_score, mean_score, std_score`, 400 balanced
items (200 testing / 200 deployment), so AUROC per layer is exactly recomputable.

The published analyses report point estimates: a peak AUROC per model and the relative depth
at which it occurs, with no uncertainty on either. Three things follow that the original
cannot distinguish, and all are computable here without a GPU:

1. UNCERTAINTY. Bootstrap items to put a CI on each layer's AUROC, and on differences between
   models. A "non-monotonic scaling" pattern requires the model-to-model differences to be
   resolvable; because the same 400 items are scored under every model, the comparison can and
   should be PAIRED, which is the most favourable possible treatment.

2. ARGMAX IDENTIFICATION. The reported "depth at which awareness is most recoverable" is an
   argmax over 15-63 candidate layers. Bootstrapping it shows whether the data identify a
   layer at all, and whether the argmax sits at a boundary -- where a bootstrap interval is
   truncated by construction and narrowness means "still rising at the edge", not "identified".

3. SELECTION BIAS. A peak AUROC is a maximum over layers, so it is upward biased. Choosing the
   layer on half the items and scoring on the other half gives an honest estimate; the gap is
   the bias in the published number.

This is a methodological reanalysis, not a claim that the underlying measurements are wrong:
the AUROC values reproduce exactly. The published caveats about recoverability-vs-use are
unusually careful, and the design issue addressed here is shared across this literature
(including our own earlier analyses, which made the same error twice).

    python scripts/reanalyze_probe_depth.py --root ../eval-awareness-reanalysis/manek/notebooks
"""

from __future__ import annotations

import argparse
import glob
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

# model -> (params_b, n_hidden_layers) for the depth-vs-parameters question
MODELS = {
    "QWEN_0.5B": (0.5, 24), "QWEN_1.5B": (1.5, 28), "QWEN_3B": (3.0, 36),
    "QWEN_7B": (7.0, 28), "QWEN_14B": (14.0, 48), "QWEN_32B": (32.0, 64),
    "GEMMA_2B": (2.0, 26), "GEMMA_9B": (9.0, 42), "GEMMA_27B": (27.0, 46),
    "LLAMA_1B": (1.0, 16), "LLAMA_3B": (3.0, 28),
}
SCORE = "mean_score"


def load_model(root: Path, model: str):
    """-> scores [items, layers], labels [items], layer numbers (sorted)."""
    files = glob.glob(str(root / f"{model}_PROBE_RESULTS" / "probe_stats_layer*.csv"))
    if not files:
        return None
    keyed = {}
    for path in files:
        match = re.search(r"layer(\d+)\.csv$", path)
        if match:
            keyed[int(match.group(1))] = path
    layers = sorted(keyed)
    base = pd.read_csv(keyed[layers[0]]).sort_values("question_id").reset_index(drop=True)
    labels = (base["type"] == "testing").astype(int).to_numpy()
    ids = base["question_id"].to_numpy()
    scores = np.zeros((len(base), len(layers)))
    for j, layer in enumerate(layers):
        frame = pd.read_csv(keyed[layer]).sort_values("question_id").reset_index(drop=True)
        if not np.array_equal(frame["question_id"].to_numpy(), ids):
            raise SystemExit(f"{model} layer {layer}: item order differs")
        scores[:, j] = frame[SCORE].to_numpy()
    return scores, labels, np.array(layers)


def auroc_curve(scores: np.ndarray, labels: np.ndarray, idx: np.ndarray | None = None):
    """Per-layer AUROC, orientation-folded so a probe sign flip is not a failure."""
    if idx is not None:
        scores, labels = scores[idx], labels[idx]
    if labels.min() == labels.max():
        return np.full(scores.shape[1], np.nan)
    out = np.empty(scores.shape[1])
    for j in range(scores.shape[1]):
        a = roc_auc_score(labels, scores[:, j])
        out[j] = max(a, 1.0 - a)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--n-split", type=int, default=500)
    parser.add_argument("--out", default="results/reanalysis_probe_depth.json")
    args = parser.parse_args()
    root = Path(args.root)
    rng = np.random.default_rng(0)

    loaded, report = {}, {"models": []}
    for model in MODELS:
        got = load_model(root, model)
        if got is None:
            print(f"  (missing {model})")
            continue
        loaded[model] = got

    # paired bootstrap: identical item set across models -> resample items once per replicate
    n_items = {m: v[1].size for m, v in loaded.items()}
    common = set.intersection(*[set(range(n)) for n in n_items.values()]) if loaded else set()
    n_common = len(common)
    print(f"{len(loaded)} models; {n_common} items shared\n")

    boot_curves: dict[str, np.ndarray] = {}
    for model, (scores, labels, layers) in loaded.items():
        params, depth = MODELS[model]
        curve = auroc_curve(scores, labels)
        n_layers = curve.size
        rel = (np.arange(n_layers) + 1) / n_layers
        am = int(np.nanargmax(curve))
        boundary = am == 0 or am >= n_layers - 1

        boots = np.empty((args.n_boot, n_layers))
        for b in range(args.n_boot):
            boots[b] = auroc_curve(scores, labels, rng.integers(0, labels.size, labels.size))
        boot_curves[model] = boots
        peak_lo, peak_hi = np.nanpercentile(boots.max(axis=1), [2.5, 97.5])
        ams = np.nanargmax(boots, axis=1)
        am_lo, am_hi = np.percentile(rel[ams], [2.5, 97.5])
        frac_bd = float(np.mean((ams == 0) | (ams >= n_layers - 1)))

        # tilt of the AUROC curve over relative depth
        tilt = float(np.polyfit(rel, curve, 1)[0])
        tilts = np.array([np.polyfit(rel, boots[b], 1)[0] for b in range(args.n_boot)])
        t_lo, t_hi = np.percentile(tilts, [2.5, 97.5])

        # selection bias in the reported peak: choose layer on half the items, score on rest
        naive, honest = [], []
        for _ in range(args.n_split):
            perm = rng.permutation(labels.size)
            a, b_ = perm[: labels.size // 2], perm[labels.size // 2:]
            ca, cb = auroc_curve(scores, labels, a), auroc_curve(scores, labels, b_)
            pick = int(np.nanargmax(ca))
            naive.append(float(np.nanmax(ca)))
            honest.append(float(cb[pick]))
        bias = float(np.mean(naive) - np.mean(honest))

        entry = {
            "model": model, "params_b": params, "hidden_layers": depth,
            "n_layers_scored": n_layers, "n_items": int(labels.size),
            "peak_auroc": float(np.nanmax(curve)),
            "peak_auroc_ci": [float(peak_lo), float(peak_hi)],
            "peak_layer_index": am, "peak_relative_depth": float(rel[am]),
            "argmax_at_boundary": bool(boundary),
            "argmax_rel_depth_ci": [float(am_lo), float(am_hi)],
            "argmax_ci_span_fraction": float(am_hi - am_lo),
            "frac_bootstrap_argmax_at_boundary": frac_bd,
            "distinct_bootstrap_argmax_layers": int(len(np.unique(ams))),
            "curve_min": float(np.nanmin(curve)), "curve_range": float(np.nanmax(curve) - np.nanmin(curve)),
            "tilt_per_relative_depth": tilt, "tilt_ci": [float(t_lo), float(t_hi)],
            "selection_bias_in_peak": bias,
            "honest_peak_crossvalidated": float(np.mean(honest)),
            "auroc_curve": [float(v) for v in curve],
        }
        report["models"].append(entry)
        print(f"{model:<11} peak={entry['peak_auroc']:.3f} "
              f"CI[{peak_lo:.3f},{peak_hi:.3f}] relD={entry['peak_relative_depth']:.3f}"
              f"{' EDGE' if boundary else '    '} argmaxCI[{am_lo:.2f},{am_hi:.2f}] "
              f"span={am_hi - am_lo:.2f} bias={bias:+.3f} honest={np.mean(honest):.3f}",
              flush=True)

    # ---- paired model-vs-model comparison of the (naive) peak, as published ----
    print("\n=== PAIRED model-to-model peak-AUROC differences (same items) ===")
    pairs = []
    names = [e["model"] for e in report["models"]]
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b_ = names[i], names[j]
            if a.split("_")[0] != b_.split("_")[0]:
                continue                       # within-family comparisons only
            d = boot_curves[b_].max(axis=1) - boot_curves[a].max(axis=1)
            lo, hi = np.percentile(d, [2.5, 97.5])
            obs = (np.nanmax(auroc_curve(*loaded[b_][:2]))
                   - np.nanmax(auroc_curve(*loaded[a][:2])))
            excl = lo > 0 or hi < 0
            pairs.append({"a": a, "b": b_, "difference": float(obs),
                          "ci95": [float(lo), float(hi)], "excludes_zero": bool(excl)})
            print(f"  {b_:<11} - {a:<11} = {obs:+.4f}  CI[{lo:+.4f},{hi:+.4f}]  "
                  f"{'RESOLVED' if excl else 'not resolved'}")
    report["paired_peak_differences"] = pairs

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
