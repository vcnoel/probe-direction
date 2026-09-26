"""Generalizability study: is the field's statistic a property of models or of a template pair?

Every published evaluation-awareness probe result fixes ONE contrastive template pair. This
runs the field's exact pipeline -- mean-difference direction from a contrastive pair, projected
onto the held-out SAD stages_oversight split, AUROC per layer, maximum over layers -- but
instantiates it once per template pair (6 eval framings x 7 non-eval framings = 42), crossed
with models. That yields a fully crossed model x template design and permits the comparison
the literature has never made:

    spread ACROSS MODELS holding the template fixed   (what the field reports)
    spread ACROSS TEMPLATES holding the model fixed   (never measured)

Framework: generalizability theory (Cronbach, Gleser, Nanda & Rajaratnam 1972), of which
Clark's (1973) language-as-fixed-effect fallacy is the special case. A two-way random-effects
decomposition gives variance components for model, template and their interaction; the
generalizability coefficient states how dependably a single-template design estimates a model
property; and the D-study inverts it to prescribe how many templates are required for a target
reliability.

Reference spreads from the literature, in max|AUROC-0.5| and peak AUROC respectively:
the published scaling range is about 0.13 across 0.27B-70B, and one published Qwen ladder
spans 0.690-0.792 in peak AUROC.

    python scripts/gstudy_analyze.py
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
GS = ROOT / "results" / "gstudy"
ORDER = ["qwen2.5-0.5b", "qwen2.5-1.5b", "qwen2.5-3b", "qwen2.5-7b",
         "qwen2.5-14b", "qwen2.5-32b"]


def unit(v):
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.clip(n, 1e-12, None)


def load(model):
    path = GS / model / "pooled.npz"
    if not path.exists():
        return None
    z = np.load(path, allow_pickle=False)
    fam = json.loads(str(z["families"]))
    return {
        "sad": z["sad"].astype(np.float64), "y": z["sad_label"].astype(int),
        "fr": z["framing"].astype(np.float64), "fname": [str(s) for s in z["framing_name"]],
        "ftask": [str(s) for s in z["framing_task"]], "fam": fam,
    }


def statistics_for_pair(d, f_eval, f_other):
    """Field pipeline for one contrastive template pair -> (max|AUROC-.5|, peakAUROC, peak rel depth)."""
    names = np.array(d["fname"])
    a = d["fr"][names == f_eval].mean(axis=0)      # [L+1, dim]
    b = d["fr"][names == f_other].mean(axis=0)
    direction = unit(a - b)                        # per layer
    n_layers = direction.shape[0]
    aurocs = np.empty(n_layers)
    for L in range(n_layers):
        scores = d["sad"][:, L, :] @ direction[L]
        raw = roc_auc_score(d["y"], scores)
        aurocs[L] = max(raw, 1.0 - raw)
    dist = np.abs(aurocs - 0.5)
    peak = int(np.argmax(dist))
    return float(dist.max()), float(aurocs[peak]), float((peak + 1) / n_layers)


def variance_components(matrix):
    """Two-way random effects, one observation per (model, template) cell.

    matrix: [n_models, n_templates]. With a single observation per cell the interaction is
    confounded with residual error; the reported interaction term is therefore
    interaction-plus-error, which is the quantity that governs the D-study anyway.
    """
    n_m, n_t = matrix.shape
    grand = matrix.mean()
    row, col = matrix.mean(axis=1), matrix.mean(axis=0)
    ms_m = n_t * ((row - grand) ** 2).sum() / (n_m - 1)
    ms_t = n_m * ((col - grand) ** 2).sum() / (n_t - 1)
    resid = matrix - row[:, None] - col[None, :] + grand
    ms_e = (resid ** 2).sum() / ((n_m - 1) * (n_t - 1))
    var_m = max((ms_m - ms_e) / n_t, 0.0)
    var_t = max((ms_t - ms_e) / n_m, 0.0)
    return var_m, var_t, max(ms_e, 0.0)


def main() -> None:
    import sys
    try:  # Windows consoles default to cp1252 and cannot encode Greek letters
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(ROOT / "results" / "gstudy_result.json"))
    args = parser.parse_args()

    loaded = {m: load(m) for m in ORDER}
    loaded = {m: d for m, d in loaded.items() if d is not None}
    if len(loaded) < 2:
        print("need at least two extracted models"); return
    any_d = next(iter(loaded.values()))
    evals = [k for k, v in any_d["fam"].items() if v == "eval"]
    others = [k for k, v in any_d["fam"].items() if v in ("neutral", "deploy")]
    pairs = list(itertools.product(evals, others))
    print(f"{len(loaded)} models x {len(pairs)} contrastive template pairs "
          f"({len(evals)} eval x {len(others)} non-eval)\n")

    dist = np.zeros((len(loaded), len(pairs)))
    peak = np.zeros_like(dist)
    depth = np.zeros_like(dist)
    models = list(loaded)
    for i, m in enumerate(models):
        for j, (fe, fo) in enumerate(pairs):
            dist[i, j], peak[i, j], depth[i, j] = statistics_for_pair(loaded[m], fe, fo)
        print(f"  {m:<14} max|AUROC-.5| over templates: "
              f"min={dist[i].min():.3f} median={np.median(dist[i]):.3f} max={dist[i].max():.3f}"
              f"  (range {dist[i].max() - dist[i].min():.3f})", flush=True)

    print(f"\n{'=' * 92}\nTEMPLATE SPREAD vs MODEL SPREAD  (statistic: max|AUROC-0.5|)\n{'=' * 92}")
    within = float(np.mean(dist.max(axis=1) - dist.min(axis=1)))
    across = float(np.mean(dist.max(axis=0) - dist.min(axis=0)))
    print(f"  mean range across TEMPLATES, model fixed : {within:.4f}")
    print(f"  mean range across MODELS, template fixed : {across:.4f}")
    print(f"  ratio template/model                     : {within / max(across, 1e-9):.2f}x")
    print(f"  published cross-model scaling range      : ~0.13 (0.27B-70B, 15 models)")
    print(f"  published Qwen peak-AUROC range          : ~0.102 (0.690-0.792)")
    print(f"  our peak-AUROC range across templates    : "
          f"{float(np.mean(peak.max(axis=1) - peak.min(axis=1))):.4f}")
    print(f"  our peak relative-depth range across templates: "
          f"{float(np.mean(depth.max(axis=1) - depth.min(axis=1))):.4f}")

    print(f"\n{'=' * 92}\nGENERALIZABILITY (Cronbach et al. 1972), model as object, template as facet\n{'=' * 92}")
    out = {"models": models, "n_pairs": len(pairs), "template_range": within,
           "model_range": across, "statistics": {}}
    for label, mat in (("max|AUROC-0.5|", dist), ("peak AUROC", peak), ("peak rel depth", depth)):
        var_m, var_t, var_e = variance_components(mat)
        total = var_m + var_t + var_e
        g1 = var_m / (var_m + var_t + var_e) if total > 0 else float("nan")
        print(f"\n  {label}")
        print(f"    var(model)={var_m:.5f}  var(template)={var_t:.5f}  var(interaction+err)={var_e:.5f}")
        print(f"    share: model {100 * var_m / total:4.1f}% | template {100 * var_t / total:4.1f}%"
              f" | interaction {100 * var_e / total:4.1f}%")
        print(f"    generalizability with ONE template (what the field uses): Eρ² = {g1:.3f}")
        needed = None
        for k in range(1, 501):
            g = var_m / (var_m + (var_t + var_e) / k) if var_m > 0 else 0.0
            if g >= 0.8:
                needed = k
                break
        print(f"    D-study: templates needed for Eρ² >= 0.80 : "
              f"{needed if needed else '>500'}")
        out["statistics"][label] = {
            "var_model": var_m, "var_template": var_t, "var_interaction": var_e,
            "g_one_template": g1, "templates_for_g80": needed,
        }

    Path(args.out).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
