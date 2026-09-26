"""The control-task analogue of Hewitt & Liang (2019), adapted to an untrained probe.

Their control randomises the LABELS of the probing task, so that a probe scoring well on the
control must be reading its own capacity rather than the representation. Here the probe has no
fitted parameters on the scoring set -- it is a difference of means over a separate contrastive set,
applied zero-shot -- so probe capacity is not the free parameter and their construction does not
transfer unchanged. Two controls are available instead, and they bracket different things:

  RANDOM DIRECTION (elsewhere in this project). Replace the direction with an isotropic AR(1)
  vector matched on layer-to-layer autocorrelation. This asks what the SCORING SET plus the
  folded-maximum convention give to a direction carrying no information at all.

  PERMUTED CONTRASTIVE LABELS (this script). Keep the estimator, the pooling, the contrastive
  prompts and the layer convention exactly as they are, and permute which framing prompts count as
  "evaluation" and which as "deployment" before taking the difference of means. This is the closer
  analogue of a control task: the construction is untouched and only the labels that give it meaning
  are destroyed.

The second is the stricter control of the two in one respect that matters. A permuted-label
direction still lies in the span of the framing activations, a subspace dominated by prompt-format
variation, whereas an isotropic direction does not. PREDICTION FIXED BEFORE RUNNING: the
permuted-label floor is at or above the AR(1) floor for that reason, which would mean the reported
statistic's margin over a content-free direction is smaller than the AR(1) comparison suggests.
If instead it falls well below AR(1), the AR(1) floor is too generous and the margin is larger than
this paper claims elsewhere; that outcome is reported with the same prominence.

    python scripts/label_permuted_control.py
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
V11 = ROOT / "results" / "gstudy_v11"


def unit(v):
    return v / np.clip(np.linalg.norm(v, axis=-1, keepdims=True), 1e-12, None)


def folded_peak(sad, y, d, cols):
    """The field's statistic: max over interior layers of |AUROC - 0.5|, folded."""
    pr = np.einsum("ild,ld->il", sad, d)
    return max(abs(max(a, 1 - a) - 0.5) for a in
               (roc_auc_score(y, pr[:, L]) for L in cols))


def lag1(d):
    u = unit(d)
    return float(np.mean([abs(np.dot(u[L], u[L + 1])) for L in range(1, u.shape[0] - 2)]))


def ar1(rng, shape, rho):
    L, dim = shape
    out = np.empty(shape)
    out[0] = rng.standard_normal(dim)
    s = np.sqrt(max(1.0 - rho ** 2, 1e-9))
    for i in range(1, L):
        out[i] = rho * out[i - 1] + s * rng.standard_normal(dim)
    return unit(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pooling", default="validated", choices=["validated", "lasttoken"])
    ap.add_argument("--draws", type=int, default=40)
    # The canonical estimand is the families of the published ladders. Enabling a fourth family in
    # the shared config silently widened this analysis from 7 models to 10, changing the population
    # the components describe -- the same estimand drift rejected for the REML fit, arriving through
    # a config edit instead of a decision. Widening is now explicit.
    ap.add_argument("--families", default="qwen2.5,gemma2,llama3.2",
                    help="comma-separated config families; default is the published ladders'")
    a = ap.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    conf = json.loads((ROOT / "configs" / "gstudy_models.json").read_text(encoding="utf-8"))
    fams = set(a.families.split(","))
    models = [k for k, v in conf["models"].items()
              if v.get("enabled") and v.get("family") in fams
              and (V11 / k / "pooled.npz").exists()]
    print(f"model set: {len(models)} models from families "
          f"{sorted(fams)}", flush=True)

    rows, out = [], {"pooling": a.pooling, "draws": a.draws, "per_model": {}}
    for mi, m in enumerate(models):
        with np.load(V11 / m / "pooled.npz", allow_pickle=False) as z:
            sad = z[f"sad_{a.pooling}"].astype(np.float64)
            y = z["sad_label"].astype(int)
            fr = z[f"framing_{a.pooling}"].astype(np.float64)
            names = np.array([str(s) for s in z["framing_name"]])
            fam = json.loads(str(z["families"]))
        cols = list(range(1, sad.shape[1] - 1))
        ev = sorted(k for k, v in fam.items() if v == "eval")
        dp = sorted(k for k, v in fam.items() if v == "deploy")
        pairs = list(itertools.product(ev, dp))
        rng = np.random.default_rng(0)

        real, perm = [], []
        for fe, fo in pairs:
            ie, io = np.flatnonzero(names == fe), np.flatnonzero(names == fo)
            real.append(folded_peak(sad, y, unit(fr[ie].mean(0) - fr[io].mean(0)), cols))
            # Permute the eval/deploy label across the SAME rows, preserving group sizes. The
            # estimator, the pooling, the prompts and the layer convention are all untouched.
            pool = np.concatenate([ie, io])
            for _ in range(2):
                q = rng.permutation(pool)
                d = unit(fr[q[:len(ie)]].mean(0) - fr[q[len(ie):]].mean(0))
                perm.append(folded_peak(sad, y, d, cols))

        dirs = [unit(fr[names == fe].mean(0) - fr[names == fo].mean(0)) for fe, fo in pairs]
        rho = float(np.mean([lag1(d) for d in dirs]))
        iso = [folded_peak(sad, y, ar1(rng, (sad.shape[1], sad.shape[2]), rho), cols)
               for _ in range(a.draws)]

        r = {"real_mean": float(np.mean(real)), "real_median": float(np.median(real)),
             "perm_mean": float(np.mean(perm)), "perm_sd": float(np.std(perm, ddof=1)),
             "perm_n": len(perm),
             "iso_mean": float(np.mean(iso)), "iso_sd": float(np.std(iso, ddof=1)),
             "rho": rho, "n_layers": int(sad.shape[1])}
        r["z_vs_perm"] = (r["real_mean"] - r["perm_mean"]) / r["perm_sd"]
        r["z_vs_iso"] = (r["real_mean"] - r["iso_mean"]) / r["iso_sd"]
        r["perm_minus_iso"] = r["perm_mean"] - r["iso_mean"]
        out["per_model"][m] = r
        rows.append((m, r))
        print(f"  {m:<15} real {r['real_mean']:.3f}   perm-label {r['perm_mean']:.3f}"
              f" +- {r['perm_sd']:.3f}   AR(1) {r['iso_mean']:.3f} +- {r['iso_sd']:.3f}"
              f"   z_perm {r['z_vs_perm']:+.2f}", flush=True)

    print(f"\n{'=' * 96}")
    print("TWO CONTROLS: PERMUTED CONTRASTIVE LABELS AGAINST AN ISOTROPIC RANDOM DIRECTION")
    print(f"{'=' * 96}")
    dz = [r["perm_minus_iso"] for _, r in rows]
    zp = [r["z_vs_perm"] for _, r in rows]
    zi = [r["z_vs_iso"] for _, r in rows]
    higher = sum(v > 0 for v in dz)
    print(f"  The permuted-label floor exceeds the AR(1) floor in {higher} of {len(rows)} models,"
          f" by {np.median(dz):+.3f} at the median.")
    print(f"  Standardised excess of the reported statistic over the permuted-label floor:"
          f" median {np.median(zp):+.2f}")
    print(f"  over the AR(1) floor: median {np.median(zi):+.2f}.")
    print(f"  Cells at or below the permuted-label floor: {sum(v <= 0 for v in zp)} of {len(rows)}."
          f"  Below AR(1): {sum(v <= 0 for v in zi)} of {len(rows)}.")
    print("")
    if higher > len(rows) / 2:
        print("  PREDICTION HELD. A permuted-label direction lies in the span of the framing")
        print("  activations, a subspace dominated by prompt format, so destroying the labels")
        print("  costs less than replacing the direction with an isotropic one. The control-task")
        print("  analogue is therefore the STRICTER comparison, and the margin over a content-free")
        print("  direction is smaller than the AR(1) figure alone implies.")
    else:
        print("  PREDICTION FAILED. The permuted-label floor sits BELOW the isotropic one, so the")
        print("  AR(1) floor is the more generous comparison and the reported margin is larger than")
        print("  this paper's AR(1)-based statements suggest. Reported at equal prominence; the")
        print("  AR(1) floor remains the primary comparison because it is the weaker claim.")

    out["summary"] = {
        "n_models": len(rows), "n_perm_above_iso": int(higher),
        "median_perm_minus_iso": float(np.median(dz)),
        "median_z_vs_perm": float(np.median(zp)), "median_z_vs_iso": float(np.median(zi)),
        "n_at_or_below_perm": int(sum(v <= 0 for v in zp)),
        "n_at_or_below_iso": int(sum(v <= 0 for v in zi)),
        "prediction_held": bool(higher > len(rows) / 2)}
    p = ROOT / "results" / f"label_permuted_control_{a.pooling}.json"
    p.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwrote {p}")


if __name__ == "__main__":
    main()
