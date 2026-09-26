"""An unbalanced contrastive set manufactures a scaling law. Here is how big one you need.

The released contrastive set is exactly counterbalanced, so the answer-token-identity component v
cancels and the estimated direction is the concept component c. Nothing in the construction enforces
that balance, and no paper reports checking it. With imbalance fraction
f = (n_A - n_B)/(n_A + n_B) the estimate becomes c + f v, so

    cos(d_hat, c) = 1 / sqrt(1 + f^2 r^2),      r = |v| / |c|.

The hazard is not that this degrades the direction. It is that r falls monotonically with model
scale -- measured at d=0.9: 8.04, 3.08, 2.36, 1.59 across the ladder -- so a fixed imbalance
degrades small models far more than large ones. A degraded direction scores lower on the held-out
set, so an unbalanced contrastive set produces a SPURIOUS POSITIVE SCALING TREND, which is the shape
of the published power law.

This sweeps f and reports, per model, the resulting statistic and the slope of the statistic on
log10 parameters, so the imbalance an author would need for an entire scaling result to be an
artefact can be stated as a number rather than gestured at.

The sets actually used by both reanalysed papers are the balanced 204, so their published numbers
are unaffected: this is a prospective hazard for anyone building a new contrastive set.

    python scripts/imbalance_hazard.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
FC = ROOT / "results" / "field_construction"
CONTRASTIVE = reanalysis("manek", "datasets", "contrastive_dataset.json")
# Parameter counts come from configs/gstudy_models.json, derived from safetensors byte
# totals. A hardcoded dict of nominal names used to live here and was wrong by 0.037 in
# log10 at 7B, the leverage point of the fit.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from external_paths import reanalysis  # noqa: E402
from result_io import params_b as _params_b  # noqa: E402
PARAMS = _params_b()
FRACS = [0.0, 0.05, 0.10, 0.15, 0.20, 0.30]
PUBLISHED_SLOPE = 0.0531       # our refit of the released scaling data, for scale comparison


def unit(v):
    return v / np.clip(np.linalg.norm(v, axis=-1, keepdims=True), 1e-12, None)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--draws", type=int, default=40)
    ap.add_argument("--statistic", default="peak", choices=["peak", "tilt"])
    a = ap.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    items = json.loads(CONTRASTIVE.read_text(encoding="utf-8"))
    letters = np.array([e["positive"][1] for e in items])
    iA, iB = np.flatnonzero(letters == "A"), np.flatnonzero(letters == "B")
    models = [m for m in PARAMS if (FC / f"{m}.npz").exists()]
    if len(models) < 3:
        print("need the field-construction extraction first"); return

    stat = {f: {} for f in FRACS}
    sds = {f: {} for f in FRACS}
    ratios, v_alone = {}, {}
    for m in models:
        with np.load(FC / f"{m}.npz", allow_pickle=False) as z:
            A = z["acts"].astype(np.float64)
        diff = A[:, 0] - A[:, 1]
        nl = diff.shape[1]
        dA, dB = diff[iA].mean(0), diff[iB].mean(0)
        v, c = (dA - dB) / 2.0, (dA + dB) / 2.0
        L9 = max(1, int(0.9 * (nl - 1)))
        ratios[m] = float(np.linalg.norm(v[L9]) / max(np.linalg.norm(c[L9]), 1e-12))

        with np.load(ROOT / "results" / "gstudy" / m / "sad_filtered_lf.npz",
                     allow_pickle=False) as z:
            sad, y = z["sad"].astype(np.float64), z["sad_label"].astype(int)
        nlay = min(nl, sad.shape[1])
        rng = np.random.default_rng(0)
        def score(d):
            curve = np.array([abs(max(au, 1 - au) - 0.5) for au in
                              (roc_auc_score(y, sad[:, L, :] @ d[L])
                               for L in range(1, nlay - 1))])
            if a.statistic == "peak":
                return float(curve.max())
            z = np.linspace(0.0, 1.0, curve.size)
            z = z - z.mean()
            return float((z * (curve - curve.mean())).sum() / max((z ** 2).sum(), 1e-12))

        # Does the artefact component discriminate the scoring set on its own? If it does, mixing
        # it in can RAISE a folded maximum, because that statistic rewards any direction that
        # separates the classes regardless of orientation or of whether it is the concept. That
        # would explain cells where imbalance improves the score rather than degrading it.
        v_alone[m] = float(score(unit(v)))

        # The f=0 reference is the DETERMINISTIC full-set direction, not a subsample. An earlier
        # version drew one 102-item subsample for f=0 while averaging 30 draws for every f>0,
        # which put a single noisy draw in the row the whole comparison rested on.
        stat[0.0][m] = float(score(unit(diff.mean(0))))
        for f in [x for x in FRACS if x > 0]:
            # a pool of 102 with the requested A/B excess, total size held fixed
            nA = int(round(102 * (1 + f) / 2))
            nB = 102 - nA
            vals = []
            for _ in range(a.draws):
                S = np.concatenate([rng.permutation(iA)[:nA], rng.permutation(iB)[:nB]])
                vals.append(score(unit(diff[S].mean(0))))
            stat[f][m] = float(np.mean(vals))
            sds[f][m] = float(np.std(vals, ddof=1))
        # a size-matched BALANCED subsample, so the f>0 rows are compared against equal n
        vals = []
        for _ in range(a.draws):
            S = np.concatenate([rng.permutation(iA)[:51], rng.permutation(iB)[:51]])
            vals.append(score(unit(diff[S].mean(0))))
        stat["balanced102"] = stat.get("balanced102", {})
        stat["balanced102"][m] = float(np.mean(vals))
        sds["balanced102"] = sds.get("balanced102", {})
        sds["balanced102"][m] = float(np.std(vals, ddof=1))
        print(f"  {m} done", flush=True)
        del A, diff

    print(f"\n{'=' * 92}\nSTATISTIC UNDER CONTRASTIVE-SET IMBALANCE\n{'=' * 92}")
    print(f"  r = |v|/|c| at d=0.9: " + ", ".join(f"{m.split('-')[-1]} {ratios[m]:.2f}"
                                                  for m in models))
    print(f"\n  {'imbalance f':<14}" + "".join(f"{m.split('-')[-1]:>10}" for m in models)
          + f"{'slope':>10}{'r':>8}")
    x = np.log10([PARAMS[m] for m in models])
    rows = {}
    for f in list(FRACS) + ["balanced102"]:
        ys = [stat[f][m] for m in models]
        slope = float(np.polyfit(x, ys, 1)[0])
        rr = float(pearsonr(x, ys)[0])
        rows[f] = {"values": ys, "slope": slope, "r": rr}
        lbl = "f=0, n=204" if f == 0.0 else (
              "balanced n=102" if f == "balanced102" else f"f={f:.2f}, n=102")
        cells = []
        for mm, vv in zip(models, ys):
            sd = sds.get(f, {}).get(mm)
            cells.append(f"{vv:.3f}+-{sd:.3f}" if sd is not None else f"{vv:.3f}      ")
        print(f"  {lbl:<15}" + "".join(f"{c:>14}" for c in cells)
              + f"{slope:+9.4f}{rr:+7.2f}")

    print("")
    print("  predicted cos(d_hat, c) = 1/sqrt(1 + f^2 r^2) at f=0.10, and the artefact",
          "direction")
    print("  scored ALONE on the evaluation set:")
    print(f"    {'model':<15}{'pred cos':>10}{'artefact alone':>16}")
    for m in models:
        pc = 1.0 / np.sqrt(1 + 0.01 * ratios[m] ** 2)
        print(f"    {m:<15}{pc:10.3f}{v_alone[m]:16.4f}")
    print("")
    print("  A folded maximum rewards any direction separating the classes, whatever its")
    print("  orientation and whether or not it is the concept. Where the artefact separates")
    print("  the scoring set, mixing it in RAISES the statistic, so imbalance SUBSTITUTES")
    print("  rather than merely degrading, and the sign of its effect is not predictable.")
    print("")
    print("  No comparison is made here to a published slope. Our +0.0531 refit is of the")
    print("  fifteen-model file including Qwen; the headline fit EXCLUDES Qwen and gives")
    print("  +0.0744. Any reproduction claim would be against a fit its authors did not")
    print("  report, on the family they deliberately dropped. The statement is only that")
    print("  under their construction and their set, four Qwen give a positive slope.")
    print(f"\n  Both reanalysed papers used the balanced 204-item set ({iA.size} and {iB.size}), so")
    print(f"  their numbers are unaffected. This is a prospective hazard for new sets, and it is")
    print(f"  cheap to preclude: report the answer-key balance, or counterbalance by construction.")

    out = {"fracs": FRACS, "models": models, "r_at_d09": ratios,
           "statistic": {str(f): rows[f]["values"] for f in rows},
           "slope": {str(f): rows[f]["slope"] for f in rows},
           "correlation": {str(f): rows[f]["r"] for f in rows},
           "sd": {str(f): sds.get(f, {}) for f in rows},
           "artefact_alone": v_alone,
           "n_positive_A": int(iA.size), "n_positive_B": int(iB.size)}
    p = ROOT / "results" / "imbalance_hazard.json"
    p.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwrote {p}")


if __name__ == "__main__":
    main()
