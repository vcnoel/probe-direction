"""What sets a wrapper's scaling sign on the standard scoring set (docs/sign-mechanism-registration.md).

M1a  per model, Spearman across wrappers of the paper's statistic against |cos(direction, class axis)|
     at the direction's read layer;
M1b  per ladder, Spearman across wrappers of the scaling r of the statistic against the scaling r of
     that alignment;
M1c  per model, cos(class axis, content axis) by layer.
Real directions and the 30 same-class contrasts, full-task, the paper's estimator. The statistic
recomputed here must reproduce the October per-wrapper cells.

    EAS_ACTIVATION_ROOT=/path/to/root python scripts/sign_mechanism.py
"""

from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from controlled_transfer import CELLS_IN, REPRO_TOL, load_model, unit  # noqa: E402
from gstudy_gtheory import auc_cols  # noqa: E402
from pod_ladders import LADDERS  # noqa: E402
from result_io import write_result  # noqa: E402

OUT = ROOT / "results" / "post_submission" / "sign_mechanism.json"


def model_quantities(ladder, model):
    names, tasks, fr, sad, y, ev, dp = load_model(ladder, model)
    frs = ev + dp
    real = [(frs.index(e), frs.index(d)) for e, d in itertools.product(ev, dp)]
    null = [(frs.index(a), frs.index(b)) for g in (ev, dp) for a, b in itertools.combinations(g, 2)]
    rows = {f: np.flatnonzero(names == f) for f in frs}
    L, width = fr.shape[1], fr.shape[2]
    fold = np.empty((L, len(real) + len(null)))
    cos_u = np.empty((L, len(real) + len(null)))
    cos_uc = np.empty(L)
    for l in range(L):
        mu = np.stack([fr[rows[f], l, :].astype(np.float64).mean(0) for f in frs])
        D = np.vstack([unit(np.stack([mu[a] - mu[b] for a, b in real])),
                       unit(np.stack([mu[a] - mu[b] for a, b in null]))])
        S = sad[:, l, :].astype(np.float64)
        u = unit(S[y == 1].mean(0) - S[y == 0].mean(0))
        a = auc_cols(y, S @ D.T)
        fold[l] = np.abs(np.maximum(a, 1.0 - a) - 0.5)
        cos_u[l] = D @ u
        cos_uc[l] = float(unit(D[:len(real)].mean(0)) @ u)
    read = fold.argmax(0)
    stat = fold.max(0)
    align = np.abs(cos_u[read, np.arange(fold.shape[1])])
    return dict(stat=stat, align=align, read=read, cos_uc=cos_uc, L=L, width=width,
                n_real=len(real), n_null=len(null))


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    out = {}
    for name, (tmpl, params) in LADDERS.items():
        models = list(params)
        lp = np.array([np.log10(params[m]) for m in models])
        with np.load(CELLS_IN / f"matched_{name}.npz", allow_pickle=False) as z:
            ref = z["real_full"]
        Q = []
        for i, m in enumerate(models):
            q = model_quantities(name, m)
            diff = float(np.max(np.abs(q["stat"][:q["n_real"]] - ref[i])))
            if diff > REPRO_TOL:
                raise SystemExit(f"{name}/{m}: statistic differs from October cells by {diff:.4g}")
            q["repro"] = diff
            Q.append(q)
            print(f"  {name}/{m}: reproduced to {diff:.1e}", flush=True)
        nr = Q[0]["n_real"]
        sl = {"real": slice(0, nr), "content_free": slice(nr, None)}
        m1a = {k: [float(spearmanr(q["stat"][s], q["align"][s])[0]) for q in Q] for k, s in sl.items()}
        m1b = {}
        for k, s in sl.items():
            S = np.array([q["stat"][s] for q in Q])
            A = np.array([q["align"][s] for q in Q])
            rs = np.array([np.corrcoef(lp, S[:, j])[0, 1] for j in range(S.shape[1])])
            ra = np.array([np.corrcoef(lp, A[:, j])[0, 1] for j in range(A.shape[1])])
            ok = ~np.isnan(rs) & ~np.isnan(ra)
            m1b[k] = {"spearman_r_score_vs_r_align": float(spearmanr(rs[ok], ra[ok])[0]),
                      "same_sign_share": float(np.mean(np.sign(rs[ok]) == np.sign(ra[ok]))),
                      "n": int(ok.sum())}
        m1c = []
        for m, q in zip(models, Q):
            med_read = int(np.median(q["read"][:nr]))
            m1c.append({"model": m, "cos_at_median_read_layer": float(q["cos_uc"][med_read]),
                        "max_abs_cos_over_layers": float(np.abs(q["cos_uc"]).max()),
                        "median_read_layer": med_read, "n_layers": q["L"],
                        "unrelated_expectation": float(1 / np.sqrt(q["width"]))})
        out[name] = {"models": models,
                     "m1a": {"per_model": m1a, "median_real": float(np.median(m1a["real"])),
                             "holds": bool(np.median(m1a["real"]) >= 0.5)},
                     "m1b": {**m1b, "holds": bool(m1b["real"]["spearman_r_score_vs_r_align"] >= 0.5)},
                     "m1c": m1c, "repro_maxdiff": float(max(q["repro"] for q in Q))}
        o = out[name]
        print(f"{name:16s} M1a median {o['m1a']['median_real']:+.2f} (content-free "
              f"{np.median(m1a['content_free']):+.2f})  M1b {m1b['real']['spearman_r_score_vs_r_align']:+.2f} "
              f"same sign {m1b['real']['same_sign_share']:.2f} (content-free "
              f"{m1b['content_free']['spearman_r_score_vs_r_align']:+.2f})  M1c cos "
              + " ".join(f"{r['cos_at_median_read_layer']:+.2f}" for r in m1c), flush=True)
    write_result(OUT, {"ladders": out, "registration": "docs/sign-mechanism-registration.md"},
                 kind="measured",
                 statistic="paper's folded statistic; |cos| of each direction with the scoring set's "
                           "class-mean difference at the direction's read layer; cos of that class axis "
                           "with the mean evaluation-minus-deployment direction",
                 note="post-submission analysis registered before computation; not part of the "
                      "submitted manuscript",
                 pooling="validated for directions and scoring (Qwen2.5: lf scoring, all-token "
                         "directions, as pod_ladders)",
                 layers="ALL; read layer = argmax of the folded statistic per direction",
                 scoring_array="results/pod_*/*.npz and results/gstudy (via EAS_ACTIVATION_ROOT)",
                 data_root="pod", model_population=sorted({m for v in out.values() for m in v["models"]}))
    print(f"wrote {OUT.relative_to(ROOT).as_posix()}")


if __name__ == "__main__":
    main()
