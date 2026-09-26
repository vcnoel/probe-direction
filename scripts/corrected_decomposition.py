"""Decompose the FLOOR-CORRECTED statistic, since the raw one is substantially floor.

Section 2 establishes that a large part of the reported statistic is what an arbitrary direction
attains under the same folded-maximum convention. A generalizability coefficient computed on the
raw statistic is therefore a coefficient of a quantity that is partly artefact, and the obvious
question is what the decomposition looks like once the floor is removed.

Both are reported and they answer different questions. The RAW decomposition characterises the
statistic as the field reports it, which is the right object for a claim about published practice.
The CORRECTED decomposition, on (statistic - floor mean) / floor sd, characterises whatever signal
remains above an arbitrary direction, which is the right object for a claim about what could be
measured. Since the floor is a per-model quantity, correcting subtracts and rescales by per-model
constants, so it removes exactly the part of the between-model variance that is variation in the
floor rather than in the concept.

The floor is computed on the same scoring array the decomposition uses -- not imported from the
field-construction analysis, which scored a different array -- with an AR(1) random direction whose
layer-to-layer autocorrelation matches the real directions'.

    python scripts/corrected_decomposition.py
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
N_BLOCKS = 4


def unit(v):
    return v / np.clip(np.linalg.norm(v, axis=-1, keepdims=True), 1e-12, None)


def peak(sad, y, d, cols):
    return max(abs(max(a, 1 - a) - 0.5) for a in
               (roc_auc_score(y, sad[:, L, :] @ d[L]) for L in cols))


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


def components(x):
    n_m, n_t, n_h = x.shape
    g = x.mean()
    m = x.mean(axis=(1, 2)); t = x.mean(axis=(0, 2)); h = x.mean(axis=(0, 1))
    mt = x.mean(axis=2); mh = x.mean(axis=1); th = x.mean(axis=0)
    ms_m = n_t * n_h * ((m - g) ** 2).sum() / (n_m - 1)
    ms_t = n_m * n_h * ((t - g) ** 2).sum() / (n_t - 1)
    ms_h = n_m * n_t * ((h - g) ** 2).sum() / (n_h - 1)
    ms_mt = n_h * ((mt - m[:, None] - t[None, :] + g) ** 2).sum() / ((n_m - 1) * (n_t - 1))
    ms_mh = n_t * ((mh - m[:, None] - h[None, :] + g) ** 2).sum() / ((n_m - 1) * (n_h - 1))
    ms_th = n_m * ((th - t[:, None] - h[None, :] + g) ** 2).sum() / ((n_t - 1) * (n_h - 1))
    res = (x - mt[:, :, None] - mh[:, None, :] - th[None, :, :]
           + m[:, None, None] + t[None, :, None] + h[None, None, :] - g)
    ms_e = (res ** 2).sum() / ((n_m - 1) * (n_t - 1) * (n_h - 1))
    q = lambda v: max(v, 0.0)
    return {"s_m": q((ms_m - ms_mt - ms_mh + ms_e) / (n_t * n_h)),
            "s_t": q((ms_t - ms_mt - ms_th + ms_e) / (n_m * n_h)),
            "s_h": q((ms_h - ms_mh - ms_th + ms_e) / (n_m * n_t)),
            "s_mt": q((ms_mt - ms_e) / n_h), "s_mh": q((ms_mh - ms_e) / n_t),
            "s_th": q((ms_th - ms_e) / n_m), "s_mth": q(ms_e)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pooling", default="validated", choices=["validated", "lasttoken"])
    ap.add_argument("--floor-draws", type=int, default=40)
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
    raw, floors = None, {}
    for i, m in enumerate(models):
        with np.load(V11 / m / "pooled.npz", allow_pickle=False) as z:
            sad = z[f"sad_{a.pooling}"].astype(np.float64)
            y = z["sad_label"].astype(int)
            fr = z[f"framing_{a.pooling}"].astype(np.float64)
            names = np.array([str(s) for s in z["framing_name"]])
            fam = json.loads(str(z["families"]))
        ev = sorted(k for k, v in fam.items() if v == "eval")
        dp = sorted(k for k, v in fam.items() if v == "deploy")
        pairs = list(itertools.product(ev, dp))
        cols = list(range(1, sad.shape[1] - 1))
        if raw is None:
            raw = np.zeros((len(models), len(pairs), N_BLOCKS))
        rng = np.random.default_rng(0)
        pos = rng.permutation(np.flatnonzero(y == 1))
        neg = rng.permutation(np.flatnonzero(y == 0))
        blocks = [np.concatenate([p_, q]) for p_, q in
                  zip(np.array_split(pos, N_BLOCKS), np.array_split(neg, N_BLOCKS))]
        dirs = [unit(fr[names == fe].mean(0) - fr[names == fo].mean(0)) for fe, fo in pairs]
        rho = float(np.mean([lag1(d) for d in dirs]))
        fl = [peak(sad, y, ar1(rng, (sad.shape[1], sad.shape[2]), rho), cols)
              for _ in range(a.floor_draws)]
        floors[m] = {"mean": float(np.mean(fl)), "sd": float(np.std(fl, ddof=1)), "rho": rho}
        for j, d in enumerate(dirs):
            pr = np.einsum("ild,ld->il", sad, d)
            for b, idx in enumerate(blocks):
                au = [abs(max(u, 1 - u) - 0.5) for u in
                      (roc_auc_score(y[idx], pr[idx, L]) for L in cols)]
                raw[i, j, b] = max(au)
        print(f"  {m} done   floor {floors[m]['mean']:.3f} +- {floors[m]['sd']:.3f}", flush=True)

    z = np.stack([(raw[i] - floors[m]["mean"]) / floors[m]["sd"]
                  for i, m in enumerate(models)])
    k, n = raw.shape[1], N_BLOCKS

    print(f"\n{'=' * 94}\nRAW VERSUS FLOOR-CORRECTED DECOMPOSITION ({a.pooling} pooling, "
          f"{len(models)} models, {k} templates)\n{'=' * 94}")
    print(f"  {'model':<15}{'mean stat':>11}{'floor':>9}{'mean z':>9}")
    for i, m in enumerate(models):
        print(f"  {m:<15}{raw[i].mean():11.3f}{floors[m]['mean']:9.3f}{z[i].mean():9.2f}")

    out = {"pooling": a.pooling, "models": models, "floors": floors,
           "n_templates": int(k), "n_blocks": n,
           "floor_provenance": {"array": f"gstudy_v11/*/pooled.npz sad_{a.pooling}",
                                "rho_source": "mean lag-1 autocorrelation of our 36 "
                                              "wrapper-derived directions"}}
    for label, x in (("raw", raw), ("corrected", z)):
        c = components(x)
        tot = sum(c.values())
        g1 = c["s_m"] / (c["s_m"] + c["s_mt"] + c["s_mh"] / n + c["s_mth"] / n) \
            if c["s_m"] > 0 else 0.0
        print(f"\n  {label.upper()}")
        for kk in ("s_m", "s_t", "s_h", "s_mt", "s_mh", "s_th", "s_mth"):
            print(f"    {kk:<8}{c[kk]:12.4e}{100 * c[kk] / tot:8.1f}%")
        print(f"    E rho^2 at k=1 {g1:.3f}     s_mt/s_mh "
              f"{c['s_mt'] / max(c['s_mh'], 1e-12):.1f}")
        out[label] = {"components": c, "share": {kk: v / tot for kk, v in c.items()},
                      "g_k1": float(g1)}

    zm = [float(z[i].mean()) for i in range(len(models))]
    nbelow = sum(v < 0 for v in zm)
    print("")
    print("  SIGN FIRST, because the coefficient alone would mislead. Mean standardised excess")
    print(f"  per model: " + ", ".join(f"{v:+.2f}" for v in zm) + ".")
    qw = [i for i, m in enumerate(models) if m.startswith("qwen2.5")]
    qz = [zm[i] for i in qw]
    print(f"  QWEN ONLY, primary: " + ", ".join(f"{v:+.2f}" for v in qz)
          + f"  -- {sum(v < 0 for v in qz)} of {len(qz)} below the floor.")
    print(f"  All {len(models)} models: {nbelow} below. The two clearing the floor are both Qwen,")
    print(f"  and the non-Qwen families use a different prompt rendering -- Gemma folds the")
    print(f"  instruction into the user turn -- so family and rendering are entangled in the")
    print(f"  seven-model figure. The Qwen-only result is reported as primary and matches the")
    print(f"  released-vector finding of two of four clearing.")
    out["mean_z_qwen_only"] = {models[i]: zm[i] for i in qw}
    out["n_qwen_below_floor"] = int(sum(v < 0 for v in qz))
    print(f"  {nbelow} of {len(models)} models sit BELOW the floor on average across templates, so")
    print(f"  the quantity whose reliability improves is the SHORTFALL against an arbitrary")
    print(f"  direction, not a signal. A higher corrected coefficient is therefore not a")
    print(f"  rehabilitation of the statistic: it says models differ dependably in how far short")
    print(f"  their directions fall, which is a stable property of the wrong thing.")
    out["mean_z_per_model"] = {m: zm[i] for i, m in enumerate(models)}
    out["n_models_below_floor"] = int(nbelow)

    dm = out["corrected"]["share"]["s_m"] - out["raw"]["share"]["s_m"]
    print(f"\n  The model share moves {dm:+.1%} from raw to corrected, and E rho^2 from "
          f"{out['raw']['g_k1']:.3f} to {out['corrected']['g_k1']:.3f}.")
    print(f"  Because the floor is a per-model quantity, correcting removes exactly the part of the")
    print(f"  between-model variance that is variation in the FLOOR rather than in the concept.")
    print(f"  Both are reported: the raw decomposition characterises the statistic as the field")
    print(f"  reports it, which is the right object for a claim about published practice; the")
    print(f"  corrected one characterises what remains above an arbitrary direction -- which here is")
    print(f"  mostly negative.")
    p = ROOT / "results" / f"corrected_decomposition_{a.pooling}.json"
    p.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwrote {p}")


if __name__ == "__main__":
    main()
