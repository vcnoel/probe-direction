"""Bootstrap intervals for the variance components of results/pod_gtheory.json.

Each replicate resamples the three facets that are treated as random: the scoring items (with
replacement, within each label, then split into the same four label-stratified blocks), the six
evaluation framings and the six deployment framings (each with replacement, so a replicate has
36 wrappers built from the resampled framings). The models are the ladder and are held fixed.
The three-way crossed decomposition is then refitted with the same code as scripts/pod_ladders.py
(components_3way, coefficients), and three quantities are recorded per replicate:

  model share        s_m / total
  model x wrapper    s_mt / total
  E rho^2            relative coefficient for one wrapper on the full item set (n = 4 blocks)

Intervals are percentile 2.5 and 97.5 over N_REPS replicates, seed 0. The point value stored
beside each interval is the full-data estimate from pod_gtheory.json, recomputed here.

    python scripts/bootstrap_components.py            # every ladder in one process
    python scripts/bootstrap_components.py olmo2      # one ladder, to results/pod_gtheory_bootstrap_parts/
    python scripts/bootstrap_components.py --merge    # merge the parts into the result file
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import pod_ladders as pl  # noqa: E402
from gstudy_gtheory import coefficients, components_3way, total  # noqa: E402
from result_io import write_result  # noqa: E402

N_REPS = 200
N_BLOCKS = pl.N_BLOCKS


def cells(P, Y, pairs_idx, pos, neg, rng):
    pos, neg = rng.permutation(pos), rng.permutation(neg)
    blocks = [np.concatenate([p, q]) for p, q in
              zip(np.array_split(pos, N_BLOCKS), np.array_split(neg, N_BLOCKS))]
    return np.array([[[pl.peak(Y[b], P[i][j][b]) for b in blocks] for j in pairs_idx]
                     for i in range(len(P))])


def summarise(c):
    tot = total(c)
    return (c["s_m"] / tot, c["s_mt"] / tot, coefficients(c, k=1, n=N_BLOCKS)[0])


def main() -> None:
    import warnings
    warnings.filterwarnings("ignore")
    out = {}
    only = sys.argv[1:]
    if only == ["--merge"]:
        return merge()
    for name, (tmpl, params) in pl.LADDERS.items():
        if only and name not in only:
            continue
        t0 = time.time()
        models = list(params)
        gstudy = tmpl == "GSTUDY"
        paths = models if gstudy else [tmpl.format(m) for m in models]
        if not gstudy and any(not (ROOT / p).exists() for p in paths):
            print(f"  SKIP {name}")
            continue
        P = []
        for p in paths:
            proj, y, pairs, ev, dp = pl.load_qwen25(p) if gstudy else pl.load(p)
            P.append(proj)
        Y = y
        grid = {(fe, fo): j for j, (fe, fo) in enumerate(pairs)}
        pos0, neg0 = np.flatnonzero(Y == 1), np.flatnonzero(Y == 0)

        # full-data point estimate, same blocks as pod_ladders (seed 0 permutation)
        rng0 = np.random.default_rng(0)
        x0 = cells(P, Y, range(len(pairs)), pos0, neg0, rng0)
        point = summarise(components_3way(x0))

        rng = np.random.default_rng(0)
        reps = []
        for _ in range(N_REPS):
            pos = rng.choice(pos0, size=len(pos0), replace=True)
            neg = rng.choice(neg0, size=len(neg0), replace=True)
            e = rng.choice(len(ev), size=len(ev), replace=True)
            d = rng.choice(len(dp), size=len(dp), replace=True)
            idx = [grid[(ev[a], dp[b])] for a in e for b in d]
            reps.append(summarise(components_3way(cells(P, Y, idx, pos, neg, rng))))
        reps = np.array(reps)
        lo, hi = np.nanpercentile(reps, 2.5, axis=0), np.nanpercentile(reps, 97.5, axis=0)
        keys = ("model_share", "model_x_wrapper_share", "erho_single")
        out[name] = {"models": models, "n_reps": N_REPS,
                     **{k: {"point": float(point[i]), "lo": float(lo[i]), "hi": float(hi[i])}
                        for i, k in enumerate(keys)}}
        print(f"  {name}: " + "  ".join(f"{k} {point[i]:.3f} [{lo[i]:.3f}, {hi[i]:.3f}]"
                                      for i, k in enumerate(keys))
              + f"  ({time.time() - t0:.0f}s)", flush=True)

    if only:
        import json
        (ROOT / "results" / "pod_gtheory_bootstrap_parts").mkdir(exist_ok=True)
        for k, v in out.items():
            (ROOT / "results" / "pod_gtheory_bootstrap_parts" / f"{k}.json").write_text(
                json.dumps(v, indent=1), encoding="utf-8")
        return
    write_all(out)


def merge():
    import json
    d = ROOT / "results" / "pod_gtheory_bootstrap_parts"
    write_all({p.stem: json.loads(p.read_text(encoding="utf-8")) for p in sorted(d.glob("*.json"))})


def write_all(out):
    pop = sorted({m for v in out.values() for m in v["models"]})
    write_result(ROOT / "results" / "pod_gtheory_bootstrap.json", {"ladders": out},
                 kind="measured", pooling="validated for scoring and for the direction arrays",
                 layers="ALL", scoring_array="results/pod_*/*.npz", data_root="pod",
                 statistic="folded max over layers, 3-way crossed model x wrapper x item-block, "
                           "bootstrap over items and over evaluation and deployment framings",
                 model_population=pop)
    print("wrote results/pod_gtheory_bootstrap.json")


if __name__ == "__main__":
    main()
