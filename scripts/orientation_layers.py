"""Orientation of held-out framings at the embedding output against the interior band.

Post-submission descriptive check (docs/post-submission-checks.md), not registered. Amendment 1 of
docs/controlled-transfer-registration.md scores whether a direction built from one evaluation and one
deployment framing (train tasks) orders two other framings correctly on held-out tasks, over the
interior band (relative depth 0.25 to 0.75). Hidden state 0 is the embedding output: mean token
embeddings, no computation by the network. If orientation is already high there, it can be carried
by shared vocabulary rather than by a representation the network builds. This reports, per model:

  real              Amendment 1's real orientation (36 directions, disjoint test pairs)
  baseline          the 200 crossing partitions: median and 95th percentile of the per-partition
                    means, and the medians split by direction type, evaluation-minus-deployment
                    (a and b from different families) against same-class (both evaluation or both
                    deployment) directions
  chance            0.5

at hidden state 0 and over the band, for every model of the Qwen3, Qwen3.5, OLMo-2 and Qwen2.5
ladders (non-thinking rendering where there is a switch; activations as
scripts/controlled_transfer.py loads them), and for the paraphrase models with training on the
original framings and testing on version v = 0, 1, 2 (P7d of docs/paraphrase-registration.md).
The band values must reproduce the stored Amendment 1 values (controlled_transfer.json) and the
ungated P7d values (paraphrase_analysis_ungated.json).

    EAS_ACTIVATION_ROOT=<root> python scripts/orientation_layers.py
"""

from __future__ import annotations

import itertools
import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import controlled_transfer as ct  # noqa: E402
import paraphrase_analysis as pa  # noqa: E402
from pod_ladders import LADDERS  # noqa: E402
from result_io import write_result  # noqa: E402

OUT = ROOT / "results" / "post_submission" / "orientation_layers.json"
LADDER_KEYS = ("qwen3_nothink", "qwen3.5_nothink", "olmo2", "qwen2.5_paper")


def layer_sets(L):
    d = np.arange(L) / (L - 1)
    return {"state_0": np.array([0]),
            "band": np.flatnonzero((d >= ct.BAND[0]) & (d <= ct.BAND[1]))}


def g_matrix(Ftr, Fte, lay, n_tasks):
    """G[s, l, w, p]: held-out mean of framing w (Fte) on unit(m_a - m_b) from train tasks (Ftr)."""
    rng = np.random.default_rng(ct.SPLIT_SEED)                   # the registered splits
    splits = [rng.permutation(n_tasks) for _ in range(ct.N_SPLITS)]
    half = n_tasks // 2
    n_fr = Ftr.shape[0]
    pairs = list(itertools.combinations(range(n_fr), 2))
    a_i, b_i = np.array([p[0] for p in pairs]), np.array([p[1] for p in pairs])
    G = np.empty((ct.N_SPLITS, len(lay), n_fr, len(pairs)), np.float32)
    for li, l in enumerate(lay):
        tr, te = Ftr[:, :, l, :].astype(np.float64), Fte[:, :, l, :].astype(np.float64)
        for s, perm in enumerate(splits):
            m = tr[:, perm[:half], :].mean(1)
            G[s, li] = te[:, perm[half:], :].mean(1) @ ct.unit(m[a_i] - m[b_i]).T
    return G, pairs


def orientation(G, pairs, n_eval):
    """Amendment 1's statistic, with the baseline also split by direction type."""
    n_fr = G.shape[2]
    pid = {p: i for i, p in enumerate(pairs)}
    pf = ((G[:, :, :, None, :] - G[:, :, None, :, :]) > 0).mean(axis=(0, 1))

    def ori(u, v, u2, v2):
        return pf[u2, v2, pid[(u, v)]] if u < v else pf[v2, u2, pid[(v, u)]]

    real = [(e, d) for e in range(n_eval) for d in range(n_eval, n_fr)]
    Ro = float(np.mean([ori(e, d, e2, d2) for e, d in real for e2, d2 in real
                        if e2 != e and d2 != d]))
    b_all, b_ed, b_same = [], [], []
    for g1, g2 in ct.crossing_partitions(n_eval, n_fr):
        dirs = [(a, b) for a in g1 for b in g2]
        vals = [(a, b, ori(a, b, a2, b2)) for a, b in dirs for a2, b2 in dirs if a2 != a and b2 != b]
        b_all.append(np.mean([v for _, _, v in vals]))
        b_ed.append(np.mean([v for a, b, v in vals if (a < n_eval) != (b < n_eval)]))
        b_same.append(np.mean([v for a, b, v in vals if (a < n_eval) == (b < n_eval)]))
    return {"real": Ro, "baseline_median": float(np.median(b_all)),
            "baseline_p95": float(np.percentile(b_all, 95)),
            "baseline_eval_minus_deploy_directions_median": float(np.median(b_ed)),
            "baseline_same_class_directions_median": float(np.median(b_same)),
            "chance": 0.5, "real_above_baseline_p95": bool(Ro > np.percentile(b_all, 95))}


def stack(fr, names, tasks, keys, task_ids):
    return fr[pa.rows_by(list(names), list(tasks), keys, task_ids)]          # [12, T, L, d]


def ladder_model(ladder, model):
    names, tasks, fr, _sad, _y, ev, dp = ct.load_model(ladder, model)
    frs = ev + dp
    task_ids = sorted(set(tasks[np.isin(names, frs)]))
    F = stack(fr, names, tasks, frs, task_ids)
    L = F.shape[2]
    res = {"n_hidden_states": L}
    for tag, lay in layer_sets(L).items():
        G, pairs = g_matrix(F, F, lay, len(task_ids))
        res[tag] = orientation(G, pairs, len(ev))
        res[tag]["states"] = [int(lay[0]), int(lay[-1])]
    return res


def paraphrase_model(model):
    with np.load(pa.ACTS / f"{model}__nothink.npz", allow_pickle=False) as z:
        names = [str(s) for s in z["framing_name"]]
        tasks = [str(s) for s in z["framing_task"]]
        fr = z["framing_validated"]
    orig = {n for n in names if not re.search(r"_p[12]$", n)}
    ev = sorted(n for n in orig if n.startswith("eval_"))
    dp = sorted(n for n in orig if n.startswith("deploy_"))
    frs, task_ids = ev + dp, sorted(set(tasks))
    F = {v: stack(fr, names, tasks, [pa.vname(f, v) for f in frs], task_ids) for v in pa.VERSIONS}
    L = F[0].shape[2]
    res = {"n_hidden_states": L}
    for tag, lay in layer_sets(L).items():
        res[tag] = {}
        for v in pa.VERSIONS:
            G, pairs = g_matrix(F[0], F[v], lay, len(task_ids))
            res[tag][f"v{v}"] = orientation(G, pairs, len(ev))
    return res


def line(tag, r):
    return (f"{tag:7s} real {r['real']:.3f}  base med {r['baseline_median']:.3f} p95 "
            f"{r['baseline_p95']:.3f}  E-D dirs {r['baseline_eval_minus_deploy_directions_median']:.3f}"
            f"  same-class {r['baseline_same_class_directions_median']:.3f}")


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    ct_json = json.loads((ROOT / "results" / "post_submission" / "controlled_transfer.json")
                         .read_text(encoding="utf-8"))["ladders"]
    out, pop, repro = {"ladders": {}, "paraphrase": {}}, [], []
    for ladder in LADDER_KEYS:
        stored = {r["model"]: r["real_orientation"]
                  for r in ct_json[ladder]["amendment_1"]["a1"]["per_model"]}
        out["ladders"][ladder] = {}
        for model in LADDERS[ladder][1]:
            r = ladder_model(ladder, model)
            r["band_reproduces_stored_a1"] = abs(r["band"]["real"] - stored[model])
            repro.append(r["band_reproduces_stored_a1"])
            out["ladders"][ladder][model] = r
            pop.append(model)
            print(f"{ladder:16s} {model:26s} " + line("state0", r["state_0"]) + " | " +
                  line("band", r["band"]) + f" | vs stored A1 {r['band_reproduces_stored_a1']:.1e}",
                  flush=True)
    ung = json.loads((ROOT / "results" / "post_submission" / "paraphrase" /
                      "paraphrase_analysis_ungated.json").read_text(encoding="utf-8"))["ladders"]
    for ladder in ("qwen3_nothink", "qwen3.5_nothink"):
        for rec in ung[ladder]["per_model"]:
            model = rec["model"]
            r = paraphrase_model(model)
            r["band_reproduces_ungated_p7d"] = max(abs(r["band"][f"v{v}"]["real"] -
                                                       rec["p7d"][f"v{v}"]["real_orientation"])
                                                   for v in pa.VERSIONS)
            repro.append(r["band_reproduces_ungated_p7d"])
            out["paraphrase"][model] = r
            for v in pa.VERSIONS:
                print(f"paraphrase {model:14s} v{v}  " + line("state0", r["state_0"][f"v{v}"]) +
                      " | " + line("band", r["band"][f"v{v}"]), flush=True)
    out["max_abs_diff_band_vs_stored"] = float(max(repro))
    out["check"] = "post-submission descriptive check, not registered (docs/post-submission-checks.md)"
    write_result(OUT, out, kind="measured",
                 statistic="Amendment 1 orientation (share of correctly ordered held-out framing "
                           "pairs), real directions against the 200 crossing partitions, at hidden "
                           "state 0 and over relative depth 0.25-0.75",
                 note="hidden state 0 is the embedding output; paraphrase rows train on the "
                      "original framings and test on version v",
                 pooling="validated framing rows (Qwen2.5: all-token, as controlled_transfer)",
                 layers="hidden state 0; relative depth 0.25-0.75",
                 scoring_array="held-out framing rows of results/pod_*/*.npz, results/gstudy and "
                               "results/post_submission/paraphrase/acts",
                 data_root="pod + gstudy + post_submission/paraphrase",
                 model_population=sorted(set(pop) | set(out["paraphrase"])))
    print(f"wrote {OUT.relative_to(ROOT).as_posix()}  (band reproduction max |d| "
          f"{out['max_abs_diff_band_vs_stored']:.1e})")


if __name__ == "__main__":
    main()
