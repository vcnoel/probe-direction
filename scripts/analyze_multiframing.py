"""Leave-one-framing-out test for a cue-general evaluation axis.

The question the single-framing design could not answer: is there a representational
component shared by *evaluation framings in general*, or only the footprint of one fixed
prefix? Here the axis is built contrastively from training framings only --

    axis = normalize( mean(eval framing means) - mean(neutral framing means) )

-- and then scored on a held-out eval framing and a held-out neutral framing that never
touched axis construction. The statistic is

    delta = mean_t cos(shift[held-out eval, t], axis) - mean_t cos(shift[held-out neutral, t], axis)

aggregated over every (held-out eval, held-out neutral) pair. delta > 0 means the axis
generalizes to unseen evaluation wordings, which a one-prefix footprint cannot do.

Null: the family labels carry the claim, so we permute them. With six framings per family
there are C(12,6)/2 = 462 distinct label splits, which we enumerate EXACTLY rather than
sample. A large delta under permuted labels would mean the pipeline manufactures
separation from any arbitrary grouping of framings.

Controls: task-span position and prompt length are recorded per row, so we also report
their correlation with the axis projection and the family means, and a per-layer delta
curve (no fixed band is assumed).

    python scripts/analyze_multiframing.py
"""

from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "multiframing"
FRAMINGS = ROOT / "data" / "framings_multi.json"
MODELS = [("qwen2.5-0.5b", 0.5), ("qwen2.5-1.5b", 1.5), ("qwen2.5-3b", 3.0), ("qwen2.5-7b", 7.0)]


def unit(a: np.ndarray) -> np.ndarray:
    return a / np.clip(np.linalg.norm(a, axis=-1, keepdims=True), 1e-12, None)


def load(model: str):
    payload = np.load(RESULTS / model / "activations.npz", allow_pickle=False)
    vectors = payload["mean_vec"].astype(np.float64)      # [N, L+1, d]
    task_ids = [str(t) for t in payload["task_id"]]
    conditions = [str(c) for c in payload["condition"]]
    starts = payload["task_start"].astype(float)
    tasks = sorted(set(task_ids))
    framings = sorted(set(conditions))
    index = {(t, c): i for i, (t, c) in enumerate(zip(task_ids, conditions))}
    keep = [t for t in tasks if all((t, f) in index for f in framings)]
    n_layers, dim = vectors.shape[1], vectors.shape[2]
    cube = np.zeros((len(keep), len(framings), n_layers, dim))
    start = np.zeros((len(keep), len(framings)))
    for ti, task in enumerate(keep):
        for fi, framing in enumerate(framings):
            row = index[(task, framing)]
            cube[ti, fi] = vectors[row]
            start[ti, fi] = starts[row]
    return cube, keep, framings, start


def families() -> dict[str, str]:
    spec = json.loads(FRAMINGS.read_text(encoding="utf-8"))
    return {k: v["family"] for k, v in spec.items() if not k.startswith("_")}


def framing_means(shifts: np.ndarray) -> np.ndarray:
    """shifts [T, F, d] -> per-framing mean unit direction [F, d]."""
    return unit(unit(shifts).mean(axis=0))


def residualize_on_position(shifts: np.ndarray, starts: np.ndarray) -> np.ndarray:
    """Remove any component of the shift linearly predictable from task-span position.

    Eval framings start marginally later than neutral ones (89-103 vs 88-96 tokens), so a
    contrastive axis could encode absolute position rather than evaluation. This projects
    out the least-squares position direction, computed over all (task, framing) rows.
    """
    flat = shifts.reshape(-1, shifts.shape[-1])
    pos = starts.reshape(-1).astype(np.float64)
    pos = pos - pos.mean()
    denom = float(pos @ pos)
    if denom < 1e-9:
        return shifts
    direction = (flat * pos[:, None]).sum(axis=0) / denom      # d/dposition
    residual = flat - np.outer(pos, direction)
    return residual.reshape(shifts.shape)


def lofo_delta(shifts: np.ndarray, eval_idx: list[int], neut_idx: list[int]) -> float:
    """Mean held-out delta over every (held-out eval, held-out neutral) pair."""
    means = framing_means(shifts)
    units = unit(shifts)                                   # [T, F, d]
    deltas = []
    for held_e in eval_idx:
        for held_n in neut_idx:
            tr_e = [i for i in eval_idx if i != held_e]
            tr_n = [i for i in neut_idx if i != held_n]
            axis = unit(means[tr_e].mean(axis=0) - means[tr_n].mean(axis=0))
            s_e = float((units[:, held_e] @ axis).mean())
            s_n = float((units[:, held_n] @ axis).mean())
            deltas.append(s_e - s_n)
    return float(np.mean(deltas))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--out", default=str(ROOT / "results" / "multiframing_lofo.json"))
    args = parser.parse_args()

    fam = families()
    report: dict = {"models": []}

    for model, params in MODELS:
        if not (RESULTS / model / "activations.npz").exists():
            print(f"missing {model}; skipping")
            continue
        cube, tasks, framings, starts = load(model)
        anchor = [i for i, f in enumerate(framings) if fam.get(f) == "deploy"]
        eval_idx = [i for i, f in enumerate(framings) if fam.get(f) == "eval"]
        neut_idx = [i for i, f in enumerate(framings) if fam.get(f) == "neutral"]
        n_layers = cube.shape[2]

        # shift vs the deploy anchor, per layer
        anchor_vec = cube[:, anchor].mean(axis=1)                       # [T, L, d]
        shifts_all = cube - anchor_vec[:, None]                         # [T, F, L, d]

        # primary: mean over all layers except the embedding layer
        band = np.arange(1, n_layers)
        shifts = shifts_all[:, :, band].mean(axis=2)                    # [T, F, d]

        observed = lofo_delta(shifts, eval_idx, neut_idx)

        # position-residualized variant: the same statistic after projecting out any
        # component linearly predictable from task-span start
        shifts_resid = residualize_on_position(shifts, starts)
        observed_resid = lofo_delta(shifts_resid, eval_idx, neut_idx)

        # EXACT permutation over family labels: all distinct 6/6 splits of the 12 framings
        pool = eval_idx + neut_idx
        null, null_resid = [], []
        seen = set()
        for combo in combinations(range(len(pool)), len(eval_idx)):
            key = frozenset(combo)
            complement = frozenset(set(range(len(pool))) - set(combo))
            if complement in seen:
                continue
            seen.add(key)
            e = [pool[i] for i in combo]
            n = [pool[i] for i in range(len(pool)) if i not in combo]
            null.append(lofo_delta(shifts, e, n))
            null_resid.append(lofo_delta(shifts_resid, e, n))
        null_arr = np.array(null)
        null_resid_arr = np.array(null_resid)
        p_value = float((np.sum(null_arr >= observed) + 1) / (null_arr.size + 1))
        p_resid = float((np.sum(null_resid_arr >= observed_resid) + 1) / (null_resid_arr.size + 1))

        # task bootstrap on the observed statistic
        rng = np.random.default_rng(0)
        boot = np.array([
            lofo_delta(shifts[rng.integers(0, len(tasks), len(tasks))], eval_idx, neut_idx)
            for _ in range(args.n_boot)
        ])
        lo, hi = np.percentile(boot, [2.5, 97.5])

        # per-layer delta curve
        curve = [lofo_delta(shifts_all[:, :, L], eval_idx, neut_idx) for L in range(n_layers)]

        # position / length control: does the axis projection track task-span start?
        means = framing_means(shifts)
        axis_full = unit(means[eval_idx].mean(axis=0) - means[neut_idx].mean(axis=0))
        proj = unit(shifts) @ axis_full                                  # [T, F]
        pos_r = float(np.corrcoef(proj.ravel(), starts.ravel())[0, 1])
        fam_start = {
            "eval": float(starts[:, eval_idx].mean()),
            "neutral": float(starts[:, neut_idx].mean()),
        }

        entry = {
            "model": model, "params_b": params, "n_tasks": len(tasks),
            "n_framings": len(framings), "observed_delta": observed,
            "delta_ci": [float(lo), float(hi)],
            "exact_permutation_p": p_value, "n_permutations": int(null_arr.size),
            "null_mean": float(null_arr.mean()), "null_max": float(null_arr.max()),
            "delta_position_residualized": observed_resid,
            "exact_permutation_p_position_residualized": p_resid,
            "null_max_position_residualized": float(null_resid_arr.max()),
            "layer_curve_delta": [float(v) for v in curve],
            "proj_vs_task_start_r": pos_r, "mean_task_start": fam_start,
        }
        report["models"].append(entry)
        print(f"{model}: delta={observed:+.4f} CI[{lo:+.4f},{hi:+.4f}] p={p_value:.4f} | "
              f"pos-residualized delta={observed_resid:+.4f} p={p_resid:.4f} | "
              f"null max {null_arr.max():+.4f} | pos_r={pos_r:+.3f}", flush=True)

    Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"\n{'=' * 104}")
    print(f"{'model':<15}{'size':>6}{'delta':>9}{'95% CI':>20}{'p':>8}"
          f"{'delta|pos':>11}{'p|pos':>8}{'null max':>10}{'pos r':>8}")
    for e in report["models"]:
        ci = e["delta_ci"]
        print(f"{e['model']:<15}{e['params_b']:>5.1f}B{e['observed_delta']:>9.4f}"
              f"  [{ci[0]:>+.4f},{ci[1]:>+.4f}]{e['exact_permutation_p']:>8.4f}"
              f"{e['delta_position_residualized']:>11.4f}"
              f"{e['exact_permutation_p_position_residualized']:>8.4f}"
              f"{e['null_max']:>10.4f}{e['proj_vs_task_start_r']:>8.3f}")
    print("\ndelta > 0 with small exact p => an eval axis that generalizes to unseen eval wordings.")
    print("delta ~ null => the 'eval direction' was the footprint of particular prefixes.")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
