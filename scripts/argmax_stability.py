"""Is a reported 'peak layer' identified, and is a reported 'peak score' biased?

Published claims of the form "the layer at which evaluation awareness is most linearly
recoverable moves from late layers in small models to early layers in large ones" rest on an
**argmax over layers reported without uncertainty**, and on a **maximum over layers reported
as a point estimate**. Both are unsafe:

* an argmax over 24-64 candidate layers on a nearly-flat curve is dominated by noise, and
  boundary argmaxes (layer 1-2, or the final layer) are the signature of a slight tilt rather
  than a localized peak;
* a maximum over layers is upward biased by selection, so a curve of maxima across model
  sizes can appear non-monotonic purely from selection noise.

This script quantifies both on our own per-layer curves, which are the same kind of object:

1. ARGMAX STABILITY - task bootstrap of the whole curve, recording where it peaks. Reports
   the peak-depth interval and how much of the network it spans. If that interval covers a
   large fraction of the depth, "the peak moved" is not supported.
2. SELECTION BIAS - split-half: choose the peak layer on half the tasks, evaluate the
   statistic at that layer on the held-out half. The gap between the naive maximum and the
   cross-validated value is the selection bias in a reported peak score.

    python scripts/argmax_stability.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "multiframing"
FRAMINGS = ROOT / "data" / "framings_multi.json"
MODELS = [("qwen2.5-0.5b", 0.5, 24), ("qwen2.5-1.5b", 1.5, 28),
          ("qwen2.5-3b", 3.0, 36), ("qwen2.5-7b", 7.0, 28)]


def unit(a: np.ndarray) -> np.ndarray:
    return a / np.clip(np.linalg.norm(a, axis=-1, keepdims=True), 1e-12, None)


def load_units(model: str, fam: dict[str, str]):
    """Per-(task, framing, layer) unit shift vs the deploy anchor."""
    payload = np.load(RESULTS / model / "activations.npz", allow_pickle=False)
    vectors = payload["mean_vec"].astype(np.float32)
    task_ids = [str(t) for t in payload["task_id"]]
    conds = [str(c) for c in payload["condition"]]
    framings = sorted(set(conds))
    tasks = sorted(set(task_ids))
    index = {(t, c): i for i, (t, c) in enumerate(zip(task_ids, conds))}
    keep = [t for t in tasks if all((t, f) in index for f in framings)]
    n_layers, dim = vectors.shape[1], vectors.shape[2]
    cube = np.zeros((len(keep), len(framings), n_layers, dim), dtype=np.float32)
    for ti, task in enumerate(keep):
        for fi, framing in enumerate(framings):
            cube[ti, fi] = vectors[index[(task, framing)]]
    anchor = [i for i, f in enumerate(framings) if fam.get(f) == "deploy"]
    shifts = cube - cube[:, anchor].mean(axis=1, keepdims=True)
    del cube
    # Drop layer 0: token embeddings are context-independent, so the task-token shift
    # between framings is identically zero there and the unit direction is undefined.
    shifts = shifts[:, :, 1:, :]
    U = unit(shifts.astype(np.float32))
    ev = [i for i, f in enumerate(framings) if fam.get(f) == "eval"]
    nu = [i for i, f in enumerate(framings) if fam.get(f) == "neutral"]
    return U, ev, nu, len(keep), n_layers - 1


def curve_from_counts(U: np.ndarray, counts: np.ndarray, ev, nu) -> np.ndarray:
    """Per-layer LOFO delta for a (possibly bootstrap-weighted) task set.

    Uses the identity that the mean projection of held-out framing f onto an axis equals
    (sum_t c_t U[t,f]) . axis / n, so everything reduces to the framing-sum vectors.
    """
    n = counts.sum()
    S = np.tensordot(counts.astype(np.float32), U, axes=([0], [0]))     # [F, L, d]
    M = unit(S)                                                        # framing directions
    n_layers = U.shape[2]
    out = np.empty(n_layers)
    norms = np.linalg.norm(S, axis=-1)                                 # [F, L]
    for L in range(n_layers):
        m = M[:, L, :]                                                 # [F, d]
        gram = m @ m.T
        deltas = []
        for he in ev:
            for hn in nu:
                tr_e = [i for i in ev if i != he]
                tr_n = [i for i in nu if i != hn]
                a = m[tr_e].mean(axis=0) - m[tr_n].mean(axis=0)
                a_norm = np.linalg.norm(a)
                if a_norm < 1e-9:
                    continue
                # (S_f . axis)/n  =  |S_f| * (m_f . a) / (|a| * n)
                s_e = norms[he, L] * float(gram[he, tr_e].mean() - gram[he, tr_n].mean()) / a_norm
                s_n = norms[hn, L] * float(gram[hn, tr_e].mean() - gram[hn, tr_n].mean()) / a_norm
                deltas.append((s_e - s_n) / n)
        out[L] = float(np.mean(deltas))
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-boot", type=int, default=400)
    parser.add_argument("--n-split", type=int, default=200)
    parser.add_argument("--out", default=str(ROOT / "results" / "argmax_stability.json"))
    args = parser.parse_args()

    spec = json.loads(FRAMINGS.read_text(encoding="utf-8"))
    fam = {k: v["family"] for k, v in spec.items() if not k.startswith("_")}
    rng = np.random.default_rng(0)
    report = {"models": []}

    for model, params, depth in MODELS:
        if not (RESULTS / model / "activations.npz").exists():
            continue
        U, ev, nu, n_tasks, n_layers = load_units(model, fam)
        full = curve_from_counts(U, np.ones(n_tasks), ev, nu)
        naive_peak_layer = int(np.argmax(full))
        naive_peak_value = float(full[naive_peak_layer])

        # 1. argmax stability under task bootstrap
        peaks = np.empty(args.n_boot, dtype=int)
        for b in range(args.n_boot):
            idx = rng.integers(0, n_tasks, n_tasks)
            counts = np.bincount(idx, minlength=n_tasks)
            peaks[b] = int(np.argmax(curve_from_counts(U, counts, ev, nu)))
        rel = peaks / (n_layers - 1)
        lo, hi = np.percentile(rel, [2.5, 97.5])

        # 2. selection bias: pick the peak layer on half the tasks, score on the other half
        naive, honest = [], []
        for _ in range(args.n_split):
            perm = rng.permutation(n_tasks)
            a, b_ = perm[: n_tasks // 2], perm[n_tasks // 2:]
            ca = np.bincount(a, minlength=n_tasks)
            cb = np.bincount(b_, minlength=n_tasks)
            curve_a = curve_from_counts(U, ca, ev, nu)
            curve_b = curve_from_counts(U, cb, ev, nu)
            chosen = int(np.argmax(curve_a))
            naive.append(float(curve_a[chosen]))          # in-sample maximum
            honest.append(float(curve_b[chosen]))          # held-out value at that layer
        bias = float(np.mean(naive) - np.mean(honest))

        entry = {
            "model": model, "params_b": params, "layers": depth, "n_layers_probed": n_layers,
            "naive_peak_layer": naive_peak_layer,
            "naive_peak_relative_depth": naive_peak_layer / (n_layers - 1),
            "naive_peak_value": naive_peak_value,
            "curve_min": float(full.min()), "curve_max": float(full.max()),
            "curve_range": float(full.max() - full.min()),
            "argmax_relative_depth_ci": [float(lo), float(hi)],
            "argmax_ci_width_fraction_of_depth": float(hi - lo),
            "argmax_modal_layer": int(np.bincount(peaks).argmax()),
            "distinct_argmax_layers": int(len(np.unique(peaks))),
            "selection_bias_in_peak_value": bias,
            "insample_peak_mean": float(np.mean(naive)),
            "heldout_at_selected_layer_mean": float(np.mean(honest)),
        }
        report["models"].append(entry)
        print(f"{model}: naive peak at layer {naive_peak_layer}/{n_layers - 1} "
              f"(rel {entry['naive_peak_relative_depth']:.2f}), "
              f"argmax rel-depth 95% CI [{lo:.2f},{hi:.2f}] "
              f"spanning {(hi - lo) * 100:.0f}% of depth; "
              f"selection bias {bias:+.4f}", flush=True)

    Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"\n{'=' * 104}")
    print(f"{'model':<15}{'L':>4}{'peak':>6}{'rel':>6}{'argmax 95% CI':>18}{'span':>8}"
          f"{'range':>9}{'in-samp':>9}{'held-out':>10}{'bias':>8}")
    for e in report["models"]:
        ci = e["argmax_relative_depth_ci"]
        print(f"{e['model']:<15}{e['n_layers_probed']:>4}{e['naive_peak_layer']:>6}"
              f"{e['naive_peak_relative_depth']:>6.2f}"
              f"  [{ci[0]:>.2f},{ci[1]:>.2f}]{e['argmax_ci_width_fraction_of_depth']:>10.0%}"
              f"{e['curve_range']:>9.3f}{e['insample_peak_mean']:>9.3f}"
              f"{e['heldout_at_selected_layer_mean']:>10.3f}"
              f"{e['selection_bias_in_peak_value']:>8.3f}")
    print("\nA wide argmax CI means a reported 'peak layer' is not identified by the data.")
    print("A positive bias means a reported 'peak score' is inflated by layer selection.")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
