"""Difference-vector geometry with the confound controls that decide the empirical sign.

Motivation
----------
Earlier analyses computed the "eval" direction as ``eval - deploy`` and the nuisance
direction as ``deploy - placebo``. Two problems:

1. The eval-specific contrast is ``eval - placebo``: the placebo wrapper was designed to
   hold format/length/formality fixed, so this difference cancels the shared wrapper
   component and is *anchor-free* (no sign-convention trap). It was never computed.
2. Cosine consistency is scale-invariant but not SNR-invariant. ``eval - placebo`` has a
   smaller norm than ``eval - deploy``, so a lower consistency could be pure attenuation.

This script therefore reports, for every contrast:

* ``C4``        mean cross-task cosine of the unit per-task direction (consistency)
* ``C4_xblock`` the same but only over pairs from *different* benchmark blocks, because
                45/60 tasks come from three benchmarks and within-benchmark items are
                not exchangeable
* ``C10``       participation ratio of the per-task directions (effective rank)
* ``norm``      mean pre-normalization shift norm, and its ratio to the deploy-state norm
* ``rel``       split-half reliability of the per-task direction (odd vs even layers of
                the band), and the attenuation-corrected ``C4 / rel``
* raw and per-dimension **whitened** variants (whitening removes anisotropy / rogue dims)
* a per-layer ``C4`` curve, since a fixed "last 50%" band is arbitrary across depths
* **cluster bootstrap** over benchmark blocks in addition to the plain task bootstrap

Preregistered estimator choice: the primary statistic uses the *band-averaged direction*
(cosine-of-mean), and the layer curve exposes the per-layer (mean-of-cosines) alternative.

Usage
-----
    python scripts/bootstrap_geometry.py --extract      # build compact .npz caches once
    python scripts/bootstrap_geometry.py                # analyze from caches
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"

MODELS = [
    ("local_bench60/qwen2.5-0.5b", "qwen2.5-0.5b", 0.5),
    ("local_bench60/qwen2.5-1.5b", "qwen2.5-1.5b", 1.5),
    ("local_bench60/qwen2.5-3b", "qwen2.5-3b", 3.0),
    ("local_bench60/qwen2.5-7b", "qwen2.5-7b", 7.0),
]

# (name, condition_a, condition_b). eval_vs_placebo is the eval-specific, anchor-free one.
CONTRASTS = [
    ("eval_vs_placebo", "eval", "placebo"),
    ("eval_vs_deploy", "eval", "deploy"),
    ("placebo_vs_deploy", "placebo", "deploy"),
]

BLOCK_OF = {
    "bench_gsm8k_math": "gsm8k",
    "bench_arc_science": "arc",
    "bench_mmlu_knowledge": "mmlu",
}


def task_blocks() -> dict[str, str]:
    """Map task_id -> block; all hand-written categories collapse to 'curated'."""
    blocks: dict[str, str] = {}
    for line in (ROOT / "data" / "tasks_bench.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        blocks[row["id"]] = BLOCK_OF.get(row["category"], "curated")
    return blocks


# --------------------------------------------------------------------- extraction

def extract(rel_dir: str) -> Path:
    """Reduce per-token activations to per-(task, condition, layer) token means.

    This is the sufficient statistic for every geometry statistic here and is ~15x
    smaller than the raw pickle, so it is also the artifact worth backing up off-box.
    """
    src = RESULTS / rel_dir / "token_acts.pkl"
    dst = RESULTS / rel_dir / "mean_vectors.npz"
    with open(src, "rb") as handle:
        records = pickle.load(handle)
    grouped: dict[str, dict[str, np.ndarray]] = {}
    for record in records:
        grouped.setdefault(record["task_id"], {})[record["condition"]] = np.asarray(
            record["tok"], dtype=np.float32
        )
    conditions = sorted({c for value in grouped.values() for c in value})
    tasks = sorted(t for t, value in grouped.items() if len(value) == len(conditions))
    n_layers, _, dim = grouped[tasks[0]][conditions[0]].shape
    means = np.zeros((len(tasks), len(conditions), n_layers, dim), dtype=np.float32)
    for ti, task in enumerate(tasks):
        for ci, condition in enumerate(conditions):
            means[ti, ci] = grouped[task][condition].mean(axis=1)
    np.savez_compressed(
        dst, means=means, tasks=np.array(tasks), conditions=np.array(conditions)
    )
    print(f"{rel_dir}: {len(tasks)} tasks x {len(conditions)} conds x {n_layers} layers "
          f"x {dim} dims -> {dst.name} ({dst.stat().st_size / 1e6:.0f} MB)", flush=True)
    return dst


# ------------------------------------------------------------------------ helpers

def unit(matrix: np.ndarray) -> np.ndarray:
    return matrix / np.clip(np.linalg.norm(matrix, axis=-1, keepdims=True), 1e-12, None)


def participation_ratio(gram: np.ndarray) -> float:
    return float((np.trace(gram) ** 2) / (gram * gram).sum())


def off_diagonal_mean(gram: np.ndarray) -> float:
    n = gram.shape[0]
    return float((gram.sum() - np.trace(gram)) / (n * n - n))


def masked_mean(gram: np.ndarray, mask: np.ndarray) -> float:
    return float(gram[mask].mean()) if mask.any() else float("nan")


def whiten_per_layer(means: np.ndarray) -> np.ndarray:
    """Per-layer, per-dimension z-scoring using statistics over all (task, condition)."""
    flat = means.reshape(-1, means.shape[2], means.shape[3])          # [T*C, L, D]
    mu = flat.mean(axis=0, keepdims=True)
    sd = flat.std(axis=0, keepdims=True)
    return (means - mu[None]) / np.clip(sd[None], 1e-6, None)


def directions(means: np.ndarray, ci_a: int, ci_b: int, layers: np.ndarray) -> np.ndarray:
    """Band-averaged per-task difference direction (pre-normalization)."""
    return (means[:, ci_a, layers] - means[:, ci_b, layers]).mean(axis=1)


# ----------------------------------------------------------------------- analysis

def analyze_model(npz_path: Path, blocks: dict[str, str], n_boot: int, seed: int) -> dict:
    payload = np.load(npz_path, allow_pickle=False)
    means = payload["means"].astype(np.float64)
    tasks = [str(t) for t in payload["tasks"]]
    conditions = [str(c) for c in payload["conditions"]]
    n_layers = means.shape[2]
    band = np.arange(int(n_layers * 0.5), n_layers)
    block_ids = np.array([blocks.get(t, "curated") for t in tasks])
    different_block = block_ids[:, None] != block_ids[None, :]
    rng = np.random.default_rng(seed)

    variants = {"raw": means, "whitened": whiten_per_layer(means)}
    out: dict = {"model": npz_path.parent.name, "n_tasks": len(tasks),
                 "conditions": conditions, "n_layers": n_layers,
                 "band": [int(band[0]), int(band[-1])], "contrasts": {}}

    # deploy-state norm, for a scale reference on the shift norms
    if "deploy" in conditions:
        deploy_norm = float(
            np.linalg.norm(means[:, conditions.index("deploy"), band], axis=-1).mean()
        )
    else:
        deploy_norm = float("nan")
    out["deploy_state_norm"] = deploy_norm

    for name, cond_a, cond_b in CONTRASTS:
        if cond_a not in conditions or cond_b not in conditions:
            continue
        ia, ib = conditions.index(cond_a), conditions.index(cond_b)
        entry: dict = {}
        for variant, source in variants.items():
            raw_dirs = directions(source, ia, ib, band)
            norms = np.linalg.norm(raw_dirs, axis=-1)
            U = unit(raw_dirs)
            gram = U @ U.T

            # split-half reliability: odd vs even layers within the band
            odd, even = band[1::2], band[0::2]
            u_odd = unit(directions(source, ia, ib, odd))
            u_even = unit(directions(source, ia, ib, even))
            reliability = float(np.mean(np.sum(u_odd * u_even, axis=1)))

            c4 = off_diagonal_mean(gram)
            stats = {
                "C4": c4,
                "C4_cross_block": masked_mean(gram, different_block),
                "C10_rank": participation_ratio(gram),
                "shift_norm": float(norms.mean()),
                "shift_norm_rel_deploy": float(norms.mean() / deploy_norm)
                if np.isfinite(deploy_norm) else float("nan"),
                "split_half_reliability": reliability,
                "C4_attenuation_corrected": float(c4 / reliability)
                if abs(reliability) > 1e-6 else float("nan"),
            }

            # bootstraps: plain over tasks, and cluster over benchmark blocks
            n = len(tasks)
            unique_blocks = np.unique(block_ids)
            plain = np.empty((n_boot, 3))
            cluster = np.empty((n_boot, 3))
            for b in range(n_boot):
                idx = rng.integers(0, n, n)
                sub = gram[np.ix_(idx, idx)]
                plain[b] = [off_diagonal_mean(sub), participation_ratio(sub),
                            masked_mean(sub, different_block[np.ix_(idx, idx)])]
                chosen = rng.choice(unique_blocks, unique_blocks.size, replace=True)
                pool = []
                for blk in chosen:
                    members = np.flatnonzero(block_ids == blk)
                    pool.extend(rng.choice(members, members.size, replace=True))
                cidx = np.array(pool)
                csub = gram[np.ix_(cidx, cidx)]
                cluster[b] = [off_diagonal_mean(csub), participation_ratio(csub),
                              masked_mean(csub, different_block[np.ix_(cidx, cidx)])]
            for label, arr in (("task_bootstrap", plain), ("cluster_bootstrap", cluster)):
                lo, hi = np.nanpercentile(arr, [2.5, 97.5], axis=0)
                stats[label] = {
                    "C4_ci": [float(lo[0]), float(hi[0])],
                    "C10_ci": [float(lo[1]), float(hi[1])],
                    "C4_cross_block_ci": [float(lo[2]), float(hi[2])],
                }
            entry[variant] = stats

        # per-layer C4 curve (raw), exposing the mean-of-cosines alternative
        curve = []
        for layer in range(n_layers):
            single = np.array([layer])
            U_layer = unit(directions(means, ia, ib, single))
            curve.append(off_diagonal_mean(U_layer @ U_layer.T))
        entry["layer_curve_C4"] = [float(v) for v in curve]
        out["contrasts"][name] = entry
    return out


def trend(results: list[dict], contrast: str, variant: str, key: str) -> dict:
    sizes = np.array([r["params_b"] for r in results], dtype=float)
    values = np.array([r["contrasts"][contrast][variant][key] for r in results])
    slope = float(np.polyfit(np.log10(sizes), values, 1)[0])
    return {"slope_vs_log10_params": slope,
            "values": [float(v) for v in values],
            "sizes": [float(s) for s in sizes]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--extract", action="store_true", help="build .npz caches then exit")
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default=str(RESULTS / "geometry_controls.json"))
    args = parser.parse_args()

    if args.extract:
        for rel, _, _ in MODELS:
            extract(rel)
        return

    blocks = task_blocks()
    results = []
    for rel, name, params in MODELS:
        npz = RESULTS / rel / "mean_vectors.npz"
        if not npz.exists():
            print(f"missing {npz}; run --extract first")
            continue
        record = analyze_model(npz, blocks, args.n_boot, args.seed)
        record["params_b"] = params
        record["name"] = name
        results.append(record)
        print(f"analyzed {name}", flush=True)

    Path(args.out).write_text(json.dumps({"models": results}, indent=2), encoding="utf-8")

    # ------------------------------------------------------------------ reporting
    for variant in ("raw", "whitened"):
        print(f"\n{'=' * 108}\n{variant.upper()}  (band = late 50% of layers)\n{'=' * 108}")
        header = (f"{'contrast':<18}{'size':>6}{'C4':>18}{'C4 xblock':>18}"
                  f"{'C10 rank':>16}{'rel':>7}{'C4/rel':>8}{'|shift|':>9}{'/deploy':>9}")
        print(header)
        for name, _, _ in CONTRASTS:
            for r in results:
                if name not in r["contrasts"]:
                    continue
                s = r["contrasts"][name][variant]
                ci = s["cluster_bootstrap"]["C4_ci"]
                xb = s["cluster_bootstrap"]["C4_cross_block_ci"]
                print(f"{name:<18}{r['params_b']:>5.1f}B"
                      f"{s['C4']:>8.3f}[{ci[0]:>+.2f},{ci[1]:>+.2f}]"
                      f"{s['C4_cross_block']:>8.3f}[{xb[0]:>+.2f},{xb[1]:>+.2f}]"
                      f"{s['C10_rank']:>16.2f}{s['split_half_reliability']:>7.2f}"
                      f"{s['C4_attenuation_corrected']:>8.3f}"
                      f"{s['shift_norm']:>9.1f}{s['shift_norm_rel_deploy']:>9.3f}")
            print()

    print(f"{'=' * 108}\nSCALE TRENDS (slope vs log10 params)\n{'=' * 108}")
    for variant in ("raw", "whitened"):
        for key in ("C4", "C4_cross_block", "C10_rank", "C4_attenuation_corrected"):
            line = []
            for name, _, _ in CONTRASTS:
                if all(name in r["contrasts"] for r in results):
                    line.append(f"{name}={trend(results, name, variant, key)['slope_vs_log10_params']:+.3f}")
            print(f"  {variant:<9} {key:<26} " + "   ".join(line))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
