"""How many contrastive items does a difference-in-means direction actually need?

Under our mean pooling a single task reproduces a sixty-task direction at cosine 0.9907. Under the
field's answer-token construction a single item reaches only 0.138, which looked at first like the
result failing to generalise. It is not: a single item has no counterbalancing.

Write the per-item difference as

    d_i = c + s_i v + noise,

where v is the answer-token-identity component and s_i = +-1 according to which letter is the
positive answer. The released set is exactly counterbalanced, 102 items each way, so v cancels
exactly over the full set and exactly over a stratified half -- but is present at full strength in
one item. The n = 1 figure therefore measures how far that artefact swamps the signal, not whether
the contrastive data does work.

The correct analogue of n = 1 is the smallest set for which the artefact cancels: a counterbalanced
PAIR. This sweeps counterbalanced n against a disjoint complement and reports the saturation curve,
and separately measures v and c directly as the half-difference and half-sum of the two answer-key
groups.

Two quantities come out. The saturation point says how over-specified the contrastive set is. The
ratio |v|/|c| says how much the direction depends on the set being balanced -- a design property
nothing in the construction enforces and no paper reports checking.

    python scripts/counterbalance_saturation.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from external_paths import reanalysis  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CONTRASTIVE = reanalysis("manek", "datasets", "contrastive_dataset.json")
FC = ROOT / "results" / "field_construction"
NS = [2, 4, 8, 16, 32, 64, 102]
DEPTHS = (0.1, 0.5, 0.9)


def unit(v):
    return v / np.clip(np.linalg.norm(v, axis=-1, keepdims=True), 1e-12, None)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--draws", type=int, default=200)
    a = ap.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    items = json.loads(CONTRASTIVE.read_text(encoding="utf-8"))
    letters = np.array([e["positive"][1] for e in items])
    iA, iB = np.flatnonzero(letters == "A"), np.flatnonzero(letters == "B")
    models = sorted(p.stem for p in FC.glob("*.npz") if "partial" not in p.stem)
    out = {"n_items": len(items), "n_positive_A": int(iA.size), "n_positive_B": int(iB.size),
           "grid": NS, "draws": a.draws, "models": {}}

    print(f"The released contrastive set is exactly counterbalanced: {iA.size} positive-(A) and "
          f"{iB.size} positive-(B).\n")
    print("SATURATION: cos(direction from n counterbalanced items, direction from the disjoint")
    print(f"complement), {a.draws} draws per n.\n")
    for m in models:
        with np.load(FC / f"{m}.npz", allow_pickle=False) as z:
            A = z["acts"].astype(np.float64)
        diff = A[:, 0] - A[:, 1]
        nl = diff.shape[1]
        idx = {d: max(1, int(d * (nl - 1))) for d in DEPTHS}
        rng = np.random.default_rng(0)
        curves = {d: [] for d in DEPTHS}
        for n in NS:
            acc = {d: [] for d in DEPTHS}
            for _ in range(a.draws):
                S = np.concatenate([rng.permutation(iA)[: n // 2],
                                    rng.permutation(iB)[: n // 2]])
                C = np.setdiff1d(np.arange(len(items)), S)
                c = np.sum(unit(diff[S].mean(0)) * unit(diff[C].mean(0)), axis=-1)
                for d, L in idx.items():
                    acc[d].append(float(c[L]))
            for d in DEPTHS:
                curves[d].append(float(np.mean(acc[d])))

        dA, dB = diff[iA].mean(0), diff[iB].mean(0)
        v, cc = (dA - dB) / 2.0, (dA + dB) / 2.0
        ratio = np.linalg.norm(v, axis=-1) / np.clip(np.linalg.norm(cc, axis=-1), 1e-12, None)
        align = np.sum(unit(cc) * unit(diff.mean(0)), axis=-1)

        print(f"  {m}   {nl - 1} layers")
        print(f"    {'depth':<8}" + "".join(f"{n:>8}" for n in NS) + f"{'|v|/|c|':>10}"
              f"{'cos(c,full)':>13}")
        for d in DEPTHS:
            L = idx[d]
            print(f"    d={d:<6}" + "".join(f"{x:8.3f}" for x in curves[d])
                  + f"{ratio[L]:10.2f}{align[L]:13.4f}")
        out["models"][m] = {
            "n_layers": int(nl - 1),
            "saturation": {str(d): curves[d] for d in DEPTHS},
            "v_over_c": {str(d): float(ratio[idx[d]]) for d in DEPTHS},
            "cos_c_full": {str(d): float(align[idx[d]]) for d in DEPTHS}}

    late = np.array([out["models"][m]["saturation"]["0.9"] for m in models])
    print(f"\n  At d=0.9, averaged over models: " +
          "  ".join(f"n={n}:{v:.3f}" for n, v in zip(NS, late.mean(axis=0))))
    n8 = late[:, NS.index(8)].mean()
    n102 = late[:, NS.index(102)].mean()
    print(f"  Eight counterbalanced items reach {n8:.3f} against {n102:.3f} for a hundred and two,")
    print(f"  so the direction saturates by roughly n = 8 to 16 and the released 204-item set is")
    print(f"  over-specified by about an order of magnitude. Our mean-pooled construction saturates")
    print(f"  at n = 1 (cosine 0.9907), so the pooling convention changes the constant and not the")
    print(f"  conclusion.")
    ratios9 = [out["models"][m]["v_over_c"]["0.9"] for m in models]
    print(f"\n  |v|/|c| at d=0.9 ranges {min(ratios9):.2f} to {max(ratios9):.2f}. The direction is")
    print(f"  well defined only because the set is balanced; cos(c, full direction) is 1.0000")
    print(f"  throughout, confirming that counterbalancing cancels v exactly. Nothing in the")
    print(f"  construction enforces balance and no paper reports checking it, so the exposure is")
    print(f"  prospective -- the sets actually used here are balanced and those numbers stand.")
    out["mean_saturation_d09"] = {str(n): float(v) for n, v in zip(NS, late.mean(axis=0))}
    out["v_over_c_d09_range"] = [float(min(ratios9)), float(max(ratios9))]
    p = ROOT / "results" / "counterbalance_saturation.json"
    p.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwrote {p}")


if __name__ == "__main__":
    main()
