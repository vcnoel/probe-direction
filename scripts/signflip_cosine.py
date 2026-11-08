"""How far a sign-flip (label-permuted) direction is from the direction it is meant to null.

Post-submission descriptive check (docs/post-submission-checks.md), not registered. The paired
sign-flip control of scripts/released_label_permuted.py builds d(s) = mean_i s_i (a_i - b_i) with
s_i = +-1 per item. If the per-item differences share a large common component, a random sign
vector leaves part of it in d(s), and the "null" direction is still partly the real one. This
measures that, with exactly the released script's draws (seed 0 per model-rendering cell, 200
sequential draws of s):

  |cos(d(s), d_real)| per draw and scored hidden state (1 .. L-2, the layers the released
  statistic maximises over): median over draws and states, the largest per-state median, and
  the largest value; |cos| between independent draws (draw b against draw b + 100); 1/sqrt(n).
  Beside it, from results/released_label_permuted.json, the permutation mean, the AR(1) mean and
  their difference per cell, and from the ladder cells (results/post_submission/cells/<ladder>.npz,
  key cos_signflip) the per-model median |cos| between the framing sign-flip draws and the real
  wrapper directions.

    EAS_ACTIVATION_ROOT=<root> python scripts/signflip_cosine.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from pod_ladders import LADDERS  # noqa: E402
from result_io import write_result  # noqa: E402

ACT = Path(os.environ.get("EAS_ACTIVATION_ROOT", str(ROOT)))
CACHE = ACT / "results" / "released_label_permuted"
OUT = ROOT / "results" / "post_submission" / "signflip_cosine.json"
DRAWS = 200                  # scripts/released_label_permuted.py


def abscos(a, b):
    return np.abs((a * b).sum(-1)) / (np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1))


def cell(path):
    with np.load(path, allow_pickle=False) as z:
        D = z["diffs"].astype(np.float64)                  # [n, L+1, d]
    n, nlay, _ = D.shape
    real = D.mean(0)
    rng = np.random.default_rng(0)                         # the released draws, in order
    flips = np.stack([np.einsum("i,ild->ld", rng.choice([-1.0, 1.0], size=n), D) / n
                      for _ in range(DRAWS)])              # [draws, L+1, d]
    sl = slice(1, nlay - 1)
    c = abscos(flips[:, sl], real[None, sl])               # [draws, scored states]
    between = abscos(flips[:DRAWS // 2, sl], flips[DRAWS // 2:, sl])
    per_state = np.median(c, axis=0)
    return {"n_items": n, "n_scored_states": int(c.shape[1]),
            "abscos_flip_real_median": float(np.median(c)),
            "abscos_flip_real_max_per_state_median": float(per_state.max()),
            "abscos_flip_real_argmax_state": int(per_state.argmax()) + 1,
            "abscos_flip_real_max": float(c.max()),
            "abscos_between_flips_median": float(np.median(between)),
            "one_over_sqrt_n": float(1 / np.sqrt(n))}


def main() -> None:
    rel = json.loads((ROOT / "results" / "released_label_permuted.json").read_text(encoding="utf-8"))
    released = {}
    for model, rec in rel["per_model"].items():
        for variant, v in rec["variants"].items():
            r = cell(CACHE / f"{model}__{variant}__diffs.npz")
            r.update({"perm_mean": v["perm_mean"], "ar1_mean": rec["ar1_mean"],
                      "perm_minus_ar1": v["perm_minus_ar1"], "rebuilt_statistic": v["rebuilt_statistic"]})
            released[f"{model}__{variant}"] = r
            print(f"{model:13s} {variant:6s} n={r['n_items']}  |cos(flip, real)| median "
                  f"{r['abscos_flip_real_median']:.3f} (max state median "
                  f"{r['abscos_flip_real_max_per_state_median']:.3f})  between flips "
                  f"{r['abscos_between_flips_median']:.3f}  1/sqrt(n) {r['one_over_sqrt_n']:.3f}  "
                  f"perm {r['perm_mean']:.3f} vs AR(1) {r['ar1_mean']:.3f}", flush=True)
    ladders = {}
    for ladder in LADDERS:
        with np.load(ROOT / "results" / "post_submission" / "cells" / f"{ladder}.npz",
                     allow_pickle=False) as z:
            ladders[ladder] = {str(m): float(c) for m, c in zip(z["models"], z["cos_signflip"])}
        print(f"{ladder:16s} framing sign-flip median |cos| with real: " +
              "  ".join(f"{m} {c:.3f}" for m, c in ladders[ladder].items()), flush=True)
    write_result(OUT, {"released_cells": released, "ladder_cells_cos_signflip": ladders,
                       "draws": DRAWS,
                       "check": "post-submission descriptive check, not registered "
                                "(docs/post-submission-checks.md)"},
                 kind="measured",
                 statistic="|cosine| between paired sign-flip directions and the real direction "
                           "(and between independent sign-flip draws), per hidden state",
                 note="released cells: the released 204-item paired construction, hidden states "
                      "1..L-2 as the released statistic; ladder cells: median over draws and "
                      "wrappers as stored by scripts/ladder_audit_cells.py",
                 pooling="released: answer-token activations (paired differences); ladder cells: "
                         "validated framing means",
                 layers="1..L-2 (released cells); as stored (ladder cells)",
                 scoring_array="results/released_label_permuted/*__diffs.npz; "
                               "results/post_submission/cells/*.npz cos_signflip",
                 data_root="gstudy + pod",
                 model_population=sorted({k.split("__")[0] for k in released}
                                         | {m for v in ladders.values() for m in v}))
    print(f"wrote {OUT.relative_to(ROOT).as_posix()}")


if __name__ == "__main__":
    main()
