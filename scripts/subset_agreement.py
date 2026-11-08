"""Framing-disjoint wrapper subsets against random disjoint subsets, and flat against crossed k.

Post-submission descriptive check (docs/post-submission-checks.md), not registered; first computed
by the adversarial audit. Two 3 x 3 wrapper subsets that share no framing (3 evaluation and 3
deployment framings each, complementary) are the strictest test that two wrapper averages rank the
models alike. Random disjoint subsets of the same size (9 wrappers) share framings, so they agree
more for that reason alone. Recomputed here from the per-wrapper statistic of the ladder cells
(results/post_submission/cells/<ladder>.npz, real_full), with the stored values of
results/post_submission/arm_disjoint.json beside them:

  framing_disjoint   mean Kendall tau across models between the two subset averages, over the
                     200 distinct framing-disjoint 3 x 3 pairs (deterministic, must match)
  random_disjoint_9  mean tau for 4000 random pairs of disjoint 9-wrapper subsets (seed 0 here; the
                     stored value came from a stream shared with other analyses, so it agrees only
                     to Monte Carlo error)

and, from results/post_submission/matched_null.json, the number of wrappers needed for E rho^2 =
0.80 with flat wrapper sampling (k080) against the crossed design (k080_crossed_per_arm framings
per arm, i.e. that number squared in wrappers).

    python scripts/subset_agreement.py
"""

from __future__ import annotations

import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.stats import kendalltau

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from pod_ladders import LADDERS  # noqa: E402
from result_io import write_result  # noqa: E402

CELLS = ROOT / "results" / "post_submission" / "cells"
OUT = ROOT / "results" / "post_submission" / "subset_agreement.json"
N_RANDOM, SEED = 4000, 0


def taus(full, n_e, n_d):
    M = full.shape[0]
    g = full.reshape(M, n_e, n_d)
    seen, fd = set(), []
    for E1 in combinations(range(n_e), n_e // 2):
        E2 = tuple(i for i in range(n_e) if i not in E1)
        for D1 in combinations(range(n_d), n_d // 2):
            D2 = tuple(i for i in range(n_d) if i not in D1)
            key = frozenset([(E1, D1), (E2, D2)])
            if key in seen:
                continue
            seen.add(key)
            a = g[:, list(E1)][:, :, list(D1)].reshape(M, -1).mean(1)
            b = g[:, list(E2)][:, :, list(D2)].reshape(M, -1).mean(1)
            fd.append(kendalltau(a, b).statistic)
    rng = np.random.default_rng(SEED)
    k = (n_e // 2) * (n_d // 2)
    rd = []
    for _ in range(N_RANDOM):
        p = rng.permutation(full.shape[1])
        rd.append(kendalltau(full[:, p[:k]].mean(1), full[:, p[k:2 * k]].mean(1)).statistic)
    return len(seen), float(np.mean(fd)), float(np.mean(rd)), k


def main() -> None:
    stored = json.loads((ROOT / "results" / "post_submission" / "arm_disjoint.json")
                        .read_text(encoding="utf-8"))["ladders"]
    mn = json.loads((ROOT / "results" / "post_submission" / "matched_null.json")
                    .read_text(encoding="utf-8"))["ladders"]
    out, pop = {}, []
    for ladder in LADDERS:
        with np.load(CELLS / f"{ladder}.npz", allow_pickle=False) as z:
            full = z["real_full"].astype(float)
            n_e, n_d = len(z["eval_arms"]), len(z["deploy_arms"])
            models = [str(m) for m in z["models"]]
        n_pairs, fd, rd, k = taus(full, n_e, n_d)
        st = stored[ladder]
        real = mn[ladder]["real"]
        kc = real["k080_crossed_per_arm"]
        out[ladder] = {
            "n_framing_disjoint_pairs": n_pairs, "subset_size": k,
            "framing_disjoint_mean_tau": fd, "random_disjoint_mean_tau": rd,
            "stored_framing_disjoint_mean_tau": st["arm_disjoint_3x3_mean_tau"],
            "stored_random_disjoint_mean_tau": st["random_disjoint"][str(k)]["mean_tau"],
            "k080_flat_wrappers": real["k080"],
            "k080_crossed_framings_per_arm": kc,
            "k080_crossed_wrappers": kc * kc if kc else None,
            "erho_k1_flat": real["erho_k1"], "erho_k1_crossed": real["erho_crossed_k1"]}
        pop += models
        o = out[ladder]
        print(f"{ladder:16s} tau framing-disjoint {fd:.3f} (stored {o['stored_framing_disjoint_mean_tau']:.3f}) "
              f"vs random {rd:.3f} (stored {o['stored_random_disjoint_mean_tau']:.3f}) | k080 flat "
              f"{o['k080_flat_wrappers']} crossed {kc} per arm ({o['k080_crossed_wrappers']} wrappers)",
              flush=True)
    write_result(OUT, {"ladders": out, "n_random_pairs": N_RANDOM,
                       "check": "post-submission descriptive check, not registered; first computed "
                                "by the adversarial audit (docs/post-submission-checks.md)"},
                 kind="measured",
                 statistic="mean Kendall tau across models between two disjoint wrapper-subset "
                           "averages of the paper's statistic; k for E rho^2 = 0.80, flat and crossed",
                 note="k values read from results/post_submission/matched_null.json (real)",
                 pooling="validated (Qwen2.5: lf scoring, all-token directions, as pod_ladders)",
                 layers="ALL",
                 scoring_array="results/post_submission/cells/*.npz real_full",
                 data_root="pod + gstudy", model_population=sorted(set(pop)))
    print(f"wrote {OUT.relative_to(ROOT).as_posix()}")


if __name__ == "__main__":
    main()
