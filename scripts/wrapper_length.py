"""Does wrapper length predict the per-wrapper scaling slope? A length confound across wrappers.

Directions are differences of MEAN-pooled activations, so a longer framing puts more framing tokens
and proportionally fewer task tokens into each pooled vector. If length drives the per-wrapper
slope, the sign the paper attributes to the wording could partly be the word count. On Qwen2.5 the
longest deployment framing (deploy_a) is also the one with the most negative mean slope, and the
6/6 deployment-fixed sign span there hinges on eval_benchmark/deploy_a at r = +0.0037.

Length is an ARM-level covariate: every wrapper sharing an evaluation framing shares its length.
Correlations over the 36 wrappers are therefore tested against an arm-level permutation null
(evaluation lengths shuffled among evaluation framings and deployment lengths among deployment
framings), not against 34 degrees of freedom. Lengths are counted in tokens with each ladder's own
tokenizer, loaded offline from the local model cache (words if it is absent), over the framing text
itself: the system message plus the user template with the task removed.

Reads results/ladder_curves.json (per-wrapper statistic for every ladder) and
data/framings_multi.json.

    HF_HUB_OFFLINE=1 python scripts/wrapper_length.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from result_io import write_result  # noqa: E402

TOKENIZER = {"qwen2.5": "Qwen/Qwen2.5-0.5B-Instruct", "qwen3": "Qwen/Qwen3-0.6B",
             "qwen3.5": "Qwen/Qwen3.5-0.8B", "olmo2": "allenai/OLMo-2-0425-1B-Instruct"}
N_PERM = 20000


def family(ladder):
    return ladder.split("_")[0]


def lengths(framings, fam):
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    texts = {k: v["system"] + "\n" + v["user_template"].replace("{task}", "")
             for k, v in framings.items() if not k.startswith("_")}
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(TOKENIZER[fam])
        return {k: len(tok(t, add_special_tokens=False)["input_ids"]) for k, t in texts.items()}, \
            f"tokens ({TOKENIZER[fam]})"
    except Exception as exc:  # tokenizer not cached: fall back to words, and say so
        return {k: len(t.split()) for k, t in texts.items()}, f"words (tokenizer unavailable: {exc!r})"


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    framings = json.loads((ROOT / "data" / "framings_multi.json").read_text(encoding="utf-8"))
    lc = json.loads((ROOT / "results" / "ladder_curves.json").read_text(encoding="utf-8"))["ladders"]
    rng = np.random.default_rng(0)
    out = {}
    for name, d in lc.items():
        L, unit = lengths(framings, family(name))
        pairs = [tuple(p) for p in d["pairs"]]
        ev = sorted({e for e, _ in pairs})
        dp = sorted({p for _, p in pairs})
        x = np.array(d["statistic"])
        lp = np.log10(d["params_b"])
        a = lp - lp.mean()
        slope = (a @ (x - x.mean(axis=0))) / (a ** 2).sum()
        r = np.array(d["r"])
        le = np.array([L[e] for e in ev], float)
        ld = np.array([L[p] for p in dp], float)
        ie = np.array([ev.index(e) for e, _ in pairs])
        idp = np.array([dp.index(p) for _, p in pairs])
        covs = {"eval_length": lambda E, D: E[ie], "deploy_length": lambda E, D: D[idp],
                "eval_minus_deploy": lambda E, D: E[ie] - D[idp],
                "eval_plus_deploy": lambda E, D: E[ie] + D[idp]}
        rec = {"length_unit": unit, "eval_lengths": dict(zip(ev, le.tolist())),
               "deploy_lengths": dict(zip(dp, ld.tolist())), "covariates": {}}
        for cname, f in covs.items():
            obs = {}
            for yname, yv in (("slope", slope), ("r", r)):
                c0 = float(np.corrcoef(f(le, ld), yv)[0, 1]) if np.std(f(le, ld)) > 0 else 0.0
                null = []
                for _ in range(N_PERM):
                    cv = f(rng.permutation(le), rng.permutation(ld))
                    null.append(np.corrcoef(cv, yv)[0, 1] if np.std(cv) > 0 else 0.0)
                null = np.abs(np.array(null))
                obs[yname] = {"pearson": c0, "spearman": float(spearmanr(f(le, ld), yv).statistic),
                              "arm_permutation_p_two_sided": float((null >= abs(c0) - 1e-12).mean())}
            rec["covariates"][cname] = obs
        # arm means against arm lengths, n = 6 each
        g = slope.reshape(len(ev), len(dp))
        rec["arm_level"] = {
            "eval_arm_mean_slope_vs_length_r": float(np.corrcoef(le, g.mean(axis=1))[0, 1]),
            "deploy_arm_mean_slope_vs_length_r": float(np.corrcoef(ld, g.mean(axis=0))[0, 1]),
            "eval_arm_mean_slope": dict(zip(ev, g.mean(axis=1).tolist())),
            "deploy_arm_mean_slope": dict(zip(dp, g.mean(axis=0).tolist())),
            "longest_deploy": dp[int(np.argmax(ld))],
            "longest_deploy_rank_of_mean_slope_from_bottom": int(
                np.argsort(np.argsort(g.mean(axis=0)))[int(np.argmax(ld))]) + 1,
        }
        # what the deployment-fixed span count becomes without the longest deployment framing
        rr = r.reshape(len(ev), len(dp))
        keep = [j for j in range(len(dp)) if j != int(np.argmax(ld))]
        rec["eval_fixed_spanning_without_longest_deploy"] = int(sum(
            rr[i, keep].max() > 0 > rr[i, keep].min() for i in range(len(ev))))
        rec["deploy_fixed_spanning_without_longest_deploy"] = int(sum(
            rr[:, j].max() > 0 > rr[:, j].min() for j in keep))
        out[name] = rec
        c = rec["covariates"]
        print(f"  {name}: slope~deploy_len r {c['deploy_length']['slope']['pearson']:+.2f} "
              f"(p {c['deploy_length']['slope']['arm_permutation_p_two_sided']:.3f})  "
              f"slope~eval_len r {c['eval_length']['slope']['pearson']:+.2f} "
              f"(p {c['eval_length']['slope']['arm_permutation_p_two_sided']:.3f})  "
              f"slope~(e-d) r {c['eval_minus_deploy']['slope']['pearson']:+.2f} "
              f"(p {c['eval_minus_deploy']['slope']['arm_permutation_p_two_sided']:.3f})  [{unit[:28]}]")
    write_result(ROOT / "results" / "post_submission" / "wrapper_length.json",
                 {"ladders": out, "n_permutations": N_PERM}, kind="derived",
                 sources=["ladder_curves.json"],
                 statistic="correlation of per-wrapper slope and r with framing length, "
                           "arm-level permutation null",
                 note="post-submission audit; lengths from data/framings_multi.json")
    print("wrote results/post_submission/wrapper_length.json")


if __name__ == "__main__":
    main()
