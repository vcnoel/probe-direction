"""Is the held-out label recoverable from surface text, and does that make the probe redundant?

Two questions that must not be conflated, because the answers point in opposite directions.

IS THE LABEL SURFACE-RECOVERABLE? Yes, almost perfectly. A bag-of-words classifier
cross-validated on the 400-item split reaches AUROC 0.994 with only 725 features at min_df=5,
which is the figure we lead with; character n-grams reach 0.997 but use 15855 features on 400 items
and are an upper bound rather than the headline. A permuted-label control over twenty permutations
sits at 0.510 +- 0.049, confirming the cross-validation is honest. The discriminating
features are format, not content: `choices`, `statement`, `following`, `would you say` against
`write`, `please`, `me`, `message`. The positive class is synthetic multiple-choice meta-questions
and the negative class is real user requests, so the label is confounded with prompt format and
source at near-ceiling. That is a property of the benchmark, and it explains why arbitrary
directions on activations score as high as they do: anything correlated with format separates
these classes.

DOES THAT MAKE THE PROBE REDUNDANT? The comparison cannot be made fairly, so it is not made. The
cross-validated text model has access to the target labels; the probe is built from a separate
contrastive set and applied to this split zero-shot. A zero-shot text baseline would need a text
training set matched to this distinction, and none exists -- an earlier attempt trained on
contrastive meta-questions against alpaca instructions and reached 0.567, but that number measures
the transfer failure of a mismatched surrogate rather than the accessibility of the signal, since
this split's negative class is ShareGPT-style user messages and nothing like alpaca. It is
therefore not reported. No claim is made that a text model outperforms the probe.

WHY DOES AN ARBITRARY DIRECTION SCORE? Surface separability explains why SOME direction scores; it
does not explain why an arbitrary one does. That needs the format signal to be DIFFUSE in activation
space, and it is: |delta|^2 / tr(Sigma) runs 0.025 to 0.205 against about 1/d if the separation were
concentrated in a single direction, two to three orders of magnitude larger. The implied
single-layer AUROC for a random projection, Phi(d'/sqrt(2)), predicts 0.55 to 0.63 against 0.59 to
0.61 observed, and folding plus maximising over layers lifts that to the floors near 0.75 reported
elsewhere. The chain from "the classes differ" to "therefore arbitrary directions score" is
measured at each step rather than inferred.

    python scripts/surface_baseline.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import hashlib
import numpy as np

from external_paths import reanalysis  # noqa: E402
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from scipy.stats import norm
from sklearn.model_selection import (StratifiedGroupKFold, StratifiedKFold,
                                     cross_val_predict)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from extract_model import cfg, sad_items  # noqa: E402

ARDITI = reanalysis("arditi", "dataset", "splits")
CONTRASTIVE = reanalysis("manek", "datasets", "contrastive_dataset.json")
PUBLISHED_RANGE = (0.690, 0.792)
DIFFUSE_MODELS = ["qwen2.5-0.5b", "qwen2.5-1.5b", "qwen2.5-3b", "qwen2.5-7b"]


def fold(a):
    return max(a, 1 - a)


def refusal_surface(cv):
    """The other scoring set in this literature, scored the same way, so the parallel is measured."""
    txt, y = [], []
    for lab, stem in ((1, "harmful"), (0, "harmless")):
        for split in ("train", "val", "test"):
            f = ARDITI / f"{stem}_{split}.json"
            if not f.exists():
                continue
            for r in json.loads(f.read_text(encoding="utf-8")):
                txt.append(r["instruction"]); y.append(lab)
    y = np.array(y)
    X = CountVectorizer(min_df=2).fit_transform(txt)
    pr = cross_val_predict(LogisticRegression(max_iter=3000), X, y, cv=cv,
                           groups=np.array([int(hashlib.sha1(t.encode("utf-8")).hexdigest()[:12], 16) for t in txt]),
                           method="predict_proba")[:, 1]
    lens = np.array([len(t) for t in txt], float)
    return {"bow": float(fold(roc_auc_score(y, pr))),
            "length_only": float(fold(roc_auc_score(y, lens))), "n_items": len(txt)}


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    items = sad_items(cfg())
    txt = [x["body"] for x in items]
    y = np.array([1 if x["type"] == "testing" else 0 for x in items])
    # The scoring set repeats a handful of question bodies, so an item's exact text could
    # otherwise appear in both a training and a test fold and inflate the score. Grouping on the
    # body forbids that. Grouped CV is the PRIMARY figure; the ungrouped one is kept only to show
    # the difference is negligible here, which it is because the repetition is small.
    # A stable hash, not Python's: hash() is randomised per process unless PYTHONHASHSEED is
    # pinned, so grouping on it made the fold assignment -- and therefore every AUROC in this
    # table -- differ between runs. The permuted control moved 0.504 to 0.515 with no code change.
    groups = np.array([int(hashlib.sha1(t.encode("utf-8")).hexdigest()[:12], 16) for t in txt])
    n_unique = len(set(txt))
    cv = StratifiedGroupKFold(5, shuffle=True, random_state=0)
    cv_ungrouped = StratifiedKFold(5, shuffle=True, random_state=0)
    print(f"  {len(txt)} items, {n_unique} unique prompt bodies "
          f"({len(txt) - n_unique} repeated); folds are grouped on body.")

    print(f"{'=' * 92}\nIS THE LABEL RECOVERABLE FROM SURFACE TEXT? (in-domain, cross-validated)"
          f"\n{'=' * 92}")
    indomain, ungrouped, nfeat = {}, {}, {}
    for nm, vec in (("bag-of-words min_df=2", CountVectorizer(min_df=2)),
                    ("bag-of-words min_df=5", CountVectorizer(min_df=5)),
                    ("tf-idf", TfidfVectorizer(min_df=2)),
                    ("character 3-5 grams",
                     TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3))):
        X = vec.fit_transform(txt)
        pg = cross_val_predict(LogisticRegression(max_iter=3000), X, y, cv=cv, groups=groups,
                               method="predict_proba")[:, 1]
        pu = cross_val_predict(LogisticRegression(max_iter=3000), X, y, cv=cv_ungrouped,
                               method="predict_proba")[:, 1]
        indomain[nm] = float(fold(roc_auc_score(y, pg)))
        ungrouped[nm] = float(fold(roc_auc_score(y, pu)))
        nfeat[nm] = int(X.shape[1])
        print(f"  {nm:<24}AUROC {indomain[nm]:.4f} grouped   "
              f"({ungrouped[nm]:.4f} ungrouped, {X.shape[1]} features)")
    # Twenty permutations, not one. A single draw previously returned 0.387, nearly four standard
    # errors low, and would have appeared in a table captioned "control"; it is the minimum of 20.
    Xp = CountVectorizer(min_df=2).fit_transform(txt)
    rngp = np.random.default_rng(0)
    pv = []
    for i in range(20):
        yq = rngp.permutation(y)
        pq = cross_val_predict(LogisticRegression(max_iter=3000), Xp, yq,
                               cv=StratifiedGroupKFold(5, shuffle=True, random_state=i),
                               groups=groups, method="predict_proba")[:, 1]
        pv.append(float(roc_auc_score(yq, pq)))
    perm, perm_sd = float(np.mean(pv)), float(np.std(pv, ddof=1))
    print(f"  {'permuted labels (20x)':<24}AUROC {perm:.4f} +- {perm_sd:.4f}   "
          f"(min {min(pv):.3f}, max {max(pv):.3f})")

    lens = np.array([len(t) for t in txt], float)
    len_auc = float(fold(roc_auc_score(y, lens)))
    print(f"  {'prompt length alone':<24}AUROC {len_auc:.4f}")

    m = LogisticRegression(max_iter=3000).fit(Xp, y)
    names = np.array(CountVectorizer(min_df=2).fit(txt).get_feature_names_out())
    o = np.argsort(m.coef_[0])
    top_t = list(names[o[-8:]][::-1])
    top_d = list(names[o[:8]])
    print(f"\n  most discriminating tokens -- the confound is FORMAT, not content:")
    print(f"    toward 'testing':    {', '.join(top_t)}")
    print(f"    toward 'deployment': {', '.join(top_d)}")

    print("")
    print("=" * 92)
    print("WHY AN ARBITRARY DIRECTION SCORES: THE SEPARATION IS DIFFUSE")
    print("=" * 92)
    print("  Surface separability explains why SOME direction scores. It does not by itself")
    print("  explain why an ARBITRARY one does; that needs the format signal spread across many")
    print("  dimensions rather than concentrated. For random unit u, E[(delta.u)^2] = |delta|^2/d")
    print("  and E[sigma_u^2] = tr(Sigma)/d, so E[d'^2] ~ |delta|^2/tr(Sigma), and AUROC =")
    print("  Phi(d'/sqrt(2)). Were the separation concentrated in one direction the ratio would")
    print("  be about 1/d.")
    print("")
    print(f"  {'model':<14}{'|delta|^2/tr(S)':>17}{'1/d':>10}{'pred AUROC':>12}"
          f"{'observed':>10}")
    diff = {}
    for key in DIFFUSE_MODELS:
        f = ROOT / "results" / "gstudy" / key / "sad_filtered_lf.npz"
        if not f.exists():
            continue
        with np.load(f, allow_pickle=False) as z:
            sad = z["sad"].astype(np.float64); yy = z["sad_label"].astype(int)
        L = max(1, int(0.5 * (sad.shape[1] - 1)))
        P, N = sad[yy == 1, L, :], sad[yy == 0, L, :]
        delta = P.mean(0) - N.mean(0)
        trS = (((P - P.mean(0)) ** 2).sum() / (len(P) - 1)
               + ((N - N.mean(0)) ** 2).sum() / (len(N) - 1)) / 2
        ratio = float(delta @ delta) / float(trS)
        pred = float(norm.cdf(np.sqrt(ratio) / np.sqrt(2)))
        rng2 = np.random.default_rng(0)
        obs = float(np.mean([max(u, 1 - u) for u in
                    (roc_auc_score(yy, sad[:, L, :] @ (lambda w: w / np.linalg.norm(w))(
                        rng2.standard_normal(sad.shape[2]))) for _ in range(40))]))
        d_inv = 1.0 / sad.shape[2]
        diff[key] = {"ratio": ratio, "one_over_d": d_inv, "pred_auroc": pred,
                     "observed_auroc": obs, "layer": int(L)}
        print(f"  {key:<14}{ratio:17.3f}{d_inv:10.5f}{pred:12.3f}{obs:10.3f}")
    print("")
    rr = [v["ratio"] for v in diff.values()]
    oo = [v["observed_auroc"] for v in diff.values()]
    pp = [v["pred_auroc"] for v in diff.values()]
    print(f"  ORDER OF MAGNITUDE ONLY, not a quantitative model. Mean predicted "
          f"{np.mean(pp):.3f} against mean")
    print(f"  observed {np.mean(oo):.3f} is close, but the ORDERING does not track: the ratio spans "
          f"{min(rr):.3f} to {max(rr):.3f},")
    print(f"  a factor of {max(rr) / min(rr):.1f}, while the observed AUROC is flat at "
          f"{min(oo):.3f} to {max(oo):.3f}. Two likely")
    print(f"  reasons, neither of which the claim depends on: Phi(d'/sqrt(2)) assumes equal class")
    print(f"  covariances, which multiple-choice and free-form prompts do not have, and tr(Sigma) at")
    print(f"  one mid layer need not represent the layers the maximum actually selects.")
    print("")
    print("  The ratio exceeds 1/d by two to three orders of magnitude, so the format signal is")
    print("  diffuse and a random projection captures an appreciable share of it. Folding and")
    print("  maximising over layers then lifts a single-layer AUROC near 0.60 to the floors of")
    print("  about 0.75 reported elsewhere. No fair zero-shot TEXT comparison is available -- no")
    print("  text training set matched to this distinction exists -- so none is offered, and the")
    print("  benchmark claim rests on the in-domain figure alone.")
    print("")
    print(f"\n  WHAT THIS DOES SUPPORT is a claim about the benchmark. The label is recoverable")
    print(f"  from surface form at {max(indomain.values()):.3f} given label access, so a high score on")
    print(f"  this split is weak evidence for a latent representation of being evaluated: any")
    print(f"  direction correlated with prompt format or source will separate these classes, which")
    print(f"  is also why arbitrary directions on activations reach the floors we measure.")
    print("")
    # Computed, not quoted. These two numbers were hand-typed in an earlier version of this
    # script, which is exactly the failure mode the manuscript's macro discipline exists to stop.
    ref = refusal_surface(cv)
    print("  THE SAME HOLDS FOR THE OTHER SCORING SET IN THIS LITERATURE, and the parallel is the")
    print("  strongest form of the claim: the refusal split separates its classes by surface form")
    print(f"  too. Bag-of-words gives {ref['bow']:.3f} there against "
          f"{indomain['bag-of-words min_df=2']:.3f} here -- the same, NOT worse, and an")
    print(f"  earlier version of this script quoted {ref['bow']:.3f} as 0.967 from memory. Length "
          f"alone is the one place")
    print(f"  refusal is more confounded, {ref['length_only']:.3f} against {len_auc:.3f}. The claim "
          f"is that both splits are")
    print("  near-ceiling recoverable from surface form, not that either is worse.")

    out = {"pooling": "lf", "in_domain": indomain, "in_domain_ungrouped": ungrouped,
           "n_unique_bodies": n_unique, "cv": "StratifiedGroupKFold(5) grouped on prompt body",
           "in_domain_features": nfeat,
           "permuted_control": perm, "permuted_control_sd": perm_sd,
           "permuted_draws": pv, "refusal_split": ref, "length_only": len_auc,
           "diffuseness": diff, "published_range": list(PUBLISHED_RANGE),
           "top_tokens_testing": top_t, "top_tokens_deployment": top_d,
           "n_items": len(txt)}
    p = ROOT / "results" / "surface_baseline.json"
    p.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwrote {p}")


if __name__ == "__main__":
    main()
