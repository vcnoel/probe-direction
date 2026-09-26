"""The statistic for every model and wrapper on every ladder, for the figures that show the ladders together.

scripts/pod_ladders.py stores what each ladder's 36 correlations add up to (range, counts, spans) but
not the correlations themselves, and a figure that draws the distribution needs them. This computes
the same full-set statistic with pod_ladders' own loaders and scorer, imported rather than copied,
so a ladder added to pod_ladders.LADDERS appears here with no edit. Ladders whose arrays are not on
disk are skipped and named.

    python scripts/ladder_curves.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from pod_ladders import LADDERS, load, load_qwen25, peak  # noqa: E402
from result_io import write_result  # noqa: E402


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    out, skipped = {}, []
    for name, (tmpl, params) in LADDERS.items():
        models = list(params)
        gstudy = tmpl == "GSTUDY"
        paths = models if gstudy else [tmpl.format(m) for m in models]
        if not gstudy and any(not (ROOT / p).exists() for p in paths):
            skipped.append(name)
            print(f"  SKIP {name}: arrays not on disk")
            continue
        lp = np.log10([params[m] for m in models])
        rows, pairs = [], None
        for p in paths:
            proj, y, pairs, _, _ = load_qwen25(p) if gstudy else load(p)
            rows.append([peak(y, proj[j]) for j in range(len(pairs))])
        x = np.array(rows)
        r = [float(np.corrcoef(lp, x[:, j])[0, 1]) for j in range(x.shape[1])]
        out[name] = {"models": models, "params_b": [params[m] for m in models],
                     "pairs": [list(pr) for pr in pairs], "statistic": x.tolist(),
                     "r": r}
        print(f"  {name}: {len(models)} models, r {min(r):+.3f} to {max(r):+.3f}", flush=True)
    pop = sorted({m for v in out.values() for m in v["models"]})
    write_result(ROOT / "results" / "ladder_curves.json",
                 {"ladders": out, "skipped": skipped}, kind="measured",
                 pooling="validated for scoring and for the direction arrays "
                         "(Qwen2.5: lf scoring, all-token directions, as pod_ladders)",
                 layers="ALL", scoring_array="results/pod_*/*.npz and results/gstudy",
                 data_root="pod", statistic="folded max over layers, full scoring set",
                 model_population=pop)
    print("wrote results/ladder_curves.json")


if __name__ == "__main__":
    main()
