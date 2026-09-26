"""Are the reported correlations computed on the right x-axis?

Every scaling statistic in this project regresses on log10(parameters), and every script takes those
parameters from a hardcoded dict of nominal names: 0.5, 1.5, 3.0, 7.0. Nominal names are marketing
rounding. Qwen3-0.6B holds 0.75B parameters by weight, and Qwen2.5-7B is not 7.0B either. Since 7B
is the leverage point of a four-point fit and carries the floor result, a wrong x is not obviously
harmless.

This recomputes every correlation and slope in the wrapper distribution under actual parameter counts
derived from the safetensors byte totals, and reports the movement. It changes no analysis by itself:
the outcome decides whether the parameter source is a conventions note or a corrections-table row.

    python scripts/param_count_check.py
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
NOMINAL = {"qwen2.5-0.5b": 0.5, "qwen2.5-1.5b": 1.5, "qwen2.5-3b": 3.0, "qwen2.5-7b": 7.0}
REPO = {"qwen2.5-0.5b": "Qwen/Qwen2.5-0.5B-Instruct", "qwen2.5-1.5b": "Qwen/Qwen2.5-1.5B-Instruct",
        "qwen2.5-3b": "Qwen/Qwen2.5-3B-Instruct", "qwen2.5-7b": "Qwen/Qwen2.5-7B-Instruct"}


def actual_params_b(repo):
    from huggingface_hub import HfApi, hf_hub_download
    from huggingface_hub.utils import EntryNotFoundError
    try:
        idx = json.loads(Path(hf_hub_download(repo, "model.safetensors.index.json"))
                         .read_text(encoding="utf-8"))
        nbytes = int(idx["metadata"]["total_size"])
    except EntryNotFoundError:
        info = HfApi().model_info(repo, files_metadata=True)
        nbytes = sum(s.size or 0 for s in info.siblings if s.rfilename.endswith(".safetensors"))
    cfg = json.loads(Path(hf_hub_download(repo, "config.json")).read_text(encoding="utf-8"))
    per = 4 if str(cfg.get("torch_dtype")) == "float32" else 2
    return nbytes / per / 1e9


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    keys = list(NOMINAL)
    actual = {k: actual_params_b(REPO[k]) for k in keys}

    print(f"{'model':<16}{'nominal':>9}{'actual':>9}{'log10 nom':>11}{'log10 act':>11}{'shift':>8}")
    for k in keys:
        ln, la = math.log10(NOMINAL[k]), math.log10(actual[k])
        print(f"  {k:<14}{NOMINAL[k]:>9.2f}{actual[k]:>9.2f}{ln:>11.3f}{la:>11.3f}{la - ln:>+8.3f}")

    xn = np.array([math.log10(NOMINAL[k]) for k in keys])
    xa = np.array([math.log10(actual[k]) for k in keys])

    wc = json.loads((ROOT / "results" / "wrapper_curves.json").read_text(encoding="utf-8"))
    M = np.array(wc["matrix"])                     # [4 models, 36 wrappers]
    if wc["models"] != keys:
        raise SystemExit(f"model order mismatch: {wc['models']} vs {keys}")

    rn = np.array([float(np.corrcoef(xn, M[:, j])[0, 1]) for j in range(M.shape[1])])
    ra = np.array([float(np.corrcoef(xa, M[:, j])[0, 1]) for j in range(M.shape[1])])
    sn = np.array([float(np.polyfit(xn, M[:, j], 1)[0]) for j in range(M.shape[1])])
    sa = np.array([float(np.polyfit(xa, M[:, j], 1)[0]) for j in range(M.shape[1])])

    dr, ds = ra - rn, sa - sn
    flips = int(((rn > 0) != (ra > 0)).sum())
    print("")
    print(f"  {M.shape[1]} wrappers, correlation with log10 parameters:")
    print(f"    nominal range {rn.min():+.3f} to {rn.max():+.3f}; "
          f"actual range {ra.min():+.3f} to {ra.max():+.3f}")
    print(f"    max |change| in r     {np.abs(dr).max():.4f}   median {np.median(np.abs(dr)):.4f}")
    print(f"    max |change| in slope {np.abs(ds).max():.4f}")
    print(f"    sign flips            {flips} of {M.shape[1]}")
    print(f"    positive count        {int((rn > 0).sum())} nominal -> {int((ra > 0).sum())} actual")
    print(f"    median r              {np.median(rn):+.4f} -> {np.median(ra):+.4f}")

    out = {"nominal": NOMINAL, "actual_params_b": actual,
           "actual_source": "safetensors total_size / dtype bytes",
           "n_wrappers": int(M.shape[1]),
           "r_range_nominal": [float(rn.min()), float(rn.max())],
           "r_range_actual": [float(ra.min()), float(ra.max())],
           "max_abs_dr": float(np.abs(dr).max()),
           "median_abs_dr": float(np.median(np.abs(dr))),
           "max_abs_dslope": float(np.abs(ds).max()),
           "sign_flips": flips,
           "n_positive_nominal": int((rn > 0).sum()),
           "n_positive_actual": int((ra > 0).sum()),
           "median_r_nominal": float(np.median(rn)),
           "median_r_actual": float(np.median(ra))}

    material = flips > 0 or np.abs(dr).max() > 0.02
    out["material"] = bool(material)
    print("")
    if material:
        print("  MATERIAL. The x-axis was wrong by enough to change a reported quantity, so the")
        print("  parameter source becomes a corrections-table row and every script must read counts")
        print("  from weights rather than from a nominal dict.")
    else:
        print("  NOT MATERIAL at the precision the paper reports. No published quantity changes, so")
        print("  this is a conventions note -- the parameter source is now declared -- and not a")
        print("  corrections row. The dicts are still replaced so the next ladder cannot inherit the")
        print("  problem.")
    p = ROOT / "results" / "param_count_check.json"
    p.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwrote {p}")


if __name__ == "__main__":
    main()
