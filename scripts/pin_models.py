"""Pin every model to a commit SHA and record its parameter count from weights.

Two channels this closes, both of which the paper's own reanalysis section documents as failures in
other people's artifacts.

REVISION. A Hugging Face repo can be updated in place. The Qwen3 thinking default, OLMo-2-32B's
missing bos_token and every parameter count verified here live in files that can change under us. A
template revision between now and camera-ready would silently alter every prompt; the prompt hash
would prove the prompts changed and say nothing about why. So each model records the commit SHA it
was verified at, and extraction passes revision= so a moved tag cannot become an undeclared
convention.

PARAMETERS. Every scaling regression in this project used a hardcoded dict of nominal names -- 0.5,
1.5, 3.0, 7.0 -- where the weights say 0.49, 1.54, 3.09, 7.62. The 7B entry is the leverage point of
a four-point fit, and its log10 was wrong by 0.037. Nothing published changed by more than 0.010 in
r, but the x-axis is now derived from the safetensors byte total and declared, so the next ladder
cannot inherit the problem.

    python scripts/pin_models.py            # report
    python scripts/pin_models.py --write    # update configs/gstudy_models.json and ladders.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONF = ROOT / "configs" / "gstudy_models.json"
LADDERS = ROOT / "configs" / "ladders.json"


def head_sha(repo):
    from huggingface_hub import HfApi
    return HfApi().model_info(repo).sha


def params_b(repo):
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
    return round(nbytes / per / 1e9, 4)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    conf = json.loads(CONF.read_text(encoding="utf-8"))
    changes = []
    print("existing ladder:")
    for key, spec in conf["models"].items():
        if not spec.get("enabled"):
            continue
        repo = spec["hf_id"]
        sha, pb = head_sha(repo), params_b(repo)
        old = spec.get("params_b")
        changes.append((key, old, pb, sha))
        print(f"  {key:<15}{old:>8} -> {pb:>8}   {sha[:12]}")
        if a.write:
            spec["params_b"] = pb
            spec["params_b_source"] = "safetensors total_size / dtype bytes"
            spec["revision"] = sha

    if LADDERS.exists():
        lad = json.loads(LADDERS.read_text(encoding="utf-8"))
        print("\nnew ladders:")
        for family, L in lad["ladders"].items():
            for m in L["models"]:
                sha = head_sha(m["repo"])
                print(f"  {m['repo']:<42}{m['params_b_from_weights']:>8}   {sha[:12]}")
                if a.write:
                    m["revision"] = sha
        if a.write:
            lad["revision_policy"] = ("every from_pretrained passes revision=; a repo updated in "
                                      "place cannot silently change a prompt or a parameter count")
            LADDERS.write_text(json.dumps(lad, indent=2), encoding="utf-8")

    if a.write:
        conf["_params_note"] = ("params_b is derived from safetensors byte totals, not from the "
                                "model name; nominal names understate several models and the 7B "
                                "entry was wrong by 0.037 in log10")
        CONF.write_text(json.dumps(conf, indent=2) + "\n", encoding="utf-8")
        print(f"\nwrote {CONF} and {LADDERS}")
    else:
        print("\ndry run; pass --write to apply")


if __name__ == "__main__":
    main()
