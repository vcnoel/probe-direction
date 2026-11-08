"""Re-extract the Qwen2.5 framing rows with both pooling conventions in one pass.

The Qwen2.5 row of the ladder table builds its directions from results/gstudy/<model>/pooled.npz,
whose framing array pools all tokens (scripts/gstudy_extract.py); the other ladders drop BOS and
whitespace-only tokens. This reruns the same framing prompts (data/framings_multi.json x
data/tasks_bench.jsonl, the same build_prompts call, the same eager-attention bfloat16 model load)
through scripts/extract_model.py's pool_both, which stores the validated and the all-token mean
from one forward pass, in float32 as the paper's Qwen2.5 arrays. The SAD scoring set
(sad_filtered_lf.npz, already BOS- and whitespace-dropped) is not touched, so the only change the
analysis sees is the pooling of the direction arrays. Registered in
docs/paraphrase-registration.md (companion analysis).

Writes $EAS_ACTIVATION_ROOT/results/post_submission/qwen25_validated/acts/<model>.npz. Atomic and
resumable.

    EAS_ACTIVATION_ROOT=<root> python scripts/extract_qwen25_validated.py --model qwen2.5-0.5b
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
from eval_awareness_spectral.prompts import build_prompts, load_framings, load_tasks  # noqa: E402
from extract_model import cfg, pool_both  # noqa: E402

ACT = Path(os.environ.get("EAS_ACTIVATION_ROOT", str(ROOT)))
OUT = ACT / "results" / "post_submission" / "qwen25_validated" / "acts"
MODELS = ("qwen2.5-0.5b", "qwen2.5-1.5b", "qwen2.5-3b", "qwen2.5-7b")


def run(key, delete_weights=False):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    spec = cfg()["models"][key]
    OUT.mkdir(parents=True, exist_ok=True)
    dest = OUT / f"{key}.npz"
    if dest.exists():
        print(f"{dest.name}: already extracted, skipping", flush=True)
        return None
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(spec["hf_id"], revision=spec["revision"])
    model = AutoModelForCausalLM.from_pretrained(
        spec["hf_id"], revision=spec["revision"], dtype=torch.bfloat16,
        attn_implementation="eager").to("cuda").eval()
    t_load = time.time() - t0
    multi = ROOT / "data" / "framings_multi.json"
    fr = load_framings(str(multi))
    names = [k for k in fr if not k.startswith("_")]
    tasks = load_tasks(str(ROOT / "data" / "tasks_bench.jsonl"))
    prompts = build_prompts(tasks, fr, tok, only_conditions=names, position_match=False)
    for p in prompts:
        assert p.task_text in p.text, f"task text not verbatim in {p.condition}/{p.task_id}"
    print(f"[{key}] {len(prompts)} framing prompts, loaded in {t_load:.0f}s", flush=True)
    t1 = time.time()
    val, allt, _ = pool_both(model, tok, [p.text for p in prompts], log_every=270)
    t_fwd = time.time() - t1
    fams = {k: v.get("family", "?") for k, v in json.loads(multi.read_text(encoding="utf-8")).items()
            if not k.startswith("_")}
    meta = {"key": key, "hf_id": spec["hf_id"], "revision": spec["revision"],
            "params_b": spec["params_b"], "dtype": "bfloat16", "storage_dtype": "float32",
            "attn": "eager", "pooling": "pool_both of scripts/extract_model.py",
            "seconds_load": t_load, "seconds_forward": t_fwd,
            "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
            "transformers": __import__("transformers").__version__}
    tmp = dest.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, framing_validated=val, framing_alltokens=allt,
                        framing_name=np.array([p.condition for p in prompts]),
                        framing_task=np.array([p.task_id for p in prompts]),
                        families=json.dumps(fams), meta=np.array(json.dumps(meta)))
    with np.load(tmp, allow_pickle=False) as z:
        assert z["framing_validated"].shape == val.shape
    tmp.replace(dest)
    print(f"[{key}] wrote {dest.name} {val.shape}, forward {t_fwd:.0f}s, total "
          f"{time.time() - t0:.0f}s", flush=True)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    if delete_weights:
        snap = Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface")) / "hub"
        for d in snap.glob(f"models--{spec['hf_id'].replace('/', '--')}*"):
            shutil.rmtree(d, ignore_errors=True)
    return {"key": key, "seconds_load": t_load, "seconds_forward": t_fwd,
            "seconds_total": time.time() - t0}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", action="append", choices=MODELS)
    ap.add_argument("--delete-weights", action="store_true")
    a = ap.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    if not a.model:
        ap.error("pass --model (repeatable)")
    for key in a.model:
        r = run(key, a.delete_weights)
        if r:
            print(json.dumps(r), flush=True)


if __name__ == "__main__":
    main()
