"""Numerical floor of the reproduction gate (docs/paraphrase-registration.md, amendment 2).

The paraphrase extraction renders 36 framings per task, so its padded batches differ from those of
the paper's pod arrays (18 framings per task), and it runs on a different GPU and software stack.
bfloat16 arithmetic makes pooled activations depend on both. To separate the two, this re-extracts
the paper's own framing prompt set (all 18 framings x 60 tasks, the order and batch of the pod run)
on the analysis GPU and compares the paper's statistic of the 36 original wrappers across three
extractions of the same 720 original-framing prompts:

  S  the stored pod arrays (results/pod_qwen3[5]/<model>__nothink.npz)
  F  this extraction: S's prompt order and batch, the analysis machine
  N  the paraphrase extraction's version-0 rows: the analysis machine, a different batch order

  floor_batch    N against F: same machine and software, batch composition only
  floor_machine  F against S: same batch composition, machine and software only
  gate           N against S: what the registered gate measured

    EAS_ACTIVATION_ROOT=<root> python scripts/paraphrase_noise_floor.py extract --model qwen3-1.7b
    EAS_ACTIVATION_ROOT=<root> python scripts/paraphrase_noise_floor.py summary
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
import extract_paraphrase as xp  # noqa: E402
from eval_awareness_spectral.prompts import build_prompts, load_framings, load_tasks  # noqa: E402
from gstudy_gtheory import auc_cols  # noqa: E402
from pod_ladders import LADDERS, peak, unit  # noqa: E402
from result_io import write_result  # noqa: E402

ACT = xp.ACT
FLOOR = ACT / "results" / "post_submission" / "paraphrase" / "floor"
OUT = ROOT / "results" / "post_submission" / "paraphrase" / "noise_floor.json"
LADDER_OF = {m: lad for lad in ("qwen3_nothink", "qwen3.5_nothink") for m in LADDERS[lad][1]}
POD_VRAM_GB = 85.1      # what the paper's pod run read from its 80 GB card ("batch 16 from 85GB VRAM")


def pod_batch(key):
    """The batch the paper's pod run used for this model: pod/extract.py's rule at the pod's VRAM
    (16 for every rung up to 27B, 10 for Qwen3-32B)."""
    return xp.pod_extract.batch_for(xp.MODELS[key][2] / 1e9, POD_VRAM_GB)


def extract(key):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    repo, rev, _, want = xp.MODELS[key]
    FLOOR.mkdir(parents=True, exist_ok=True)
    dest = FLOOR / f"{key}__nothink.npz"
    if dest.exists():
        print(f"{dest.name}: already extracted")
        return
    tok = AutoTokenizer.from_pretrained(repo, revision=rev)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    th = hashlib.sha256((tok.chat_template or "").encode()).hexdigest()[:12]
    assert th == want, f"chat template {th} != {want}"
    fr = load_framings(str(ROOT / "data" / "framings_multi.json"))
    names = [k for k in fr if not k.startswith("_")]           # scripts/render_prompts.py
    tasks = load_tasks(str(ROOT / "data" / "tasks_bench.jsonl"))
    prompts = build_prompts(tasks, fr, tok, only_conditions=names, position_match=False,
                            template_kwargs=dict(xp.TEMPLATE_KWARGS, date_string=xp.PINNED_DATE))
    texts = [p.text.replace("\r\n", "\n") for p in prompts]
    model = AutoModelForCausalLM.from_pretrained(repo, revision=rev, dtype=torch.bfloat16,
                                                 attn_implementation="sdpa", device_map="cuda").eval()
    t0 = time.time()
    batch = pod_batch(key)
    out = xp.pod_extract.pool_reductions(model, tok, texts, batch)
    keep = np.array([p.condition.startswith(("eval_", "deploy_")) for p in prompts])
    rows = [{"name": p.condition, "task": p.task_id, "text": t}
            for p, t, k in zip(prompts, texts, keep) if k]          # the 720 stored rows
    rows_sha = hashlib.sha256(json.dumps(rows, sort_keys=True, ensure_ascii=False)
                              .encode("utf-8")).hexdigest()
    tmp = dest.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, framing_validated=out["validated"][keep],
                        framing_name=np.array([p.condition for p in prompts])[keep],
                        framing_task=np.array([p.task_id for p in prompts])[keep],
                        meta=np.array(json.dumps({"repo": repo, "revision": rev, "batch": batch,
                                                  "n_prompts_run": len(texts),
                                                  "seconds_forward": time.time() - t0,
                                                  "gpu": torch.cuda.get_device_name(0),
                                                  "torch": torch.__version__,
                                                  "torch_cuda": torch.version.cuda,
                                                  "transformers": __import__("transformers").__version__,
                                                  "framing_rows_sha256": rows_sha,
                                                  "n_framing_rows": len(rows),
                                                  **xp.git_state()})))
    tmp.replace(dest)
    print(f"{dest.name}: {len(texts)} prompts in {time.time() - t0:.0f}s")
    del model
    torch.cuda.empty_cache()


def originals(path, frs):
    with np.load(path, allow_pickle=False) as z:
        n = [str(s) for s in z["framing_name"]]
        t = [str(s) for s in z["framing_task"]]
        tasks = sorted({x for x, f in zip(t, n) if f in frs})
        pos = {(a, b): i for i, (a, b) in enumerate(zip(n, t))}
        idx = np.array([[pos[(f, x)] for x in tasks] for f in frs])
        return z["framing_validated"][idx].astype(np.float32).mean(1)    # [12, L, d]


def compare(model):
    lad = LADDER_OF[model]
    tmpl = LADDERS[lad][0]
    with np.load(ACT / tmpl.format(model), allow_pickle=False) as z:
        sad = z["sad_validated"].astype(np.float32)
        y = z["sad_label"].astype(int)
        names = sorted({str(s) for s in z["framing_name"]})
    ev = [n for n in names if n.startswith("eval_")]
    dp = [n for n in names if n.startswith("deploy_")]
    frs = ev + dp
    mu = {"S": originals(ACT / tmpl.format(model), frs),
          "F": originals(FLOOR / f"{model}__nothink.npz", frs),
          "N": originals(xp.OUT / f"{model}__nothink.npz", frs)}
    stat, dirs = {k: [] for k in mu}, {k: [] for k in mu}
    for e, d in itertools.product(range(len(ev)), range(len(ev), len(frs))):
        for k, M in mu.items():
            D = unit(M[e] - M[d])
            dirs[k].append(D)
            stat[k].append(peak(y, np.einsum("ild,ld->il", sad, D)))
    stat = {k: np.array(v) for k, v in stat.items()}
    dirs = {k: np.array(v) for k, v in dirs.items()}

    def pair(a, b):
        cos = (dirs[a] * dirs[b]).sum(-1)
        dd = np.abs(stat[a] - stat[b])
        return {"max_abs_diff_fold": float(dd.max()), "mean_abs_diff_fold": float(dd.mean()),
                "min_direction_cosine": float(cos.min()),
                "spearman_fold": float(spearmanr(stat[a], stat[b])[0])}
    with np.load(FLOOR / f"{model}__nothink.npz", allow_pickle=False) as z:
        floor_meta = json.loads(str(z["meta"]))   # rungs extracted before a field existed lack it
    return {"ladder": lad, "floor_meta": floor_meta,
            "floor_batch": pair("N", "F"), "floor_machine": pair("F", "S"),
            "gate": pair("N", "S")}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=("extract", "summary"))
    ap.add_argument("--model", action="append", choices=sorted(LADDER_OF))
    a = ap.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    if a.stage == "extract":
        for m in a.model or []:
            extract(m)
        return
    models = [m for m in LADDER_OF if (FLOOR / f"{m}__nothink.npz").exists()
              and (xp.OUT / f"{m}__nothink.npz").exists()]
    res = {}
    for m in models:
        res[m] = compare(m)
        r = res[m]
        print(f"{m:14s} gate N-S max {r['gate']['max_abs_diff_fold']:.4f} mean {r['gate']['mean_abs_diff_fold']:.4f}"
              f" | batch N-F max {r['floor_batch']['max_abs_diff_fold']:.4f} mean {r['floor_batch']['mean_abs_diff_fold']:.4f}"
              f" | machine F-S max {r['floor_machine']['max_abs_diff_fold']:.4f} mean {r['floor_machine']['mean_abs_diff_fold']:.4f}"
              f" | min cos N-S {r['gate']['min_direction_cosine']:.5f}", flush=True)
    write_result(OUT, {"models": res, "registration": "docs/paraphrase-registration.md, amendment 2"},
                 kind="measured",
                 statistic="paper's fold statistic of the 36 original wrappers (full-task directions) "
                           "from three extractions of the same original-framing prompts",
                 note="numerical floor for the reproduction gate; post-submission",
                 pooling="validated (BOS and whitespace-only tokens dropped)",
                 layers="ALL",
                 scoring_array="sad_validated of results/pod_qwen3/*__nothink.npz and "
                               "results/pod_qwen35/*__nothink.npz",
                 data_root="pod + post_submission/paraphrase", model_population=sorted(models))
    print(f"wrote {OUT.as_posix()}")


if __name__ == "__main__":
    main()
