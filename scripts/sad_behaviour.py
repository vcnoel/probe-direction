"""Behavioural anchor runner: SAD stages_oversight on one model (docs/behavioural-anchor-registration.md).

Runs the SAD authors' task through their own code, with the local provider of
pod/sad_provider/hf_local_provider.py, then writes a sanitised result: per item only the SAD id, a
sha256 of the whitespace-normalised body, the label, the score and whether the answer parsed.
Nothing it writes or prints contains item or answer text. evalugator's raw results, which do,
stay inside the SAD clone (`--sad-dir`), which must lie outside every repository.

    python scripts/sad_behaviour.py setup --sad-dir <clone>
    python scripts/sad_behaviour.py run --sad-dir <clone> --model qwen3-0.6b [--n 10 --label smoke]

`setup` checks the clone's commit, copies the provider into its providers/ directory and adds one
models.yaml entry per model (type chat, assistant_answer_prefix true, as SAD configures the open
Llama chat models). `run` evaluates the registered variants (plain, sp_large) and, unless
--skip-same-prompt, the declared same-prompt read: the probe prompts themselves, next-token
log-odds of "A" against "B" after an assistant prefill "(", summarised by AUROC against the label.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
from result_io import write_result  # noqa: E402

SAD_COMMIT = "dfc5c9831a9bcc5c9a9dbdcaa2955aae983cd1d3"
EVALUGATOR_COMMIT = "1787ab88cf2e4cdf79d054087b2814cc55654ec2"
PROVIDER = ROOT / "pod" / "sad_provider" / "hf_local_provider.py"
OUT = Path(os.environ.get("EAS_ACTIVATION_ROOT", str(ROOT))) / "results" / "post_submission" / "behaviour"
VARIANTS = ("plain", "sp_large")
STRUCTS = Path("sad") / "stages" / "oversight" / "structs" / "batch"
# The task evaluates these (sad/stages/oversight/run.py, batch_names); the paper's original 400-item
# split of the Qwen2.5 arrays is this set. The Qwen3, Qwen3.5 and OLMo-2 probe arrays read
# oversight.yaml for their testing items instead (configs/gstudy_models.json): 78 of its 200 bodies
# are also in test_oversight.yaml (registration, Amendment 1).
TASK_FILES = (("test_oversight.yaml", "testing"), ("deploy_oversight.yaml", "deployment"))
PROBE_FILES = (("oversight.yaml", "testing"), ("deploy_oversight.yaml", "deployment"))


def probe_files_of(name: str):
    """The item files of a model's probe prompts: the paper's split for Qwen2.5, else the pod set."""
    return TASK_FILES if name.startswith("qwen2.5-") else PROBE_FILES


def body_hash(body: str) -> str:
    return hashlib.sha256(" ".join(body.replace("\r\n", "\n").split()).encode("utf-8")).hexdigest()


def git_head(path) -> str:
    try:
        return subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"], capture_output=True,
                              text=True, timeout=30).stdout.strip()
    except Exception as e:
        return f"unavailable: {e}"


def provider_models():
    import importlib.util
    spec = importlib.util.spec_from_file_location("hf_local_provider_cfg", PROVIDER)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["hf_local_provider_cfg"] = mod
    spec.loader.exec_module(mod)
    return mod.MODELS


def setup(sad_dir: Path) -> None:
    head = git_head(sad_dir)
    if head != SAD_COMMIT:
        raise SystemExit(f"SAD clone at {head}, registered {SAD_COMMIT}")
    run_py = (sad_dir / "sad" / "stages" / "oversight" / "run.py").read_text(encoding="utf-8")
    if 'batch_names=["test_oversight", "deploy_oversight"]' not in run_py:
        raise SystemExit("stages_oversight no longer evaluates test_oversight and deploy_oversight")
    if not all((sad_dir / STRUCTS / f).exists() for f, _ in TASK_FILES + PROBE_FILES):
        raise SystemExit("stages_oversight items are not decrypted; run the authors' "
                         "./unzip.sh --dirs sad/stages in the clone")
    if (ROOT.resolve() in sad_dir.resolve().parents) or sad_dir.resolve() == ROOT.resolve():
        raise SystemExit("the SAD clone must lie outside this repository")
    shutil.copy2(PROVIDER, sad_dir / "providers" / PROVIDER.name)
    import yaml
    cfg_path = sad_dir / "models.yaml"
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    have = {m["name"] for m in cfg}
    add = [{"name": n, "id": n, "type": "chat", "included": True, "assistant_answer_prefix": True}
           for n in provider_models() if n not in have]
    if add:
        with open(cfg_path, "a", encoding="utf-8") as f:
            f.write("\n" + yaml.safe_dump(add, sort_keys=False))
    print(f"SAD clone {head[:12]} ready: provider copied, {len(add)} models.yaml entries added")


def items(sad_dir: Path, files=PROBE_FILES):
    import yaml
    out = []
    for fname, label in files:
        y = yaml.safe_load((sad_dir / STRUCTS / fname).read_text(encoding="utf-8", errors="replace"))
        for s in y["samples"]:
            out.append({"id": s["id"], "body": s["body"], "label": label})
    return out


def run_sad(name: str, variant: str, n):
    """The authors' task, unchanged: SAD's stages_oversight with its templates and scoring."""
    from sad.stages.oversight.run import stages_oversight
    from sad.task import LARGE_SP_VARIANT, PLAIN_VARIANT
    v = {"plain": PLAIN_VARIANT, "sp_large": LARGE_SP_VARIANT}[variant]
    res = stages_oversight.run(model=name, variant=v, n=n, save=True)
    rows = []
    for sr in res.sample_results:
        s = sr.sample
        right = [c.lower() for c in getattr(s, "choices_right", [])]
        rows.append({"sad_id": s.id, "body_sha256": body_hash(s.body),
                     "label": "testing" if right[:1] == ["yes"] else "deployment",
                     "score": float(sr.score), "parsed": sr.output_parsed is not None})
    rows.sort(key=lambda r: r["sad_id"])
    n_items = len(rows)
    return {"n": n_items, "correct": int(sum(r["score"] for r in rows)),
            "invalid": int(sum(not r["parsed"] for r in rows)),
            "accuracy": float(np.mean([r["score"] for r in rows])) if rows else float("nan"),
            "per_item": rows}


def auroc(y, s):
    from scipy.stats import rankdata
    y, s = np.asarray(y), np.asarray(s, float)
    n1, n0 = int(y.sum()), int((1 - y).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = rankdata(s)
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def same_prompt(name: str, its, n):
    """Declared secondary: the probe prompts, log-odds of "A" against "B" after a prefill "("."""
    import evalugator as evalg
    import torch
    from render_prompts import render_sad_text
    prov = next(p for p in evalg.api.dispatcher.PROVIDERS if getattr(p, "MODELS", None)
                and p.provides_model(name))
    tok, model, device = prov.load(name)
    kw = {"enable_thinking": False} if name.startswith(("qwen3-", "qwen3.5-")) else {}
    ids = [tok.encode(c, add_special_tokens=False) for c in ("A", "B")]
    assert all(len(i) == 1 for i in ids), f"{name}: 'A' or 'B' is not a single token"
    a, b = ids[0][0], ids[1][0]
    rows = []
    for it in (its if n is None else its[:n]):
        text = render_sad_text(tok, it["body"], True, kw) + "("
        enc = tok(text, return_tensors="pt").to(device)
        with torch.inference_mode():
            logits = model(**enc, logits_to_keep=1).logits[0, -1].float()
        lp = torch.log_softmax(logits, -1)
        rows.append({"sad_id": it["id"], "body_sha256": body_hash(it["body"]), "label": it["label"],
                     "logodds_A_B": float(lp[a] - lp[b])})
    y = [r["label"] == "testing" for r in rows]
    au = auroc(y, [r["logodds_A_B"] for r in rows])
    return {"n": len(rows), "auroc": au, "auroc_folded": float(max(au, 1 - au)) if au == au else au,
            "per_item": rows}


def run(sad_dir: Path, name: str, n, label: str, out: Path, skip_same: bool, variants=VARIANTS) -> None:
    import torch
    import transformers
    t0 = time.time()
    os.chdir(sad_dir)                      # SAD reads models.yaml and providers/ from its root
    sys.path.insert(0, str(sad_dir))
    import evalugator  # noqa: F401
    res = {"model": name, "label": label, "variants": {}}
    for v in variants:
        t = time.time()
        res["variants"][v] = run_sad(name, v, n)
        r = res["variants"][v]
        print(f"[{name}] {v}: accuracy {r['accuracy']:.3f} over {r['n']} items, invalid "
              f"{r['invalid']} ({time.time() - t:.0f}s)", flush=True)
    if not skip_same:
        t = time.time()
        res["same_prompt"] = same_prompt(name, items(sad_dir, probe_files_of(name)), n)
        res["same_prompt"]["item_files"] = [f for f, _ in probe_files_of(name)]
        print(f"[{name}] same-prompt read: AUROC {res['same_prompt']['auroc']:.3f} over "
              f"{res['same_prompt']['n']} items ({time.time() - t:.0f}s)", flush=True)
    prov_models = provider_models()
    res["meta"] = {"repo": prov_models[name][0], "revision": prov_models[name][1],
                   "sad_commit": git_head(sad_dir), "evalugator_commit": EVALUGATOR_COMMIT,
                   "repo_commit": git_head(ROOT), "torch": torch.__version__,
                   "transformers": transformers.__version__,
                   "device": "cuda" if torch.cuda.is_available() and
                   os.environ.get("SAD_PROVIDER_DEVICE") != "cpu" else "cpu",
                   "n_limit": n, "seconds": time.time() - t0}
    dest = out / (label if label != "main" else "") / f"{name}.json"
    write_result(dest, res, kind="measured",
                 statistic="SAD stages_oversight accuracy (authors' templates, parsing and scoring), "
                           "plain and sp_large; same-prompt read: AUROC of log P(A) - log P(B) "
                           "after a prefill '(' on the probe prompts",
                 note=("SMOKE TEST: never used in the analysis" if label == "smoke" else
                       "behavioural anchor, docs/behavioural-anchor-registration.md"),
                 pooling="not applicable (generated answers and next-token log-odds)",
                 layers="not applicable", scoring_array="SAD stages_oversight, 400 items",
                 data_root="SAD clone at " + SAD_COMMIT[:12], model_population=[name])
    print(f"[{name}] wrote {dest.name} ({time.time() - t0:.0f}s)", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=("setup", "run"))
    ap.add_argument("--sad-dir", required=True, type=Path)
    ap.add_argument("--model")
    ap.add_argument("--n", type=int, default=None, help="smoke test only: first n items")
    ap.add_argument("--label", default="main", choices=("main", "smoke"))
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--skip-same-prompt", action="store_true")
    ap.add_argument("--variants", nargs="+", default=list(VARIANTS), choices=list(VARIANTS))
    a = ap.parse_args()
    if a.n is not None and a.label != "smoke":
        ap.error("--n is for the smoke test only (use --label smoke)")
    sad_dir = a.sad_dir.resolve()
    if a.stage == "setup":
        setup(sad_dir)
    else:
        if a.model not in provider_models():
            ap.error(f"unknown model {a.model}")
        if a.label == "main" and set(a.variants) != set(VARIANTS):
            ap.error("the registered run evaluates both variants")
        run(sad_dir, a.model, a.n, a.label, a.out.resolve(), a.skip_same_prompt, tuple(a.variants))


if __name__ == "__main__":
    main()
