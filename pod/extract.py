"""GPU-side extraction. Forward passes and nothing else.

Design constraints, all of them cost-driven or correctness-driven:

  * The GPU host never renders a prompt. It reads data/rendered/*.json, asserts the sha256 recorded in
    the manifest, and tokenises those exact strings. A chat template is code and some interpolate
    the date; a prompt rendered here would depend on when here was.
  * The GPU host never analyses. Analysis is CPU work and runs anywhere.
  * Both pooling conventions come from ONE forward pass. Running the model twice to get two
    reductions would double the only thing that costs GPU time.
  * Sufficient statistics only: per (prompt, layer) pooled vectors in fp16, never per-token
    tensors. This is the 5.5GB -> 189MB reduction the existing pipeline already relies on.
  * Every model is pinned by revision. A repo updated in place cannot silently change a weight.
  * Upload and verify after each model, then delete the snapshot. Remote hosts die and disks are
    smaller than the ladder.

    python pod/extract.py --dry-run              resolve everything, touch no GPU
    python pod/extract.py --ladder qwen3         run it
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

ROOT = Path(__file__).resolve().parents[1]
RENDERED = ROOT / "data" / "rendered"
# A rehearsal must not be able to write where the real run's outputs live, so the
# destination is overridable and the rehearsal sets it.
OUT = Path(os.environ.get("POD_OUT") or ROOT / "results" / "pod")
SPECIALS = {"<|begin_of_text|>", "<s>", "<BOS>", "<bos>", "<|endoftext|>"}


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def manifest():
    m = json.loads((RENDERED / "manifest.json").read_text(encoding="utf-8"))
    for g in m["groups"]:
        f = RENDERED / g["file"]
        if not f.exists():
            raise SystemExit(f"missing rendered prompts: {f}")
        payload = json.loads(f.read_text(encoding="utf-8"))
        recorded = payload.pop("sha256")
        actual = hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False)
                                .encode("utf-8")).hexdigest()
        if recorded != g["sha256"] or actual != recorded:
            raise SystemExit(f"{g['file']}: prompt hash mismatch; re-render locally, do not "
                             f"proceed -- the prompts on this pod are not the prompts analysed")
    return m


def groups_for(repo, m):
    """Every rendered prompt set for this repo, one per declared variant.

    Returning all of them rather than the first is load-bearing: a family with two renderings
    would otherwise have one silently dropped, and which one depends on dict order.
    """
    got = [(g.get("variant", "default"), g) for g in m["groups"] if repo in g["repos"]]
    if not got:
        raise SystemExit(f"{repo} is in no rendered group; render it locally first")
    seen = [v for v, _ in got]
    if len(set(seen)) != len(seen):
        raise SystemExit(f"{repo}: duplicate variant names {seen}; the manifest is inconsistent")
    return sorted(got)


def out_key(key, variant, n_variants):
    """One output file per (model, variant).

    A model with a single rendering keeps its bare name. A model with several names every one of
    them, including the primary: letting `think` claim `qwen3-8b.npz` would make the rendering an
    undeclared property of the filename, which is the failure this project exists to document.
    """
    return key if n_variants == 1 else f"{key}__{variant}"


def free_gb(path):
    """Free space in GB, honouring POD_DISK_GB when the filesystem cannot report a quota.

    statvfs is not the truth on a network filesystem. A quota-enforced network volume can report the
    free space of the whole cluster (petabytes) while the quota of the volume you were allocated is
    enforced server-side and is invisible here. This guard therefore passed with '674322GB free', and the
    Qwen3-32B download then died 30GB in with EDQUOT. A guard that cannot see the limit it checks
    is worse than no guard, because it is quoted in the log as if it had checked something.

    POD_DISK_GB declares the volume's quota. Set it and this becomes a real check; omit it on a normal
    disk and shutil is believed as before.
    """
    free = shutil.disk_usage(path).free / 1e9
    declared = os.environ.get("POD_DISK_GB")
    if not declared:
        return free
    # The quota covers the whole volume, not our subdirectory, so usage is measured from the mount
    # point -- walking up while the device number holds -- or the venv and the model cache are spent
    # without ever being counted. POD_DISK_ROOT names it outright when that walk is not wanted:
    # unbounded, this climbs to the filesystem root and tries to size the entire machine, which is
    # what it did on the first attempt here.
    root = os.environ.get("POD_DISK_ROOT")
    if root:
        p = Path(root)
    else:
        p = Path(path).resolve()
        dev = p.stat().st_dev
        while p.parent != p and p.parent != p.anchor and p.parent.stat().st_dev == dev:
            p = p.parent
    if p == Path(p.anchor):          # a filesystem root means no quota-bound volume was found
        return free
    used = sum(f.stat().st_size for f in p.rglob("*")
               if f.is_file() and not f.is_symlink()) / 1e9
    return min(free, float(declared) - used)


def vram_total(default=80.0):
    # The environment variable wins over the detected device: a dry run on a small GPU must be able to
    # plan for the large one, and detecting 17GB here would report that the ladder does not fit.
    if os.environ.get("POD_VRAM_GB"):
        return float(os.environ["POD_VRAM_GB"])
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.get_device_properties(0).total_memory / 1e9
    except Exception:
        pass
    return default


def pool_reductions(model, tok, texts, batch, log_every=200):
    """-> dict of three [n, L+1, d] fp16 arrays from ONE pass per batch.

    Deliberately avoids torch.stack over hidden_states. output_hidden_states=True already
    materialises every layer in the returned tuple, so the tuple itself is unavoidable; what stacking
    would add is a SECOND full [L+1, B, T, d] copy, several GB for Qwen3-32B at 65 states and 5120
    wide, on top of a 66GB weight footprint on an 80GB card. Reducing each layer in place as it is
    visited removes that second copy and nothing else -- the saving is a factor of two on activations,
    not a factor of L.
    """
    import numpy as np
    import torch
    n = len(texts)
    out = {}
    with torch.inference_mode():
        for i in range(0, n, batch):
            chunk = texts[i:i + batch]
            enc = tok(chunk, return_tensors="pt", padding=True).to(model.device)
            # logits_to_keep=1: the hidden states are all this reads, and full logits are a
            # [B, T, vocab] tensor -- 6.6GB at Qwen3.5-27B's batch and 248k vocabulary, which is
            # the difference between fitting 80GB and not. The residual stream is computed before
            # the head and is unchanged by it.
            hs = model(**enc, output_hidden_states=True,
                       logits_to_keep=1).hidden_states                   # tuple[L+1] [B,T,d]
            attn = enc["attention_mask"].bool()
            L1, d = len(hs), hs[0].shape[-1]
            if not out:
                for k in ("validated", "alltokens", "lasttoken"):
                    out[k] = np.empty((n, L1, d), dtype=np.float16)
            keeps = []
            for b in range(len(chunk)):
                ids = enc["input_ids"][b][attn[b]].tolist()
                k = torch.tensor(
                    [tok.decode([t]).strip() != "" and tok.decode([t]).strip() not in SPECIALS
                     for t in ids], device=hs[0].device)
                keeps.append(k if k.any() else torch.ones_like(k))
            for li, h in enumerate(hs):                     # one layer resident at a time
                for b in range(len(chunk)):
                    hb = h[b][attn[b]]                      # [T_b, d]
                    kb = keeps[b]
                    out["validated"][i + b, li] = hb[kb].mean(0).to(torch.float16).cpu().numpy()
                    out["alltokens"][i + b, li] = hb.mean(0).to(torch.float16).cpu().numpy()
                    out["lasttoken"][i + b, li] = hb[-1].to(torch.float16).cpu().numpy()
            del hs
            if (i + batch) % log_every < batch:
                log(f"    {min(i + batch, n)}/{n}")
    return out


def resident_gb(spec):
    """Weights that end up on the GPU, which is not always what is downloaded.

    A VLM checkpoint (Qwen3.5) ships a vision tower and an MTP head that AutoModelForCausalLM never
    loads, so sizing the batch from the download would under-fill the card and, on a 16GB card,
    refuse a model that fits. Disk is still planned from weight_bytes, the full download.
    """
    return spec.get("weight_bytes_text", spec["weight_bytes"]) / 1e9


def batch_for(weight_gb, total_vram_gb):
    """Batch size from the headroom left after weights. Large models get small batches; the pod is
    paid for by the hour and an OOM after a 66GB download is the expensive failure."""
    head = total_vram_gb - weight_gb * 1.05
    if head < 3:
        raise SystemExit(f"only {head:.1f}GB VRAM headroom after weights; this model does not fit")
    return max(1, min(16, int(head // 1.5)))


def run_model(key, repo, revision, groups, dry, weight_gb=0.0, vram_gb=80.0, limit=0):
    """Extract every rendered variant of one model under a single load of its weights.

    A family can declare more than one rendering of the same prompt -- Qwen3's thinking switch is
    one -- and each is a separate prompt set with its own hash and its own output file. They share
    the download and the load, which is the whole reason to loop here rather than per (model,
    variant) in main: the weights are the expensive part, the forward passes are not.
    """
    import numpy as np
    nv = len(groups)
    # A truncated run is for rehearsing the machinery, so it must be unable to pose as a real
    # extraction: the prompt count goes in the filename and in the payload's meta.
    if limit:
        key = f"{key}_LIMIT{limit}"
    todo = [(v, g) for v, g in groups if not (OUT / f"{out_key(key, v, nv)}.npz").exists()]
    for v, g in groups:
        if (v, g) not in todo:
            log(f"{out_key(key, v, nv)}: already extracted and verified, skipping")
    if not todo:
        return None
    sets = []
    for variant, g in todo:
        payload = json.loads((RENDERED / g["file"]).read_text(encoding="utf-8"))
        s_txt = [r["text"] for r in payload["sad"]]
        f_txt = [r["text"] for r in payload["framing"]]
        if limit:
            payload = {**payload, "sad": payload["sad"][:limit],
                       "framing": payload["framing"][:limit]}
            s_txt, f_txt = s_txt[:limit], f_txt[:limit]
        sets.append((variant, g, payload, s_txt, f_txt))
    if dry:
        return {"key": key, "n_prompts": sum(len(s) + len(f) for _, _, _, s, f in sets),
                "variants": [v for v, _, _, _, _ in sets],
                "group": ", ".join(g["file"] for _, g, _, _, _ in sets)}

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(repo, revision=revision)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    model = AutoModelForCausalLM.from_pretrained(
        repo, revision=revision, dtype=torch.bfloat16,
        attn_implementation="sdpa", device_map="cuda").eval()
    log(f"{key}: loaded in {time.time() - t0:.0f}s, "
        f"{torch.cuda.memory_allocated() / 1e9:.1f}GB allocated")

    bs = batch_for(weight_gb, vram_gb)
    log(f"{key}: batch {bs} from {vram_gb:.0f}GB VRAM and {weight_gb:.0f}GB weights")
    written = []
    for variant, g, payload, sad, fr in sets:
        ok, tv = out_key(key, variant, nv), time.time()
        log(f"{ok}: {len(sad) + len(fr)} prompts from {g['file']}")
        s = pool_reductions(model, tok, sad, bs)
        f = pool_reductions(model, tok, fr, bs)
        arrays = {
            "sad_validated": s["validated"], "sad_alltokens": s["alltokens"],
            "sad_lasttoken": s["lasttoken"],
            "framing_validated": f["validated"], "framing_alltokens": f["alltokens"],
            "framing_lasttoken": f["lasttoken"],
            "sad_label": np.array([1 if r["label"] == "testing" else 0 for r in payload["sad"]]),
            "framing_name": np.array([r["name"] for r in payload["framing"]]),
            "framing_task": np.array([r["task"] for r in payload["framing"]]),
            "meta": np.array(json.dumps({
                "key": ok, "model_key": key, "repo": repo, "revision": revision,
                "variant": variant, "template_kwargs": g.get("template_kwargs", {}),
                "prompt_file": g["file"], "prompt_sha256": g["sha256"],
                "template_sha12": g["template_sha12"], "bos_in_rendered": g["bos_in_rendered"],
                "dtype": "bfloat16", "storage_dtype": "float16",
                "truncated_to": limit or None,
                "pooling_filter": "BOS and whitespace-only tokens dropped for 'validated'",
            })),
        }
        OUT.mkdir(parents=True, exist_ok=True)
        dest = OUT / f"{ok}.npz"
        tmp = dest.with_suffix(".tmp.npz")
        np.savez_compressed(tmp, **arrays)
        with np.load(tmp, allow_pickle=False) as z:          # verify before rename
            assert z["sad_validated"].shape[0] == len(sad)
            assert z["framing_validated"].shape[0] == len(fr)
        tmp.replace(dest)
        log(f"{ok}: wrote {dest.name} ({dest.stat().st_size / 1e6:.0f}MB) in "
            f"{time.time() - tv:.0f}s")
        written.append({"key": ok, "seconds": time.time() - tv,
                        "mb": dest.stat().st_size / 1e6})
    del model
    torch.cuda.empty_cache()
    return {"key": key, "seconds": time.time() - t0, "written": written,
            "mb": sum(w["mb"] for w in written)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ladder", default="qwen3")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--upload", default=os.environ.get("POD_UPLOAD", ""),
                    help="rclone/scp target; each model is pushed and verified before deletion")
    ap.add_argument("--poweroff", action="store_true")
    ap.add_argument("--limit", type=int, default=0,
                    help="rehearsal only: first N prompts of each set. Output files and their meta "
                         "are marked LIMIT so a truncated run cannot be mistaken for a real one.")
    ap.add_argument("--only", default="", help="rehearsal only: substring filter on repo")
    ap.add_argument("--fit-only", action="store_true",
                    help="skip, with a log line, every model whose weights do not fit this GPU, "
                         "instead of stopping at the first one. For a local card that holds the "
                         "small rungs of a ladder whose large rungs go to the pod.")
    ap.add_argument("--keep-weights", action="store_true",
                    help="rehearsal only: do not delete the snapshot after extraction. On the pod "
                         "the disk is smaller than the ladder, so deletion is the default.")
    a = ap.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    lad = json.loads((ROOT / "configs" / "ladders.json").read_text(encoding="utf-8"))
    if a.ladder not in lad["ladders"]:
        raise SystemExit(f"unknown ladder {a.ladder}; have {sorted(lad['ladders'])}")
    models = lad["ladders"][a.ladder]["models"]
    models = sorted(models, key=lambda m: m["params_b_from_weights"])   # ascending: fail cheap
    if a.only:
        models = [s for s in models if a.only.lower() in s["repo"].lower()]
        if not models:
            raise SystemExit(f"--only {a.only!r} matched no model in ladder {a.ladder}")
    m = manifest()

    log(f"ladder {a.ladder}: {len(models)} models, ascending, "
        f"{lad['ladders'][a.ladder]['total_weight_gb']}GB of weights")
    log(f"prompt hashes verified for {len(m['groups'])} template groups")
    log(f"free disk at {ROOT}: {free_gb(ROOT):.0f}GB")

    if a.dry_run:
        print()
        for spec in models:
            gs = groups_for(spec["repo"], m)
            r = run_model(spec["repo"].split("/")[-1].lower(), spec["repo"],
                          spec.get("revision"), gs, dry=True)
            if r is None:                          # every variant already extracted and verified
                print(f"  {spec['repo']:<42} already extracted")
                continue
            need = spec["weight_bytes"] / 1e9
            vram = resident_gb(spec) * 1.15
            try:
                bs = batch_for(resident_gb(spec), vram_total())
                bstr = f"batch {bs:>2}"
            except SystemExit as e:
                bstr = "DOES NOT FIT"
            print(f"  {spec['repo']:<42} rev={str(spec.get('revision'))[:12]} "
                  f"{r['n_prompts']} prompts{' ' + '+'.join(r['variants']) if len(r['variants']) > 1 else ''}"
                  f"  disk~{need:.0f}GB  weights~{vram:.0f}GB  {bstr}")
        biggest = max(s["weight_bytes"] for s in models) / 1e9
        resident = max(resident_gb(s) for s in models)
        print(f"\n  peak single-model disk {biggest:.0f}GB, weights ~{resident * 1.15:.0f}GB "
              f"against {vram_total():.0f}GB VRAM")
        print(f"  free disk {free_gb(ROOT):.0f}GB -- policy is delete-after-upload, so only one "
              f"model coexists")
        if not a.upload:
            print("  WARNING no --upload target: results would exist only on the pod")
        print("\n  dry run complete; no GPU touched")
        return

    t_start = time.time()
    for i, spec in enumerate(models, 1):
        key = spec["repo"].split("/")[-1].lower()
        gs = groups_for(spec["repo"], m)
        if a.fit_only:
            try:
                batch_for(resident_gb(spec), vram_total())
            except SystemExit:
                log(f"{key}: {resident_gb(spec):.1f}GB of weights do not fit "
                    f"{vram_total():.0f}GB VRAM; skipped (--fit-only), run it on the pod")
                continue
        need = spec["weight_bytes"] / 1e9 * 1.3
        if free_gb(ROOT) < need:
            raise SystemExit(f"only {free_gb(ROOT):.0f}GB free, need ~{need:.0f}GB for {key}; "
                             f"refusing to start a download that cannot finish")
        try:
            run_model(key, spec["repo"], spec.get("revision"), gs, dry=False,
                      weight_gb=resident_gb(spec), vram_gb=vram_total(), limit=a.limit)
        except (OSError, RuntimeError) as e:
            # EDQUOT(122)/ENOSPC(28) arrive as a bare traceback from deep inside the downloader, and
            # on a network filesystem the guard above cannot predict them: see free_gb. Fail with
            # the one fact that makes the next run work instead of with a stack trace.
            if "quota" not in str(e).lower() and getattr(e, "errno", None) not in (28, 122):
                raise
            raise SystemExit(
                f"\n{key}: out of disk during download -- the volume's quota is smaller than "
                f"the ~{need:.0f}GB this model needs.\n"
                f"  Models already completed are on disk and are not affected.\n"
                f"  Resize the volume, then re-run: extract.py skips finished (model, variant) "
                f"pairs, so it resumes.\n"
                f"  Set POD_DISK_GB=<volume quota in GB> to make the pre-flight guard see the quota; "
                f"statvfs reports the cluster, not your volume.") from e
        if a.limit:
            key = f"{key}_LIMIT{a.limit}"
        if a.upload:
            # Every variant must land before any weights are deleted: a partial upload that looked
            # complete is the failure that costs a second run.
            for v, _ in gs:
                dest = OUT / f"{out_key(key, v, len(gs))}.npz"
                r = subprocess.run(["rclone", "copy", "--checksum", str(dest), a.upload],
                                   capture_output=True, text=True)
                if r.returncode != 0:
                    raise SystemExit(f"upload failed for {dest.name}; not deleting anything:"
                                     f"\n{r.stderr}")
                log(f"{dest.name}: uploaded and checksummed to {a.upload}")
        if a.keep_weights:
            log(f"{key}: keeping weights (--keep-weights)")
        else:
            snap = Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface")) / "hub"
            for d in snap.glob(f"models--{spec['repo'].replace('/', '--')}*"):
                shutil.rmtree(d, ignore_errors=True)
                log(f"{key}: deleted weights from {d.name}")
        el = (time.time() - t_start) / 3600
        log(f"progress {i}/{len(models)}  elapsed {el * 60:.0f}min")
        if el > 3:
            raise SystemExit("over three hours; stopping to diagnose rather than continuing")

    log(f"done: {len(models)} models in {(time.time() - t_start) / 60:.0f}min")
    if a.poweroff:
        log("powering off")
        subprocess.run(["sudo", "poweroff"])


if __name__ == "__main__":
    main()
