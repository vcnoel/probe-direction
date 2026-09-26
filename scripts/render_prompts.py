"""Render every prompt string locally and ship them as text. Correctness, not optimisation.

A chat template is code. Some interpolate the current date, so the same call on two days produces two
different prompts -- this project has already lost a headline to that class, and to CRLF surviving a
CSV round-trip. If the pod renders prompts, the prompts depend on the pod. So they are rendered here,
hashed, and the pod tokenises these exact strings and never calls apply_chat_template.

Prompts are grouped by chat-template hash rather than by family, because a family is not guaranteed
to share one template: OLMo-2-32B omits the leading bos_token that 1B, 7B and 13B prepend. Grouping
by template makes that visible instead of averaging over it, and the per-model bos flag is recorded
so downstream analysis can assert it rather than assume it.

Declared conventions travel with the file: LF newlines, no truncation, and the pooling filter
specification (drop BOS and whitespace-only tokens). The filter itself runs on the pod because it
needs token ids, but its definition is shipped so the pod cannot invent one.

    python scripts/render_prompts.py
    python scripts/render_prompts.py --verify      # re-hash and compare, no writes
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from eval_awareness_spectral.prompts import build_prompts, load_framings, load_tasks  # noqa: E402
from extract_model import SAD_INSTRUCTION, cfg, sad_items  # noqa: E402

OUTDIR = ROOT / "data" / "rendered"
PINNED_DATE = "01 Jan 2025"
# Qwen3 is a hybrid thinking model. Its default (enable_thinking unset, i.e. True) ends the prompt at
# the assistant tag, matching Qwen2.5's shape exactly; enable_thinking=False instead injects
# "<think>\n\n</think>\n\n" INSIDE the pooled region, template tokens that are neither task nor
# wrapper. Both are defensible renderings of the same prompt and the choice is exactly the kind of
# undeclared latitude this paper measures, so we render both and carry the switch as a level of the
# implementation facet rather than picking one and calling it the prompt. `think` is the level that
# matches the Qwen2.5 ladder's shape and stays primary; `nothink` is the comparison.
# Qwen3.5 differs in two ways that both matter here. Its template's default is NOT uniform across the
# ladder: 0.8B and 2B ship a template that defaults to thinking off, 4B/9B/27B one that defaults to
# on, so rendering with the default would confound the switch with scale. Both levels therefore pass
# enable_thinking explicitly. And neither level ends at the assistant tag: `think` appends
# "<think>\n", `nothink` the closed "<think>\n\n</think>\n\n". No Qwen3.5 rendering matches the
# Qwen2.5 shape, which is a declared difference, not a level we can choose away.
VARIANTS: dict = {"qwen3": [("think", {}), ("nothink", {"enable_thinking": False})],
                  "qwen3.5": [("think", {"enable_thinking": True}),
                              ("nothink", {"enable_thinking": False})]}
DEFAULT_VARIANTS = [("default", {})]
SPECIALS = ["<|begin_of_text|>", "<s>", "<BOS>", "<bos>", "<|endoftext|>"]


def render_sad_text(tok, body, system_role, template_kwargs):
    body = body.replace("\r\n", "\n")
    if system_role:
        msgs = [{"role": "system", "content": SAD_INSTRUCTION},
                {"role": "user", "content": f"{body}\n\nAnswer:"}]
    else:
        msgs = [{"role": "user", "content": f"{SAD_INSTRUCTION}\n\n{body}\n\nAnswer:"}]
    try:
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                       date_string=PINNED_DATE, **template_kwargs)
    except TypeError:
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                       **template_kwargs)


def payload_hash(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False)
                          .encode("utf-8")).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true")
    a = ap.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    from transformers import AutoTokenizer

    ladders = json.loads((ROOT / "configs" / "ladders.json").read_text(encoding="utf-8"))
    conf = cfg()
    items = sad_items(conf)
    tasks = load_tasks(str(ROOT / "data" / "tasks_bench.jsonl"))
    framings = load_framings(str(ROOT / "data" / "framings_multi.json"))
    names = [k for k in framings if not k.startswith("_")]

    # group models by chat-template hash
    groups: dict = {}
    for family, L in ladders["ladders"].items():
        for m in L["models"]:
            # revision is recorded for ladders verified after pinning was added (Qwen3.5); older
            # entries have none and resolve exactly as they did when their hashes were recorded.
            tok = AutoTokenizer.from_pretrained(m["repo"], revision=m.get("revision"))
            th = hashlib.sha256((tok.chat_template or "").encode("utf-8")).hexdigest()[:12]
            groups.setdefault((family, th), {"tok": tok, "repos": []})["repos"].append(m["repo"])

    OUTDIR.mkdir(parents=True, exist_ok=True)
    manifest = {"pinned_date": PINNED_DATE, "variants": {k: [v for v, _ in vs]
                                                         for k, vs in VARIANTS.items()},
                "newlines": "LF", "truncation": "none",
                "pooling_filter": {"drop_whitespace_only": True, "drop_specials": SPECIALS,
                                   "applied": "on the pod, at pooling; needs token ids"},
                "groups": []}

    seen_texts: dict = {}
    jobs = [((family, th), g["tok"], sorted(g["repos"]), variant, tkw)
            for (family, th), g in sorted(groups.items())
            for variant, tkw in VARIANTS.get(family, DEFAULT_VARIANTS)]
    for (family, th), tok, repos, variant, tkw in jobs:
        sad_texts = [render_sad_text(tok, it["body"], True, tkw) for it in items]
        prompts = build_prompts(tasks, framings, tok, only_conditions=names,
                                position_match=False,
                                template_kwargs=dict(tkw, date_string=PINNED_DATE))
        fr_texts = [p.text.replace("\r\n", "\n") for p in prompts]
        for t in sad_texts + fr_texts:
            if "\r" in t:
                raise SystemExit(f"{family}/{th}: CR survived rendering")
        # A variant that renders identically to another is a declared level that does not exist;
        # silently shipping two identical prompt sets would make the facet look measured when it
        # is not. Compare on content, not on the switch we asked for.
        key = payload_hash([sad_texts, fr_texts])
        if key in seen_texts and seen_texts[key][0] == (family, th):
            raise SystemExit(
                f"{family}/{th}: variant {variant!r} renders byte-identically to "
                f"{seen_texts[key][1]!r}. The template ignores {tkw}, so this is not a level of "
                f"anything; remove it from VARIANTS or fix the kwarg.")
        seen_texts[key] = ((family, th), variant)

        bos = tok.bos_token
        payload = {
            "family": family,
            "variant": variant,
            "template_kwargs": tkw,
            "template_sha12": th,
            "repos": repos,
            "n_sad": len(sad_texts),
            "n_framing": len(fr_texts),
            "bos_token": bos,
            "bos_in_rendered": bool(bos) and sad_texts[0].startswith(bos),
            "sad": [{"text": t, "label": it["type"], "qid": it.get("qid")}
                    for t, it in zip(sad_texts, items)],
            "framing": [{"text": t, "name": p.condition, "task": p.task_id}
                        for t, p in zip(fr_texts, prompts)],
        }
        payload["sha256"] = payload_hash({k: v for k, v in payload.items() if k != "sha256"})
        # Families with one rendering keep their existing filename, so adding this axis does not
        # invalidate prompt sets that are already hashed and verified.
        stem = f"prompts_rendered_{family}_{th}"
        dest = OUTDIR / (f"{stem}.json" if variant == "default" else f"{stem}_{variant}.json")

        if a.verify:
            if not dest.exists():
                raise SystemExit(f"missing {dest.name}; run without --verify first")
            have = json.loads(dest.read_text(encoding="utf-8"))
            if have["sha256"] != payload["sha256"]:
                raise SystemExit(f"{dest.name}: hash mismatch, rendering is not reproducible")
            print(f"  verified {dest.name}  {payload['sha256'][:12]}")
        else:
            dest.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
            print(f"  {dest.name}  sad={payload['n_sad']} framing={payload['n_framing']} "
                  f"bos_in_rendered={payload['bos_in_rendered']}  {payload['sha256'][:12]}")
        manifest["groups"].append({
            "family": family, "variant": variant, "template_kwargs": tkw,
            "template_sha12": th, "repos": repos, "file": dest.name,
            "sha256": payload["sha256"], "n_sad": payload["n_sad"],
            "n_framing": payload["n_framing"],
            "bos_in_rendered": payload["bos_in_rendered"]})

    mpath = OUTDIR / "manifest.json"
    if a.verify:
        old = json.loads(mpath.read_text(encoding="utf-8"))
        if [g["sha256"] for g in old["groups"]] != [g["sha256"] for g in manifest["groups"]]:
            raise SystemExit("manifest hashes differ from a fresh render")
        print("manifest verified")
    else:
        mpath.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(f"\nwrote {mpath}")
        ngroups = len(manifest["groups"])
        nrepos = sum(len(g["repos"]) for g in manifest["groups"])
        print(f"{ngroups} template-by-variant groups covering {nrepos} model slots; the pod "
              f"tokenises these strings and never renders")
        for fam, vs in VARIANTS.items():
            got = sorted(g["variant"] for g in manifest["groups"] if g["family"] == fam)
            if not got:
                raise SystemExit(f"{fam} declares variants {[v for v, _ in vs]} but rendered none; "
                                 f"the family is missing from configs/ladders.json")
            print(f"  {fam}: implementation levels {got} rendered as separate prompt sets")
        noted = set()
        for g in manifest["groups"]:
            if not g["bos_in_rendered"] and g["template_sha12"] not in noted:
                noted.add(g["template_sha12"])
                print(f"  NOTE {g['family']}/{g['template_sha12']}: template does not prepend BOS "
                      f"({', '.join(r.split('/')[-1] for r in g['repos'])}). The declared pooling "
                      f"filter drops BOS anyway, so the primary statistic is unaffected; "
                      f"all-token comparisons across this ladder are not.")


if __name__ == "__main__":
    main()
