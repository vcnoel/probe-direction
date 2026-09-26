"""Convert pod extraction output into the layout every analysis script already reads.

The pod writes one npz per (model, rendering) holding three pooled reductions. The analyses read
two files per model under results/gstudy/<key>/:

    pooled.npz             sad, sad_label, sad_qid, framing, framing_name, framing_task, families
                           pooled over EVERY token -- the direction arrays and the all-token
                           scoring level
    sad_filtered_lf.npz    sad, sad_label
                           pooled over the token set the released scoring code uses (BOS and
                           whitespace-only tokens dropped), on LF-normalised text -- the `lf`
                           level, and the paper's declared scoring array

Those two files were originally produced by two separate GPU passes (extract_model.py and
extract_sad_filtered.py). The pod produces both reductions from one pass, so this script is a
rename and a dtype cast, not a computation -- which is the point: nothing here can change a number,
so the pod's output and the existing Qwen2.5 arrays are comparable by construction.

Two declared differences from the original scripts, recorded in provenance.json rather than left to
be discovered later:

  * The pod applies no truncation; extract_sad_filtered.py truncated at --max-len. No rendered
    prompt in this project reaches that length, so the token sets coincide, but the convention
    differs and is recorded.
  * The pod's dropped-specials set adds <|endoftext|> and <bos> to the original three. Neither
    appears in a Qwen3 or OLMo-2 rendering, so again the sets coincide here.

Refuses to write into an existing directory without --force. The paper's Qwen2.5 arrays live in the
same tree and an overwrite of one of those has already cost this project a re-extraction.

    python scripts/ingest_pod.py --src results/rehearsal --dry-run
    python scripts/ingest_pod.py --src results/pod
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

GSTUDY = ROOT / "results" / "gstudy"
FRAMINGS = ROOT / "data" / "framings_multi.json"


def families():
    f = json.loads(FRAMINGS.read_text(encoding="utf-8"))
    return {k: v["family"] for k, v in f.items() if not k.startswith("_")}


def ingest(src: Path, fam: dict, force: bool, dry: bool):
    with np.load(src, allow_pickle=False) as z:
        meta = json.loads(str(z["meta"]))
        key = meta["key"]
        n_sad = z["sad_validated"].shape[0]
        n_fr = z["framing_validated"].shape[0]
        names = [str(s) for s in z["framing_name"]]
        unknown = sorted(set(names) - set(fam))
        if unknown:
            raise SystemExit(f"{src.name}: framings {unknown} are not in {FRAMINGS.name}; the "
                             f"prompt set and the framing definitions disagree")
        if n_sad == 0 or n_fr == 0:
            raise SystemExit(f"{src.name}: empty prompt set, refusing to write a file that would "
                             f"make every downstream check pass vacuously")
        if meta.get("truncated_to"):
            raise SystemExit(f"{src.name}: extracted with --limit {meta['truncated_to']}; a "
                             f"truncated rehearsal must not be ingested as a result")
        dest = GSTUDY / key
        if dest.exists() and not force:
            raise SystemExit(f"{dest} exists; pass --force only if you mean to replace arrays that "
                             f"published numbers may depend on")
        if dry:
            print(f"  {src.name} -> {dest.relative_to(ROOT)}  sad={n_sad} framing={n_fr} "
                  f"layers={z['sad_validated'].shape[1]} d={z['sad_validated'].shape[2]}")
            return None

        arrays = {
            "pooled.npz": {
                "sad": z["sad_alltokens"].astype(np.float32),
                "sad_label": z["sad_label"].astype(np.int64),
                "sad_qid": np.arange(n_sad, dtype=np.int64),
                "framing": z["framing_alltokens"].astype(np.float32),
                "framing_name": np.array(names),
                "framing_task": np.array([str(s) for s in z["framing_task"]]),
                "families": np.array(json.dumps({k: fam[k] for k in dict.fromkeys(names)})),
            },
            "sad_filtered_lf.npz": {
                "sad": z["sad_validated"].astype(np.float32),
                "sad_label": z["sad_label"].astype(np.int64),
            },
        }
        for name, a in arrays.items():
            for k, v in a.items():
                if v.dtype.kind == "f" and not np.isfinite(v).all():
                    raise SystemExit(f"{src.name}: {name}/{k} holds non-finite values")
        lab = arrays["pooled.npz"]["sad_label"]
        if len(set(lab.tolist())) < 2:
            raise SystemExit(f"{src.name}: sad_label has one class; AUROC would be undefined")

        dest.mkdir(parents=True, exist_ok=True)
        for name, a in arrays.items():
            tmp = dest / (name + ".tmp.npz")
            np.savez_compressed(tmp, **a)
            with np.load(tmp, allow_pickle=False) as chk:      # verify before rename
                assert chk["sad"].shape[0] == n_sad
            tmp.replace(dest / name)
        prov = {
            "ingested_from": src.name, "pod_meta": meta,
            "mapping": {"pooled.npz:sad|framing": "pod alltokens reduction",
                        "sad_filtered_lf.npz:sad": "pod validated reduction "
                                                   "(BOS and whitespace-only dropped)"},
            "declared_differences_from_original_scripts": [
                "no truncation on the pod; extract_sad_filtered.py truncated at --max-len",
                "dropped-specials set adds <|endoftext|> and <bos>",
            ],
            "storage": "float16 on the pod, cast to float32 here to match the existing arrays",
        }
        (dest / "provenance.json").write_text(json.dumps(prov, indent=2), encoding="utf-8")
        print(f"  {src.name} -> {dest.relative_to(ROOT)}  sad={n_sad} framing={n_fr} "
              f"layers={arrays['pooled.npz']['sad'].shape[1]}")
        return key


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="results/pod")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    src = (ROOT / a.src) if not Path(a.src).is_absolute() else Path(a.src)
    files = sorted(p for p in src.glob("*.npz") if not p.name.endswith(".tmp.npz"))
    if not files:
        raise SystemExit(f"no npz in {src}; nothing to ingest")
    fam = families()
    print(f"{len(files)} extraction files in {src}")
    done = [ingest(p, fam, a.force, a.dry_run) for p in files]
    done = [d for d in done if d]
    if a.dry_run:
        print("\n  dry run; nothing written")
    else:
        print(f"\ningested {len(done)} models: {', '.join(done)}")


if __name__ == "__main__":
    main()
