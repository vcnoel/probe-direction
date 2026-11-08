"""Item index for the behavioural anchor: SAD ids in the order of every probe array, no item text.

SAD's stages_oversight task evaluates test_oversight.yaml (testing) and deploy_oversight.yaml
(deployment). The probe arrays hold 400 items in one of two orders:

- "struct": the Qwen3, Qwen3.5 and OLMo-2 arrays, rendered from oversight.yaml then
  deploy_oversight.yaml (configs/gstudy_models.json, scripts/extract_model.py:sad_items). Their
  deployment items are the task's; of their testing items only those whose whitespace-normalised
  body is identical to a task item's are the task's items (different files, different ids).
- "split": the paper's Qwen2.5 arrays, from the encrypted 400-item split of scripts/sad_source.py
  (question_id 0..399, no SAD id), recovered from released files. It is the task's item set; a few
  bodies carry recovery noise, so a body without an identical task body is matched to the unique
  same-label task item of highest word 5-gram overlap, which must be at least 0.9.

Identical bodies within a file (SAD has a few, and SAD ids are content hashes, so they share an id)
render to identical prompts; among them the k-th occurrence takes the k-th free match, and every row
names its task item by position as well as by id. The index holds only SAD ids, body hashes, labels and match
methods, and its label order is checked against the label vector of every probe array the anchor
reads (registration, Amendment 1).

    EAS_SAD_ARCHIVE=... EAS_SAD_PASSWORD=... python scripts/sad_item_index.py --sad-dir <SAD clone>
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from result_io import write_result  # noqa: E402
from sad_behaviour import PROBE_FILES, SAD_COMMIT, STRUCTS, TASK_FILES, body_hash  # noqa: E402

OUT = ROOT / "results" / "post_submission" / "sad_item_index.json"
CONTROLLED = ROOT / "results" / "post_submission" / "controlled"
ORDER_OF_LADDER = {"qwen3_nothink": "struct", "qwen3.5_nothink": "struct", "olmo2": "struct",
                   "qwen2.5_paper": "split"}


def shingles(text: str, k: int = 5) -> set:
    w = re.findall(r"[a-z0-9]+", text.lower())
    return {" ".join(w[i:i + k]) for i in range(max(len(w) - k + 1, 1))}


def struct_items(sad_dir: Path, files):
    import yaml
    out = []
    for fname, label in files:
        y = yaml.safe_load((sad_dir / STRUCTS / fname).read_text(encoding="utf-8", errors="replace"))
        for s in y["samples"]:
            out.append({"sad_id": s["id"], "label": label, "body": s["body"], "file": fname})
    return out


def split_items():
    from sad_source import load_sad_bodies
    out = []
    for it in load_sad_bodies():
        b = it["body"].rstrip()
        if b.endswith("Answer:"):
            b = b[: -len("Answer:")]
        out.append({"question_id": int(it["question_id"]), "label": it["type"], "body": b})
    return out


def match(task, probe, overlap_fallback: bool):
    """-> per probe position, (task index or None, method)."""
    th = [body_hash(t["body"]) for t in task]
    by_key = {}
    for i, t in enumerate(task):
        by_key.setdefault((th[i], t["label"]), []).append(i)
    used, out = set(), [(None, "not in the task")] * len(probe)
    for j, it in enumerate(probe):
        group = by_key.get((body_hash(it["body"]), it["label"]), [])
        free = [i for i in group if i not in used]
        if free:
            used.add(free[0])
            out[j] = (free[0], "identical body" + (", identical-body group" if len(group) > 1 else ""))
    if not overlap_fallback:
        return out
    sh = [shingles(t["body"]) for t in task]
    for j, it in enumerate(probe):
        if out[j][0] is not None:
            continue
        q = shingles(it["body"])
        jac = np.array([len(q & s) / max(len(q | s), 1)
                        if task[i]["label"] == it["label"] and i not in used else -1.0
                        for i, s in enumerate(sh)])
        best = float(jac.max())
        top = list(np.flatnonzero(jac == best))
        top_h = {th[t] for t in top}
        rest = jac[[i for i in range(len(task)) if th[i] not in top_h]]
        second = float(rest.max()) if rest.size else -1.0
        if best < 0.9 or len(top_h) > 1 or second >= best:
            raise SystemExit(f"split item {it.get('question_id')}: no unique match (best overlap "
                             f"{best:.3f}, second {second:.3f})")
        used.add(int(top[0]))
        out[j] = (int(top[0]), f"overlap {best:.3f} (next {second:.3f})")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sad-dir", required=True, type=Path)
    a = ap.parse_args()
    sad_dir = a.sad_dir.resolve()
    head = subprocess.run(["git", "-C", str(sad_dir), "rev-parse", "HEAD"], capture_output=True,
                          text=True).stdout.strip()
    if head != SAD_COMMIT:
        raise SystemExit(f"SAD clone at {head}, registered {SAD_COMMIT}")
    task = struct_items(sad_dir, TASK_FILES)
    probes = {"struct": (struct_items(sad_dir, PROBE_FILES), False), "split": (split_items(), True)}
    if len(task) != 400 or any(len(p) != 400 for p, _ in probes.values()):
        raise SystemExit("expected 400 items in the task and in each probe order")

    out = {"sad_commit": SAD_COMMIT, "order_of_ladder": ORDER_OF_LADDER,
           "task_files": [f for f, _ in TASK_FILES],
           "task": [{"position": i, "sad_id": t["sad_id"], "label": t["label"],
                     "body_sha256": body_hash(t["body"])} for i, t in enumerate(task)],
           "orders": {}, "summary": {}}
    for name, (probe, fallback) in probes.items():
        m = match(task, probe, fallback)
        rows = []
        for j, (it, (i, how)) in enumerate(zip(probe, m)):
            r = {"position": j, "label": it["label"], "task_position": i,
                 "task_sad_id": None if i is None else task[i]["sad_id"], "match": how}
            if name == "struct":
                r["sad_id"] = it["sad_id"]
                r["file"] = it["file"]
            else:
                r["question_id"] = it["question_id"]
            rows.append(r)
        out["orders"][name] = rows
        shared = [r for r in rows if r["task_position"] is not None]
        if len({r["task_position"] for r in shared}) != len(shared):
            raise SystemExit(f"{name}: a task item is matched twice")
        out["summary"][name] = {
            "shared_with_task": len(shared),
            "shared_testing": sum(r["label"] == "testing" for r in shared),
            "shared_deployment": sum(r["label"] == "deployment" for r in shared),
            "methods": {k: sum(r["match"].split(" (")[0].split(" 0")[0] == k for r in rows)
                        for k in sorted({r["match"].split(" (")[0].split(" 0")[0] for r in rows})}}
    common = set.intersection(*[{r["task_position"] for r in rows if r["task_position"] is not None}
                                for rows in out["orders"].values()])
    out["shared_task_positions"] = sorted(common)
    out["summary"]["shared_by_every_order"] = {
        "n": len(common), "testing": sum(out["task"][i]["label"] == "testing" for i in common),
        "deployment": sum(out["task"][i]["label"] == "deployment" for i in common)}

    checks = {}
    for f in sorted(CONTROLLED.glob("*.npz")):
        ladder = f.stem.split("__")[0]
        if f.name.startswith("orient_") or ladder not in ORDER_OF_LADDER:
            continue
        with np.load(f, allow_pickle=False) as z:
            y = z["y"].astype(int)
        want = np.array([r["label"] == "testing" for r in out["orders"][ORDER_OF_LADDER[ladder]]],
                        int)
        checks[f.stem] = bool(np.array_equal(y, want))
    if not all(checks.values()):
        raise SystemExit(f"stored labels disagree with the index order for "
                         f"{[k for k, v in checks.items() if not v]}")
    out["label_checks"] = checks
    for name in probes:
        print(f"{name}: {out['summary'][name]}")
    print(f"shared by every order: {out['summary']['shared_by_every_order']}; label vectors of "
          f"{len(checks)} probe arrays agree with their order")
    write_result(OUT, out, kind="measured",
                 statistic="none: item index (SAD id, sha256 of the whitespace-normalised body, "
                           "label, match to the task's items) in the order of each probe array",
                 note="no item text; ids and hashes only (SAD terms)",
                 pooling="not applicable", layers="not applicable",
                 scoring_array="SAD stages_oversight task items; struct and split probe orders",
                 data_root="SAD clone at " + SAD_COMMIT[:12] + " and the encrypted 400-item split",
                 model_population=sorted({f.stem.split("__")[1] for f in CONTROLLED.glob("*.npz")
                                          if not f.name.startswith("orient_")
                                          and f.stem.split("__")[0] in ORDER_OF_LADDER}))
    print(f"wrote {OUT.relative_to(ROOT).as_posix()}")


if __name__ == "__main__":
    main()
