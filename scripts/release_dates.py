"""Release month of every checkpoint the paper measures, from the model hub's own record.

A reviewer reads a model set by its age, so the manuscript prints a release date beside every model.
The date is the hub repository's creation time (`createdAt` in the public model API), which is the
earliest public trace of a checkpoint that is recorded uniformly across vendors. It can precede the
vendor's announcement by a few days, never follow it. Fetched once and stored, so the build does not
depend on the network.

    python scripts/release_dates.py            # fetch and write configs/release_dates.json
"""

from __future__ import annotations

import json
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "configs" / "release_dates.json"


def repos() -> dict[str, list[str]]:
    """Ladder name -> hub ids, read from the two configs that pin the models."""
    g = json.loads((ROOT / "configs" / "gstudy_models.json").read_text(encoding="utf-8"))["models"]
    out: dict[str, list[str]] = {}
    for key, v in g.items():
        if v.get("enabled") and "hf_id" in v and not key.startswith("qwen3"):
            out.setdefault(v["family"], []).append(v["hf_id"])
    lad = json.loads((ROOT / "configs" / "ladders.json").read_text(encoding="utf-8"))["ladders"]
    for name, v in lad.items():
        out[name] = [m["repo"] for m in v["models"]]
    return out


def created(repo: str) -> str:
    with urllib.request.urlopen(f"https://huggingface.co/api/models/{repo}", timeout=30) as r:
        return json.loads(r.read().decode("utf-8"))["createdAt"]


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    table = {}
    for ladder, ids in repos().items():
        rows = {i: created(i) for i in ids}
        months = sorted(r[:7] for r in rows.values())
        table[ladder] = {"models": rows, "first_month": months[0], "last_month": months[-1]}
        print(f"  {ladder:10s} {months[0]} .. {months[-1]}  ({len(ids)} checkpoints)")
    OUT.write_text(json.dumps({
        "source": "model hub API createdAt, the repository creation time",
        "fetched_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%MZ"),
        "ladders": table}, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
