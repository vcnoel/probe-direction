"""Where the behavioural anchor's model revisions come from, checked against the provider's table.

pod/sad_provider/hf_local_provider.py pins one Hugging Face revision per model, meant to be the
revision of the stored probe arrays. This script reads each revision where it was recorded and
writes the record with its source:

- Qwen3 and OLMo-2: the `meta` field `revision` of the stored probe arrays, which the pod's
  extraction wrote (results/pod_qwen3/<model>__nothink.npz, results/pod_all/<model>.npz under
  EAS_ACTIVATION_ROOT; these arrays are not in the repository), with each array's sha256;
- Qwen3.5: configs/ladders.json, checked against the arrays' meta as well;
- Qwen2.5: configs/gstudy_models.json (the paper's g-study arrays carry no meta).

It fails if any provider revision differs from its record.

    EAS_ACTIVATION_ROOT=<root holding results/pod_*> python scripts/record_revisions.py
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from result_io import write_result  # noqa: E402

ACT = Path(os.environ.get("EAS_ACTIVATION_ROOT", str(ROOT)))
OUT = ROOT / "results" / "post_submission" / "model_revisions.json"
ARRAYS = {"qwen3-": "results/pod_qwen3/{}__nothink.npz", "qwen3.5-": "results/pod_qwen35/{}__nothink.npz",
          "olmo-2-": "results/pod_all/{}.npz"}


def provider_models():
    spec = importlib.util.spec_from_file_location("p", ROOT / "pod" / "sad_provider" / "hf_local_provider.py")
    m = importlib.util.module_from_spec(spec)
    sys.modules["evalugator"] = sys.modules.get("evalugator") or _stub()
    spec.loader.exec_module(m)
    return m.MODELS


def _stub():
    """The provider imports evalugator's request types; only its MODELS table is read here."""
    import types
    ev = types.ModuleType("evalugator")
    api = types.ModuleType("evalugator.api")
    req = types.ModuleType("evalugator.api.requests")
    req.GetTextRequest = req.GetTextResponse = object
    sys.modules.update({"evalugator.api": api, "evalugator.api.requests": req})
    return ev


def main() -> None:
    models = provider_models()
    ladders = json.loads((ROOT / "configs" / "ladders.json").read_text(encoding="utf-8"))["ladders"]
    l35 = {m["repo"]: m.get("revision") for m in ladders["qwen3.5"]["models"]}
    gst = json.loads((ROOT / "configs" / "gstudy_models.json").read_text(encoding="utf-8"))["models"]
    out, bad = {}, []
    for name, (repo, rev) in models.items():
        rec = {"repo": repo, "provider_revision": rev, "sources": []}
        prefix = next((p for p in ("qwen3.5-", "qwen3-", "olmo-2-") if name.startswith(p)), None)
        if prefix:
            f = ACT / ARRAYS[prefix].format(name)
            with np.load(f, allow_pickle=False) as z:
                meta = json.loads(str(z["meta"]))
            rec["sources"].append({"source": ARRAYS[prefix].format(name) + " (meta.revision, "
                                   "EAS_ACTIVATION_ROOT; not in the repository)",
                                   "revision": meta.get("revision"), "repo": meta.get("repo"),
                                   "array_sha256": hashlib.sha256(f.read_bytes()).hexdigest()})
        if name.startswith("qwen3.5-"):
            rec["sources"].append({"source": "configs/ladders.json", "revision": l35.get(repo), "repo": repo})
        if name.startswith("qwen2.5-"):
            g = gst[name]
            rec["sources"].append({"source": "configs/gstudy_models.json", "revision": g.get("revision"),
                                   "repo": g.get("hf_id")})
        rec["agrees"] = bool(rec["sources"]) and all(s["revision"] == rev and s["repo"] == repo
                                                     for s in rec["sources"])
        if not rec["agrees"]:
            bad.append(name)
        out[name] = rec
        print(f"{name:26s} {rev[:12]}  " + "; ".join(f"{s['source'].split(' ')[0]} "
                                                     f"{(s['revision'] or '-')[:12]}" for s in rec["sources"])
              + ("" if rec["agrees"] else "  DISAGREES"))
    if bad:
        raise SystemExit(f"provider revisions disagree with their records: {bad}")
    write_result(OUT, {"models": out}, kind="measured",
                 statistic="none: model revisions with their sources",
                 note="Qwen3 and OLMo-2 revisions are recorded only in the stored probe arrays' meta",
                 pooling="not applicable", layers="not applicable", scoring_array="not applicable",
                 data_root="EAS_ACTIVATION_ROOT probe arrays and configs/", model_population=sorted(out))
    print(f"wrote {OUT.relative_to(ROOT).as_posix()}")


if __name__ == "__main__":
    main()
