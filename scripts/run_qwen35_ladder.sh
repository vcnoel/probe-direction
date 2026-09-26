#!/usr/bin/env bash
# Extract the Qwen3.5 ladder on whatever single GPU this machine has. Resumable: extract.py skips any
# (model, variant) whose npz already exists, so a killed run is re-run, not restarted.
#
# The same script serves a small and a large GPU, and neither needs to know about the other:
#   16GB card    holds 0.8B, 2B and 4B in bf16. 9B (17.9GB of text weights) and 27B do not fit
#                and are skipped with a log line (--fit-only), not failed.
#   80GB card    copy the small card's npz into $POD_OUT first; everything done is skipped and
#                only 9B and 27B are extracted.
# Ascending in size: a broken DeltaNet code path should fail on the 0.8B.
#
# Nothing is rendered here. The prompts are the hashed files in data/rendered/, rendered locally by
# scripts/render_prompts.py; they quote SAD questions verbatim and must never be committed or
# uploaded anywhere public. extract.py refuses to start if any hash disagrees with the manifest.
#
#   PY              python with torch 2.11 / transformers 5.8 (the pins in pod/run.sh), e.g.
#                   PY=/path/to/env/bin/python
#   POD_OUT         output directory (default results/pod_qwen35, where pod_ladders.py reads)
#   POD_UPLOAD      rclone target; if set, each model is pushed and checksummed after extraction
#   DELETE_WEIGHTS  1 deletes each snapshot after its npz is written (pod/run.sh deletes by
#                   default; here the default is to keep, because re-downloading is the
#                   failure-prone step)
#   POD_VRAM_GB     override detected VRAM
#   DRY=1           verify prompts and print the plan; no GPU, no download
#
#   bash scripts/run_qwen35_ladder.sh
set -euo pipefail

cd "$(dirname "$0")/.."
PY=${PY:-$(command -v python3 || command -v python)}
DRY=${DRY:-0}
export POD_OUT=${POD_OUT:-results/pod_qwen35}
export HF_HUB_DISABLE_PROGRESS_BARS=1
export HF_HUB_DISABLE_SYMLINKS_WARNING=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
mkdir -p "$POD_OUT"
LOG="$POD_OUT/extract.log"

say() { printf '\n\033[1m== %s\033[0m\n' "$*"; }

say "environment"
"$PY" - <<'PY'
import importlib.util as u
import torch, transformers
print(f"  torch {torch.__version__}  transformers {transformers.__version__}")
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES as M
if "qwen3_5" not in M:
    raise SystemExit("  this transformers cannot load Qwen3.5; use the pinned 5.8.0")
fast = all(u.find_spec(p) for p in ("fla", "causal_conv1d"))
print(f"  DeltaNet kernels: {'fla + causal_conv1d' if fast else 'torch fallback (slower, same maths)'}")
if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    print(f"  {p.name}  {p.total_memory / 1e9:.0f}GB")
else:
    print("  no CUDA device")
PY

say "plan (verifies every prompt hash against the manifest)"
"$PY" pod/extract.py --ladder qwen3.5 --dry-run

if [ "$DRY" = "1" ]; then
  say "DRY=1: stopping before the GPU"
  exit 0
fi

say "extracting qwen3.5 into $POD_OUT (log: $LOG)"
args=(--ladder qwen3.5 --fit-only)
[ "${DELETE_WEIGHTS:-0}" = "1" ] || args+=(--keep-weights)
[ -n "${POD_UPLOAD:-}" ] && args+=(--upload "$POD_UPLOAD")
"$PY" pod/extract.py "${args[@]}" 2>&1 | tee -a "$LOG"

say "status"
missing=0
for m in qwen3.5-0.8b qwen3.5-2b qwen3.5-4b qwen3.5-9b qwen3.5-27b; do
  for v in think nothink; do
    f="$POD_OUT/${m}__${v}.npz"
    if [ -f "$f" ]; then echo "  ok       $f"; else echo "  missing  $f"; missing=$((missing + 1)); fi
  done
done
if [ "$missing" -eq 0 ]; then
  echo "  ladder complete; analyse with: $PY scripts/pod_ladders.py"
else
  echo "  $missing outputs missing; the rest need a larger GPU (see the skip lines above)"
fi
