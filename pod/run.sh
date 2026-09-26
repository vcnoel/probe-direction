#!/usr/bin/env bash
# GPU-host bootstrap. Idempotent: safe to re-run after a dropped session, a killed process, or a
# host that rebooted. Every step either is already done or does itself again from scratch.
#
# GPU time is the scarce resource, so the ordering is deliberate:
#   1. verify the prompts BEFORE installing anything          -- a hash mismatch means stop
#   2. verify the upload target BEFORE the first forward pass -- results that cannot leave are lost
#   3. extract ascending in size                              -- fail on the 0.6B, not the 32B
#   4. upload and verify each model, then delete its weights  -- the disk is smaller than the ladder
#
#   POD_UPLOAD        rclone target, e.g. remote:results          REQUIRED unless DRY=1
#   POD_PROMPTS       rclone source holding data/rendered/prompts_rendered_*.json. The prompt
#                     bodies quote SAD benchmark questions verbatim and are deliberately NOT in
#                     git, so they are fetched here and checked against the manifest, which is.
#                     Omit only if you have already copied them onto the host by other means.
#   POD_LADDER        qwen3 (default) | olmo2 | qwen3.5  (qwen3.5: also set POD_OUT=results/pod_qwen35
#                     and copy in any npz an earlier run already produced; they are skipped)
#   POD_VRAM_GB       override device detection
#   HF_TOKEN          only needed for gated repos; neither target ladder is gated
#   DRY=1             resolve everything, touch no GPU, require no upload target
#   PY                python interpreter (default python3, then python) -- set it to rehearse this
#                     script on a machine whose `python` is not the one with torch installed
#
#   bash pod/run.sh
set -euo pipefail

cd "$(dirname "$0")/.."
ROOT=$(pwd)
LADDER=${POD_LADDER:-qwen3}
DRY=${DRY:-0}
PY=${PY:-$(command -v python3 || command -v python)}
[ -x "$PY" ] || { echo "no python found; set PY=/path/to/python" >&2; exit 1; }

say() { printf '\n\033[1m== %s\033[0m\n' "$*"; }

# ---------------------------------------------------------------------------- 0. where things live
# HF_HOME goes on the largest writable volume, not $HOME: provider images give / a small disk and
# mount the real one elsewhere, and a 66GB download into a 20GB / is the classic failure.
# HF_HOME_CANDIDATES lists the directories to consider (default: /data /scratch).
if [ -z "${HF_HOME:-}" ]; then
  best=$ROOT avail=0
  for c in ${HF_HOME_CANDIDATES:-/data /scratch} "$ROOT" "$HOME"; do
    [ -d "$c" ] && [ -w "$c" ] || continue
    a=$(df -Pk "$c" | awk 'NR==2{print $4}')
    [ "$a" -gt "$avail" ] && { avail=$a; best=$c; }
  done
  export HF_HOME="$best/hf"
  printf 'HF_HOME=%s (%s GB free)\n' "$HF_HOME" "$((avail / 1024 / 1024))"
fi
mkdir -p "$HF_HOME"
export HF_HUB_DISABLE_PROGRESS_BARS=1     # keeps the log readable in a scrollback
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

# ------------------------------------------------------------------- 1. the prompts, before anything
# Fetched rather than cloned: the repo carries the manifest (hashes and counts) but not the prompt
# bodies, because those quote benchmark questions verbatim. The hash check below is what makes the
# out-of-band copy trustworthy -- it is the same check either way.
if [ -n "${POD_PROMPTS:-}" ]; then
  say "fetching rendered prompts from $POD_PROMPTS"
  command -v rclone >/dev/null || { curl -fsSL https://rclone.org/install.sh | bash; }
  rclone copy --checksum --include 'prompts_rendered_*.json' "$POD_PROMPTS" data/rendered/
fi

say "verifying rendered prompts (no network, no GPU)"
if ! ls data/rendered/prompts_rendered_*.json >/dev/null 2>&1; then
  echo "no prompt bodies in data/rendered/. They are not in git by design: set POD_PROMPTS to an" >&2
  echo "rclone source holding them, or copy them onto this host yourself." >&2
  exit 1
fi
"$PY" - <<'PY'
import hashlib, json, sys
from pathlib import Path
d = Path("data/rendered")
m = json.loads((d / "manifest.json").read_text(encoding="utf-8"))
for g in m["groups"]:
    p = json.loads((d / g["file"]).read_text(encoding="utf-8"))
    rec = p.pop("sha256")
    got = hashlib.sha256(json.dumps(p, sort_keys=True, ensure_ascii=False)
                         .encode("utf-8")).hexdigest()
    if not (rec == g["sha256"] == got):
        sys.exit(f"{g['file']}: hash mismatch. Re-render locally and re-upload; the prompts on "
                 f"this host are not the prompts analysed.")
    print(f"  ok {g['file']}  {rec[:12]}  variant={g.get('variant', 'default')}")
print(f"{len(m['groups'])} prompt sets verified")
PY

# --------------------------------------------------------------------------- 2. dependencies, pinned
# Pinned to the versions the local rehearsal ran under. A provider image ships whatever it ships, and a
# transformers minor bump has changed hidden_states shapes before.
# DRY=1 installs nothing: a rehearsal of this script on a working machine must not be able to
# change that machine's environment, which an earlier version of it did.
say "checking pinned dependencies"
if "$PY" - <<'PY'
import importlib.metadata as md, sys
want = {"torch": "2.11.0", "transformers": "5.8.0", "numpy": "2.4.4"}
bad = []
for p, v in want.items():
    try:
        got = md.version(p)
    except md.PackageNotFoundError:
        bad.append(f"{p} missing"); continue
    if not got.startswith(v):
        bad.append(f"{p} {got}, want {v}")
for p in ("huggingface_hub", "accelerate"):
    try:
        md.version(p)
    except md.PackageNotFoundError:
        bad.append(f"{p} missing")
print("  " + ("; ".join(bad) if bad else "all pins satisfied"))
sys.exit(1 if bad else 0)
PY
then :
elif [ "$DRY" = "1" ]; then
  echo "  DRY=1: not installing anything"
else
  "$PY" -m pip install --quiet "torch==2.11.0" "transformers==5.8.0" "numpy==2.4.4" \
      huggingface_hub accelerate
fi
# hf_transfer parallelises the download, which is the per-model bottleneck, but it is optional and
# its env flag is fatal when the package is absent -- so the flag is set only if the import works.
if [ "$DRY" != "1" ] && ! "$PY" -c "import hf_transfer" 2>/dev/null; then
  "$PY" -m pip install --quiet hf_transfer || true
fi
if "$PY" -c "import hf_transfer" 2>/dev/null; then
  export HF_HUB_ENABLE_HF_TRANSFER=1
  echo "  hf_transfer enabled"
else
  echo "  hf_transfer unavailable; downloads will be single-stream"
fi
"$PY" - <<'PY'
import torch, transformers, numpy
print(f"  torch {torch.__version__}  transformers {transformers.__version__}  "
      f"numpy {numpy.__version__}")
print(f"  cuda={torch.cuda.is_available()}", end="")
if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    print(f"  {p.name}  {p.total_memory / 1e9:.0f}GB")
else:
    print()
PY

# ------------------------------------------------------------- 3. the exit route, before any GPU work
if [ "$DRY" != "1" ]; then
  say "verifying the upload target before spending GPU time"
  : "${POD_UPLOAD:?set POD_UPLOAD to an rclone target, or DRY=1. Results that cannot leave the host are lost when it dies.}"
  command -v rclone >/dev/null || { curl -fsSL https://rclone.org/install.sh | bash; }
  # A real round trip: a target that lists but cannot be written to is the failure that matters.
  probe=$(mktemp -d)/upload-probe.txt
  date -u +%FT%TZ > "$probe"
  rclone copy --checksum "$probe" "$POD_UPLOAD" || {
    echo "cannot write to $POD_UPLOAD; fix credentials before extracting anything" >&2; exit 1; }
  rclone lsf "$POD_UPLOAD" | grep -q upload-probe.txt || {
    echo "wrote to $POD_UPLOAD but cannot see the file; not proceeding" >&2; exit 1; }
  rclone delete "$POD_UPLOAD/upload-probe.txt" || true
  echo "  upload target writable and readable: $POD_UPLOAD"
fi

# ---------------------------------------------------------------------------------- 4. plan, then run
say "plan"
"$PY" pod/extract.py --ladder "$LADDER" --dry-run

if [ "$DRY" = "1" ]; then
  say "DRY=1: stopping before the GPU"
  exit 0
fi

say "extracting ladder $LADDER"
# Idempotent by construction: extract.py skips any (model, variant) whose npz already exists, so a
# re-run after a dropped session resumes rather than repeating.
"$PY" pod/extract.py --ladder "$LADDER" --upload "$POD_UPLOAD"

say "done"
echo "Results are in $POD_UPLOAD. Ingest locally with:"
echo "    python scripts/ingest_pod.py --src <downloaded dir>"
echo "    python scripts/rehearsal_check.py <keys>"
if [ "${POD_POWEROFF:-0}" = "1" ]; then
  echo "powering off in 60s (POD_POWEROFF=1); ^C to cancel"
  sleep 60
  poweroff
fi
