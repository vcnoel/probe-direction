#!/usr/bin/env bash
# Post-submission extractions on one A100 80GB, in one resumable run (docs/paraphrase-registration.md):
#
#   P3  Qwen2.5-7B framing rows, validated and all-token pooling in one pass
#       (scripts/extract_qwen25_validated.py)
#   P7  original + paraphrased framings, non-thinking rendering
#       (scripts/extract_paraphrase.py): Qwen3-8B, 14B, 32B and Qwen3.5-9B, 27B
#
# Only framing prompts are run. They are rendered here from committed data (framings and the task
# pool, no benchmark text) by the same code as on the laptop, and the prompt hash is checked
# against the laptop's before any download. The SAD scoring set is NOT re-extracted: the analysis
# scripts read it from the paper's stored arrays on the analysis machine.
#
# Resumable: every extraction skips a model whose npz exists, so a re-run after a dropped session
# continues. Each model is uploaded and checksummed before the next starts; weights are deleted
# after the npz is written (the disk is smaller than the ladder).
#
#   POD_UPLOAD      rclone target for the npz files                  REQUIRED unless DRY=1
#   DRY=1           preflight and plan only: no install, no download, no GPU
#   POD_ALLOW_GPU=1 accept a GPU other than an A100 80GB (estimates below assume an A100)
#   POD_DISK_GB     the volume's quota when statvfs cannot see it (see pod/extract.py free_gb)
#   PY              python interpreter
#
# Estimated A100 80GB time (laptop forward passes scaled by parameters, plus downloads at
# ~0.5 GB/s, plus load): see the table printed by the plan step and the session report.
#
#   bash pod/run_post_submission.sh
set -euo pipefail

cd "$(dirname "$0")/.."
ROOT=$(pwd)
DRY=${DRY:-0}
PY=${PY:-$(command -v python3 || command -v python)}
[ -x "$PY" ] || { echo "no python found; set PY=/path/to/python" >&2; exit 1; }
say() { printf '\n\033[1m== %s\033[0m\n' "$*"; }

export HF_HOME=${HF_HOME:-/workspace/hf}
[ "$HF_HOME" = "/workspace/hf" ] || echo "NOTE HF_HOME is $HF_HOME, not /workspace/hf"
export EAS_ACTIVATION_ROOT=${EAS_ACTIVATION_ROOT:-$ROOT}
export HF_HUB_DISABLE_PROGRESS_BARS=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
P7_MODELS="qwen3-8b qwen3.5-9b qwen3-14b qwen3.5-27b qwen3-32b"     # ascending: fail cheap
P3_MODELS="qwen2.5-7b"
PROMPT_SHA12=39980ca8f0b8          # the laptop rendering of the 36 x 60 framing prompts

# ------------------------------------------------------------------------------ 0. preflight
say "preflight"
mkdir -p "$HF_HOME"
"$PY" - <<'PY'
import os, shutil, subprocess, sys
gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
                     capture_output=True, text=True)
if gpu.returncode != 0:
    sys.exit("  no nvidia-smi: no GPU on this host")
name, mem = [s.strip() for s in gpu.stdout.splitlines()[0].split(",")]
print(f"  GPU {name}, {int(mem) / 1024:.0f} GiB")
if not ("A100" in name and int(mem) >= 79000) and os.environ.get("POD_ALLOW_GPU") != "1":
    sys.exit("  expected an A100 80GB; set POD_ALLOW_GPU=1 to proceed on this GPU")
sys.path.insert(0, "pod")
from extract import free_gb
for p in ("/workspace", os.environ["HF_HOME"]):
    if os.path.isdir(p):
        f = free_gb(p)
        print(f"  free disk at {p}: {f:.0f} GB")
        if f < 150:
            sys.exit(f"  need at least 150 GB free at {p} (largest download 65.5 GB, outputs ~11 GB)")
print(f"  HF_HOME={os.environ['HF_HOME']}  EAS_ACTIVATION_ROOT={os.environ['EAS_ACTIVATION_ROOT']}")
PY

# ------------------------------------------------------------------- 1. dependencies, pinned
# The pins of pod/run.sh: transformers 5.8.0 is the version the paper's Qwen3.5 arrays were
# extracted with, and it loads the Qwen3.5 text decoder through AutoModelForCausalLM.
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
if [ "$DRY" != "1" ] && ! "$PY" -c "import hf_transfer" 2>/dev/null; then
  "$PY" -m pip install --quiet hf_transfer || true
fi
"$PY" -c "import hf_transfer" 2>/dev/null && export HF_HUB_ENABLE_HF_TRANSFER=1 || true

# --------------------------------------------- 2. the prompts are the laptop's prompts
say "rendering check (tokenizers only)"
for m in $P7_MODELS; do
  line=$("$PY" scripts/extract_paraphrase.py --render-only --model "$m" 2>/dev/null | tail -1)
  echo "  $line"
  case "$line" in
    *"sha $PROMPT_SHA12"*) ;;
    *) echo "  $m renders differently from the laptop (want sha $PROMPT_SHA12); stopping" >&2; exit 1 ;;
  esac
done

# ------------------------------------------------- 3. the exit route, before any GPU work
if [ "$DRY" != "1" ]; then
  say "verifying the upload target"
  : "${POD_UPLOAD:?set POD_UPLOAD to an rclone target, or DRY=1}"
  command -v rclone >/dev/null || { curl -fsSL https://rclone.org/install.sh | bash; }
  probe=$(mktemp -d)/upload-probe.txt
  date -u +%FT%TZ > "$probe"
  rclone copy --checksum "$probe" "$POD_UPLOAD" || { echo "cannot write to $POD_UPLOAD" >&2; exit 1; }
  rclone lsf "$POD_UPLOAD" | grep -q upload-probe.txt || { echo "cannot read back $POD_UPLOAD" >&2; exit 1; }
  rclone delete "$POD_UPLOAD/upload-probe.txt" || true
fi

ACTS_P7=$EAS_ACTIVATION_ROOT/results/post_submission/paraphrase/acts
ACTS_P7_SMALL=$EAS_ACTIVATION_ROOT/results/post_submission/paraphrase/acts_validated_only
ACTS_P3=$EAS_ACTIVATION_ROOT/results/post_submission/qwen25_validated/acts
say "plan"
for m in $P3_MODELS; do
  [ -f "$ACTS_P3/$m.npz" ] && echo "  P3 $m  done" || echo "  P3 $m  to extract"
done
for m in $P7_MODELS; do
  [ -f "$ACTS_P7/${m}__nothink.npz" ] && echo "  P7 $m  done" || echo "  P7 $m  to extract (batch: the paper's pod rule)"
done
if [ "$DRY" = "1" ]; then say "DRY=1: stopping before the GPU"; exit 0; fi

upload() {   # $1 file, $2 subdirectory of the target
  rclone copy --checksum "$1" "$POD_UPLOAD/$2" || { echo "upload failed for $1" >&2; exit 1; }
  echo "  uploaded $(basename "$1") -> $POD_UPLOAD/$2"
}

# ------------------------------------------------------------------------- 4. extraction
T0=$(date +%s)
for m in $P3_MODELS; do
  say "P3 $m"
  "$PY" scripts/extract_qwen25_validated.py --model "$m" --delete-weights
  upload "$ACTS_P3/$m.npz" qwen25_validated
done
for m in $P7_MODELS; do
  say "P7 $m"
  "$PY" scripts/extract_paraphrase.py --model "$m" --batch 0 --delete-weights
  f=$ACTS_P7/${m}__nothink.npz
  # A validated-only copy: what scripts/paraphrase_analysis.py reads, at half the download.
  mkdir -p "$ACTS_P7_SMALL"
  [ -f "$ACTS_P7_SMALL/$(basename "$f")" ] || "$PY" - "$f" "$ACTS_P7_SMALL/$(basename "$f")" <<'PY'
import sys, numpy as np
with np.load(sys.argv[1], allow_pickle=False) as z:
    keep = {k: z[k] for k in z.files if k != "framing_alltokens"}
tmp = sys.argv[2] + ".tmp.npz"
np.savez_compressed(tmp, **keep)
import os; os.replace(tmp, sys.argv[2])
PY
  upload "$f" paraphrase/acts
  upload "$ACTS_P7_SMALL/$(basename "$f")" paraphrase/acts_validated_only
  echo "  elapsed $(( ($(date +%s) - T0) / 60 )) min"
done

say "done in $(( ($(date +%s) - T0) / 60 )) min"
cat <<EOF
On the analysis machine (which holds results/pod_qwen3, results/pod_qwen35 and results/gstudy):
  copy paraphrase/acts_validated_only/*.npz (or paraphrase/acts/*.npz) into
       \$EAS_ACTIVATION_ROOT/results/post_submission/paraphrase/acts/
  copy qwen25_validated/*.npz into \$EAS_ACTIVATION_ROOT/results/post_submission/qwen25_validated/acts/
  python scripts/qwen25_validated.py        # P3, four rungs
  python scripts/paraphrase_analysis.py     # P7a-d on both ladders, P7c now that >= 4 rungs exist
EOF
