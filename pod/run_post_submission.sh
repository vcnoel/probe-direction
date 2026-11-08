#!/usr/bin/env bash
# Post-submission extractions on one A100 80GB, in one resumable run (docs/paraphrase-registration.md):
#
#   P7  original + paraphrased framings, non-thinking rendering
#       (scripts/extract_paraphrase.py): Qwen3-8B, 14B, 32B and Qwen3.5-9B, 27B, each preceded by
#       the amendment-2 floor extraction (scripts/paraphrase_noise_floor.py: the paper's own 18
#       framings x 60 tasks in the pod run's order and batch) under the same weight load cycle.
#       The registered gate already stopped P7 on both ladders, so these rungs add descriptive
#       numbers only (P7c mainly, which needs at least 4 rungs); see the registration's notes.
#   P3  not run here (P3_MODELS is empty): the four-rung validated Qwen2.5 answer exists from
#       results/gstudy_v11, and a cross-machine re-extraction would fail qwen25_validated.py's
#       reproduction gate and write nothing. scripts/extract_qwen25_validated.py stays pod-ready.
#
# Only framing prompts are run. They are rendered here from committed data (framings and the task
# pool, no benchmark text) by the same code as on the laptop, and the prompt hash is checked
# against the laptop's before any download. The SAD scoring set is NOT re-extracted: the analysis
# scripts read it from the paper's stored arrays on the analysis machine.
#
# Provenance: run from a clean checkout of this branch at a recorded commit. The commit and the
# working-tree status are printed here and written into every extraction's meta; a tree with
# modified tracked files is refused. Do NOT apply rename_package.patch: it renames the package
# but misses scripts/extract_paraphrase.py, scripts/extract_qwen25_validated.py and
# scripts/paraphrase_noise_floor.py, which would then fail to import.
#
# Billing. Launch inside tmux so a dropped SSH session does not kill the run; the script keeps its
# own log, so no outer tee is needed:
#     tmux new -s post
#     POD_AUTOSTOP=1 POD_DISK_GB=<volume quota> POD_UPLOAD=<rclone target> bash pod/run_post_submission.sh
#     (detach with Ctrl-b d; reattach with tmux attach -t post; log in $POD_LOG_DIR/post_run.log)
# The EXIT trap is set before anything else can fail. With POD_AUTOSTOP=1 it uploads the logs, then
# calls pod_stop (pod/pod_stop.sh, bash and runpodctl only): `runpodctl stop pod`, a verified status
# check, and, if the pod is still RUNNING (RunPod has documented stop as unavailable for pods with a
# network volume), `runpodctl remove pod` provided no output awaits upload. A detached watchdog calls
# the same function after POD_WATCHDOG_HOURS (default 4) whatever happens to this shell, and removes
# the pod even with an upload pending. Every step of both is logged to $POD_LOG_DIR/post_stop.log.
# A STOPPED POD STILL BILLS ITS VOLUME STORAGE: after retrieving the outputs, terminate it
# (runpodctl remove pod <id>) and delete any network volume no longer needed.
# A real run refuses to start without POD_AUTOSTOP=1 unless POD_KEEP_RUNNING=1.
#
# Resumable: every extraction skips a model whose npz exists, so a re-run after a dropped session
# continues. Each model is uploaded and checksummed before the next starts; weights are deleted
# after the npz is written. HF_HOME is on the local container disk (not the volume): one model at a
# time, at most 65.5 GB.
#
#   POD_UPLOAD          rclone target for the npz files and logs        REQUIRED unless DRY=1
#   POD_AUTOSTOP=1      stop (or remove) the pod when the run ends or aborts
#   POD_KEEP_RUNNING=1  run without autostop, deliberately
#   POD_DISK_GB         the volume quota in GB; REQUIRED whenever RUNPOD_POD_ID is set or the
#                       outputs are on a network volume (statvfs reports the cluster there)
#   POD_WATCHDOG_HOURS  whole hours >= 1 before the watchdog stops the pod (default 4)
#   POD_LOCAL_DIR       local container-disk directory for HF_HOME (default /root; HF_HOME=$it/hf)
#   POD_LOG_DIR         logs, pid and state files (default: post_logs next to the checkout)
#   DRY=1               preflight and plan only: no install, no download, no extraction, no stop
#   DRY_INSTALL=1       as DRY=1, but install the pinned dependencies first
#   POD_ALLOW_GPU=1     accept a GPU other than an A100 80GB (estimates below assume an A100)
#   PY                  python interpreter
#
# Estimated time on one A100 80GB. Forward passes are the laptop's measured times (RTX 5080 Laptop,
# batch 16) extrapolated linearly in parameters, with no credit for the A100 being faster, so they
# are an upper estimate:
#   step                     download  forward: floor + paraphrase    batch   step timeouts
#   P7 Qwen3-8B               16.4 GB    2.1 +  3.4 min                16      30 + 25 min
#   P7 Qwen3.5-9B             19.3 GB    3.3 +  5.8 min                16      30 + 30 min
#   P7 Qwen3-14B              29.5 GB    3.6 +  5.8 min                16      45 + 35 min
#   P7 Qwen3.5-27B            55.6 GB    9.3 + 16.0 min                16      75 + 60 min
#   P7 Qwen3-32B              65.5 GB    7.8 + 12.5 min                10      75 + 60 min
#   forward passes ~70 min; downloads 186.3 GB ~16 min at 200 MB/s; 10 weight loads ~14 min;
#   compressed writes ~16 GB ~7 min; uploads ~3 min; setup ~10 min.
#   Total about 2.0 A100-hours as an upper estimate (about 1.4 if the A100 runs the forward passes
#   twice as fast as the laptop). Disk: 65.5 GB peak on the container disk (HF cache), about 19 GB
#   of outputs on the volume; 90 GB if both share one filesystem.
set -euo pipefail

# ------------------------------------------------------- 0. billing guard, before anything can fail
# Plain bash only up to the trap: nothing above it can exit, so every later exit (the cd, the
# python lookup and version check included) runs autostop.
DRY=${DRY:-0}
DRY_INSTALL=${DRY_INSTALL:-0}
[ "$DRY_INSTALL" = "1" ] && DRY=1
SCRIPT_DIR=$(cd "$(dirname "$0")" 2>/dev/null && pwd) || SCRIPT_DIR=$(dirname "$0")
LOGDIR=${POD_LOG_DIR:-$SCRIPT_DIR/../../post_logs}
mkdir -p "$LOGDIR" 2>/dev/null || LOGDIR=/tmp
LOGDIR=$(cd "$LOGDIR" 2>/dev/null && pwd) || LOGDIR=/tmp
export POD_STOP_LOG=$LOGDIR/post_stop.log
export POD_PENDING_FLAG=$LOGDIR/post_pending_upload
export POD_OUTPUTS_ON_NETWORK=0
rm -f "$POD_PENDING_FLAG" 2>/dev/null || true

autostop() {
  local rc=$?
  trap '' PIPE
  set +e
  [ "$DRY" = "1" ] && exit "$rc"
  printf '%s [autostop] run ended with status %s\n' "$(date -u +%FT%TZ)" "$rc" >> "$POD_STOP_LOG"
  if [ -n "${POD_UPLOAD:-}" ] && command -v rclone >/dev/null 2>&1; then
    for f in post_run.log post_pip_freeze.txt post_nvidia_smi.txt post_stop.log; do
      [ -f "$LOGDIR/$f" ] && timeout 120 rclone copy "$LOGDIR/$f" "$POD_UPLOAD/logs" >> "$POD_STOP_LOG" 2>&1
    done
    printf '%s [autostop] logs uploaded to %s/logs\n' "$(date -u +%FT%TZ)" "$POD_UPLOAD" >> "$POD_STOP_LOG"
  fi
  if [ "${POD_AUTOSTOP:-0}" = "1" ]; then
    if declare -F pod_stop >/dev/null; then
      pod_stop autostop
    else
      printf '%s [autostop] pod_stop not loaded; STOP THE POD BY HAND\n' "$(date -u +%FT%TZ)" >> "$POD_STOP_LOG"
    fi
  fi
  exit "$rc"
}
trap autostop EXIT
trap 'exit 130' INT TERM HUP

. "$SCRIPT_DIR/pod_stop.sh"
exec > >(tee -a "$LOGDIR/post_run.log") 2>&1
echo "== run started $(date -u +%FT%TZ); logs in $LOGDIR"

case "${POD_WATCHDOG_HOURS:-4}" in
  ''|*[!0-9]*|0*) echo "POD_WATCHDOG_HOURS must be a whole number of hours >= 1" >&2; exit 1 ;;
esac
if [ -n "${RUNPOD_POD_ID:-}" ] && [ -z "${POD_DISK_GB:-}" ]; then
  echo "POD_DISK_GB (the volume quota in GB) is required on a pod" >&2; exit 1
fi
if [ "$DRY" != "1" ]; then
  if [ "${POD_AUTOSTOP:-0}" = "1" ]; then
    command -v runpodctl >/dev/null || { echo "POD_AUTOSTOP=1 but runpodctl is not installed" >&2; exit 1; }
    [ -n "${RUNPOD_POD_ID:-}" ] || { echo "POD_AUTOSTOP=1 but RUNPOD_POD_ID is not set" >&2; exit 1; }
  elif [ "${POD_KEEP_RUNNING:-0}" != "1" ]; then
    echo "refusing to start without POD_AUTOSTOP=1 (set POD_KEEP_RUNNING=1 to run without it)" >&2
    exit 1
  fi
fi
# The stop is only as good as the pod id: probe it while stopping is still cheap.
if [ -n "${RUNPOD_POD_ID:-}" ] && command -v runpodctl >/dev/null; then
  if timeout 60 runpodctl get pod "$RUNPOD_POD_ID" >/dev/null 2>&1; then
    echo "runpodctl sees pod $RUNPOD_POD_ID"
  else
    echo "runpodctl get pod $RUNPOD_POD_ID failed: the stop could not reach this pod" >&2
    [ "$DRY" = "1" ] || exit 1
  fi
fi
# Hard cap on billing, independent of this shell: whatever happens to the run (a hang, a killed tmux
# session, a trap that never fires), pod_stop runs POD_WATCHDOG_HOURS after the start.
if [ "$DRY" != "1" ]; then
  if [ -n "${RUNPOD_POD_ID:-}" ] && command -v runpodctl >/dev/null; then
    pidf=$LOGDIR/post_watchdog.pid
    if [ -f "$pidf" ] && old=$(cat "$pidf") && kill -0 "$old" 2>/dev/null; then
      pkill -P "$old" 2>/dev/null || true
      kill "$old" 2>/dev/null || true
      echo "stopped an older watchdog (pid $old)"
    fi
    nohup bash -c "sleep $(( ${POD_WATCHDOG_HOURS:-4} * 3600 )) && . '$SCRIPT_DIR/pod_stop.sh' && pod_stop_log watchdog '${POD_WATCHDOG_HOURS:-4} h elapsed' && pod_stop watchdog force" \
      >> "$LOGDIR/post_watchdog.log" 2>&1 < /dev/null &
    wd=$!
    disown "$wd" 2>/dev/null || true
    echo "$wd" > "$pidf"
    sleep 1
    kill -0 "$wd" 2>/dev/null || { echo "the watchdog did not start" >&2; exit 1; }
    echo "watchdog pid $wd stops pod $RUNPOD_POD_ID in ${POD_WATCHDOG_HOURS:-4} h"
  else
    echo "WARNING no watchdog: RUNPOD_POD_ID or runpodctl missing; stop the pod by hand" >&2
  fi
fi

# --------------------------------------------------------------------------- 1. where we are
cd "$SCRIPT_DIR/.."
ROOT=$(pwd)
PY=${PY:-$(command -v python3 || command -v python)}
[ -x "$PY" ] || { echo "no python found; set PY=/path/to/python" >&2; exit 1; }
say() { printf '\n\033[1m== %s\033[0m\n' "$*"; }
# The result files hash their producers with ast.dump, which changes between Python minor versions,
# and the repository is written for 3.11 (README); older interpreters are refused, DRY=1 included.
"$PY" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)' \
  || { echo "python >= 3.11 required, have $("$PY" -V 2>&1)" >&2; exit 1; }

COMMIT=$(git rev-parse HEAD)
DIRTY=$(git status --porcelain --untracked-files=no)
echo "commit $COMMIT"
if [ -n "$DIRTY" ]; then
  echo "modified tracked files:" >&2; echo "$DIRTY" >&2
  [ "$DRY" = "1" ] || { echo "refusing to extract from a modified tree; commit or reset first" >&2; exit 1; }
fi
git status --porcelain | sed 's/^/  status: /'

export HF_HOME=${HF_HOME:-${POD_LOCAL_DIR:-/root}/hf}
export EAS_ACTIVATION_ROOT=${EAS_ACTIVATION_ROOT:-$ROOT}
export HF_HUB_DISABLE_PROGRESS_BARS=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
unset POD_VRAM_GB                  # batches are passed explicitly below, as the paper's pod rule gives
P7_MODELS=${P7_MODELS:-"qwen3-8b qwen3.5-9b qwen3-14b qwen3.5-27b qwen3-32b"}   # ascending: fail cheap
P3_MODELS=""                       # see the header: the Qwen2.5 four-rung answer exists already
PROMPT_SHA12=39980ca8f0b8          # the laptop rendering of the 36 x 60 framing prompts
batch_of() {      # the paper's pod rule on an 80 GB card; the floor extraction uses the same
  case "$1" in qwen3-32b) echo 10 ;; *) echo 16 ;; esac
}
timeouts_of() {   # minutes: floor step (includes the download), paraphrase step; ~3x the estimate
  case "$1" in
    qwen3-8b) echo "30 25" ;; qwen3.5-9b) echo "30 30" ;; qwen3-14b) echo "45 35" ;;
    qwen3.5-27b|qwen3-32b) echo "75 60" ;; *) echo "60 60" ;;
  esac
}

# ------------------------------------------------------------------------------ 2. preflight
say "preflight"
mkdir -p "$HF_HOME" "$EAS_ACTIVATION_ROOT"
# statvfs on a network volume reports the cluster, not the quota (pod/extract.py free_gb), so a
# free-space check there means nothing unless POD_DISK_GB declares the quota. Local filesystems are
# listed; anything else is treated as a network volume.
for path in "$EAS_ACTIVATION_ROOT" "$HF_HOME"; do
  fstype=$(stat -f -c %T "$path" 2>/dev/null || echo unknown)
  case "$fstype" in
    ext2/ext3|ext4|xfs|btrfs|overlayfs|overlay|tmpfs|zfs|f2fs) echo "  $path on local $fstype" ;;
    *) echo "  $path on $fstype: treated as a network volume"
       [ "$path" = "$EAS_ACTIVATION_ROOT" ] && export POD_OUTPUTS_ON_NETWORK=1
       if [ -z "${POD_DISK_GB:-}" ]; then
         echo "  POD_DISK_GB is required when $path is on a network volume (the volume quota in GB)" >&2
         exit 1
       fi ;;
  esac
done
command -v nvidia-smi >/dev/null && nvidia-smi -q > "$LOGDIR/post_nvidia_smi.txt" 2>&1 || true
"$PY" - <<'PY'
import os, subprocess, sys
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
# The run's disk peak, from the plan: the HF cache holds one model at a time (Qwen3-32B, 65.5 GB,
# deleted after its npz is written) and the outputs total about 19 GB. The brief's earlier blanket
# 150 GB is replaced by these, with a margin: 70 GB for the cache, 25 GB for the outputs, 90 GB
# when both are on one filesystem.
hf, out = os.environ["HF_HOME"], os.environ["EAS_ACTIVATION_ROOT"]
f_hf, f_out = free_gb(hf), free_gb(out)
quota = " (POD_DISK_GB quota)" if os.environ.get("POD_DISK_GB") else ""
print(f"  free: HF_HOME {hf} {f_hf:.0f} GB, outputs {out} {f_out:.0f} GB{quota}")
need = [(hf, f_hf, 70), (out, f_out, 25)]
if os.stat(hf).st_dev == os.stat(out).st_dev:
    need = [(out, min(f_hf, f_out), 90)]
for p, f, n in need:
    if f < n:
        sys.exit(f"  {f:.0f} GB free at {p}, the run needs about {n} GB there")
PY

# ------------------------------------------------------------------- 3. dependencies, pinned
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
# scipy and scikit-learn are imported by the floor extraction (gstudy_gtheory -> gstudy_analyze)
for p in ("huggingface_hub", "accelerate", "scipy", "scikit-learn"):
    try:
        md.version(p)
    except md.PackageNotFoundError:
        bad.append(f"{p} missing")
print("  " + ("; ".join(bad) if bad else "all pins satisfied"))
sys.exit(1 if bad else 0)
PY
then :
elif [ "$DRY" = "1" ] && [ "$DRY_INSTALL" != "1" ]; then
  echo "  DRY=1: not installing anything (DRY_INSTALL=1 installs, then runs the same checks)"
else
  # one resolver call, so pip picks scipy and scikit-learn builds that declare numpy 2.4.4 supported
  "$PY" -m pip install --quiet "torch==2.11.0" "transformers==5.8.0" "numpy==2.4.4" \
      huggingface_hub accelerate scipy scikit-learn
fi
if { [ "$DRY" != "1" ] || [ "$DRY_INSTALL" = "1" ]; } && ! "$PY" -c "import hf_transfer" 2>/dev/null; then
  "$PY" -m pip install --quiet hf_transfer || true
fi
"$PY" -c "import hf_transfer" 2>/dev/null && export HF_HUB_ENABLE_HF_TRANSFER=1 || true
"$PY" -m pip freeze > "$LOGDIR/post_pip_freeze.txt" 2>&1 || true
# Import and device check of everything the extraction loads, binary compatibility included: a scipy
# built against another numpy, or a torch that cannot reach the GPU, fails here, before any download.
if ! "$PY" - <<'PY'
import subprocess, sys
sys.path.insert(0, "scripts"); sys.path.insert(0, "src")
import numpy, scipy, scipy.stats, sklearn.metrics                      # noqa: F401
import torch, accelerate                                                # noqa: F401
from transformers import AutoModelForCausalLM                           # noqa: F401
import transformers
import extract_paraphrase, paraphrase_noise_floor                       # noqa: F401
print(f"  imports ok: numpy {numpy.__version__}, scipy {scipy.__version__}, scikit-learn "
      f"{sklearn.__version__}, torch {torch.__version__}, transformers {transformers.__version__}, "
      f"accelerate {accelerate.__version__}")
drv = subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                     capture_output=True, text=True).stdout.strip()
print(f"  torch CUDA {torch.version.cuda}, driver {drv or 'unknown'}")
assert torch.cuda.is_available(), "torch cannot see a CUDA device"
a = torch.randn(64, 64, device="cuda", dtype=torch.bfloat16)
assert torch.isfinite((a @ a).float()).all(), "CUDA matmul returned non-finite values"
torch.cuda.synchronize()
print(f"  CUDA matmul ok on {torch.cuda.get_device_name(0)}")
PY
then
  echo "  import or device check failed (output above)" >&2
  [ "$DRY" = "1" ] || exit 1
fi
# The floor extraction derives its batch from the same rule; refuse a disagreement.
for m in $P7_MODELS; do
  fb=$("$PY" -c "import sys; sys.path[:0] = ['scripts', 'src']; import paraphrase_noise_floor as nf; print(nf.pod_batch('$m'))" 2>/dev/null || echo "?")
  [ "$fb" = "$(batch_of "$m")" ] || { echo "  batch mismatch for $m: floor $fb, paraphrase $(batch_of "$m")" >&2; [ "$DRY" = "1" ] || exit 1; }
done

# --------------------------------------------- 4. the prompts are the laptop's prompts
say "rendering check (tokenizers only)"
for m in $P7_MODELS; do
  if ! out=$("$PY" scripts/extract_paraphrase.py --render-only --model "$m" 2>&1); then
    echo "$out" >&2
    echo "  render check failed for $m (output above); stopping" >&2
    exit 1
  fi
  line=$(printf '%s\n' "$out" | grep -E "^$m: " | tail -1 || true)
  echo "  $line"
  case "$line" in
    *"sha $PROMPT_SHA12"*) ;;
    *) echo "$out" >&2
       echo "  $m renders differently from the laptop (want sha $PROMPT_SHA12); stopping" >&2
       exit 1 ;;
  esac
done

# ------------------------------------------------- 5. the exit route, before any GPU work
if [ "$DRY" != "1" ]; then
  say "verifying the upload target"
  : "${POD_UPLOAD:?set POD_UPLOAD to an rclone target, or DRY=1}"
  command -v rclone >/dev/null || { curl -fsSL https://rclone.org/install.sh | bash; }
  probe=$(mktemp -d)/upload-probe.txt
  date -u +%FT%TZ > "$probe"
  rclone copy --checksum "$probe" "$POD_UPLOAD" || { echo "cannot write to $POD_UPLOAD" >&2; exit 1; }
  rclone lsf "$POD_UPLOAD" --include upload-probe.txt | grep -q . \
    || { echo "cannot read back $POD_UPLOAD" >&2; exit 1; }
  rclone deletefile "$POD_UPLOAD/upload-probe.txt" || true
fi

ACTS_P7=$EAS_ACTIVATION_ROOT/results/post_submission/paraphrase/acts
ACTS_P7_SMALL=$EAS_ACTIVATION_ROOT/results/post_submission/paraphrase/acts_validated_only
ACTS_P3=$EAS_ACTIVATION_ROOT/results/post_submission/qwen25_validated/acts
FLOOR=$EAS_ACTIVATION_ROOT/results/post_submission/paraphrase/floor
say "plan"
for m in $P3_MODELS; do
  [ -f "$ACTS_P3/$m.npz" ] && echo "  P3 $m  done" || echo "  P3 $m  to extract"
done
for m in $P7_MODELS; do
  read -r tf tp <<< "$(timeouts_of "$m")"
  [ -f "$ACTS_P7/${m}__nothink.npz" ] && echo "  P7 $m  done" \
    || echo "  P7 $m  to extract, batch $(batch_of "$m"), timeouts $tf + $tp min"
done
if [ "$DRY" = "1" ]; then say "DRY=1: stopping before the GPU"; exit 0; fi

upload() {   # $1 file, $2 subdirectory of the target
  rclone copy --checksum "$1" "$POD_UPLOAD/$2" || { echo "upload failed for $1" >&2; exit 1; }
  echo "  uploaded $(basename "$1") -> $POD_UPLOAD/$2"
}

# ------------------------------------------------------------------------- 6. extraction
T0=$(date +%s)
for m in $P3_MODELS; do
  say "P3 $m"
  echo "P3 $m" > "$POD_PENDING_FLAG"
  timeout 60m "$PY" scripts/extract_qwen25_validated.py --model "$m" --delete-weights
  upload "$ACTS_P3/$m.npz" qwen25_validated
  rm -f "$POD_PENDING_FLAG"
done
for m in $P7_MODELS; do
  say "P7 $m"
  read -r tf tp <<< "$(timeouts_of "$m")"
  echo "P7 $m" > "$POD_PENDING_FLAG"      # from here an output may exist that is not uploaded
  timeout "${tf}m" "$PY" scripts/paraphrase_noise_floor.py extract --model "$m"
  upload "$FLOOR/${m}__nothink.npz" paraphrase/floor
  timeout "${tp}m" "$PY" scripts/extract_paraphrase.py --model "$m" --batch "$(batch_of "$m")" --delete-weights
  f=$ACTS_P7/${m}__nothink.npz
  # A validated-only copy: what scripts/paraphrase_analysis.py reads, at half the download.
  mkdir -p "$ACTS_P7_SMALL"
  [ -f "$ACTS_P7_SMALL/$(basename "$f")" ] || timeout 20m "$PY" - "$f" "$ACTS_P7_SMALL/$(basename "$f")" <<'PY'
import sys, numpy as np
with np.load(sys.argv[1], allow_pickle=False) as z:
    keep = {k: z[k] for k in z.files if k != "framing_alltokens"}
tmp = sys.argv[2] + ".tmp.npz"
np.savez_compressed(tmp, **keep)
import os; os.replace(tmp, sys.argv[2])
PY
  upload "$f" paraphrase/acts
  upload "$ACTS_P7_SMALL/$(basename "$f")" paraphrase/acts_validated_only
  rm -f "$POD_PENDING_FLAG"                 # everything of this model is uploaded
  echo "  elapsed $(( ($(date +%s) - T0) / 60 )) min"
done

say "done in $(( ($(date +%s) - T0) / 60 )) min"
cat <<EOF
On the analysis machine (which holds results/pod_qwen3, results/pod_qwen35 and results/gstudy):
  copy paraphrase/acts_validated_only/*.npz (or paraphrase/acts/*.npz) into
       \$EAS_ACTIVATION_ROOT/results/post_submission/paraphrase/acts/
  copy paraphrase/floor/*.npz into \$EAS_ACTIVATION_ROOT/results/post_submission/paraphrase/floor/
  python scripts/paraphrase_noise_floor.py summary      # amendment-2 floors, all rungs
  python scripts/paraphrase_analysis.py                 # gated: records the registered stop
  python scripts/paraphrase_analysis.py --ungated       # descriptive, P7c now that >= 4 rungs exist
Extraction commit: $COMMIT (also in every npz's meta). Logs: $POD_UPLOAD/logs.
Then terminate the pod (a stopped pod still bills its volume): runpodctl remove pod <id>.
EOF
