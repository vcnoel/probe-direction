# Stop (or, failing that, remove) the RunPod pod this run is billed on. Sourced by
# pod/run_post_submission.sh for its EXIT trap and by its detached watchdog: one function, both
# paths. Bash and runpodctl only, no python, so it works whatever state the python environment is in.
# Nothing here depends on the log being writable: every log write ignores its own failure.
#
# RunPod's documentation has disagreed on whether a pod with a network volume can be stopped (older
# pages: terminate only; later pages: stop allowed). So the stop is verified, and if the pod is not
# confirmed stopped afterwards it is removed (terminated) -- but only when nothing would be lost: no
# output awaits upload (the flag file POD_PENDING_FLAG is absent), or the outputs sit on a network
# volume, which outlives the pod, or the caller forces it (the watchdog's hard cap).
#
# Verification reads `runpodctl get pod <id>`, which reports the pod's DesiredStatus. The legacy
# command filters the id on the client: an absent pod, or a key scoped away from it, prints only the
# header and still exits 0. So a status counts only from a row carrying the pod id; a missing row is
# "unknown", never "stopped".
#
# A stopped pod still bills its volume storage. Retrieve the outputs, then terminate the pod
# (runpodctl remove pod <id>) and delete any network volume that is no longer needed.
#
# Environment (all optional except RUNPOD_POD_ID):
#   RUNPOD_POD_ID          the pod to stop
#   POD_STOP_LOG           log file, every step is appended (default /tmp/pod_stop.log)
#   POD_PENDING_FLAG       file present while an output exists that is not yet uploaded
#   POD_OUTPUTS_ON_NETWORK 1 if the outputs live on a network volume (survive pod removal)
#   POD_STOP_TIMEOUT       seconds for runpodctl stop / remove        (default 120)
#   POD_GET_TIMEOUT        seconds for each runpodctl get             (default 60)
#   POD_VERIFY_TRIES       status checks after the stop               (default 6)
#   POD_VERIFY_SLEEP       seconds between status checks              (default 20; 6 x 20 s = 2 min)

pod_stop_log() {
  { printf '%s [%s] %s\n' "$(date -u +%FT%TZ)" "$1" "$2" >> "${POD_STOP_LOG:-/tmp/pod_stop.log}"; } 2>/dev/null || true
}

pod_still_running() {   # 0 = still RUNNING or unknown (not verified), 1 = confirmed not RUNNING
  local tag=$1 out rc row i
  pod_stop_log "$tag" "verifying with runpodctl get pod (it reports DesiredStatus; only a row with the pod id counts)"
  for i in $(seq 1 "${POD_VERIFY_TRIES:-6}"); do
    out=$(timeout -k 10 "${POD_GET_TIMEOUT:-60}" runpodctl get pod "$RUNPOD_POD_ID" 2>&1); rc=$?
    row=$(printf '%s\n' "$out" | grep -F -- "$RUNPOD_POD_ID")
    if [ "$rc" -ne 0 ]; then
      pod_stop_log "$tag" "check $i: get pod rc=$rc, status unknown: $(printf '%s' "$out" | tr '\n' ' ' | cut -c1-200)"
    elif [ -z "$row" ]; then
      pod_stop_log "$tag" "check $i: get pod printed no row for $RUNPOD_POD_ID (absent pod or scoped key), status unknown: $(printf '%s' "$out" | tr '\n' ' ' | cut -c1-200)"
    elif printf '%s' "$row" | grep -q RUNNING; then
      pod_stop_log "$tag" "check $i: DesiredStatus RUNNING: $(printf '%s' "$row" | cut -c1-200)"
    else
      pod_stop_log "$tag" "check $i: row no longer RUNNING: $(printf '%s' "$row" | cut -c1-200)"
      return 1
    fi
    [ "$i" -lt "${POD_VERIFY_TRIES:-6}" ] && sleep "${POD_VERIFY_SLEEP:-20}"
  done
  return 0
}

pod_stop() {   # $1 caller tag (autostop | watchdog), $2 "force" to allow removal with pending outputs
  local tag=${1:-stop} force=${2:-} out rc
  set +e
  trap '' PIPE
  if [ -z "${RUNPOD_POD_ID:-}" ] || ! command -v runpodctl >/dev/null 2>&1; then
    pod_stop_log "$tag" "cannot stop: RUNPOD_POD_ID or runpodctl missing; STOP THE POD BY HAND"
    return 1
  fi
  pod_stop_log "$tag" "runpodctl stop pod $RUNPOD_POD_ID (timeout ${POD_STOP_TIMEOUT:-120} s)"
  out=$(timeout -k 10 "${POD_STOP_TIMEOUT:-120}" runpodctl stop pod "$RUNPOD_POD_ID" 2>&1); rc=$?
  [ "$rc" -eq 124 ] || [ "$rc" -eq 137 ] && pod_stop_log "$tag" "stop timed out"
  pod_stop_log "$tag" "stop rc=$rc: $(printf '%s' "$out" | tr '\n' ' ' | cut -c1-300)"
  if ! pod_still_running "$tag"; then
    pod_stop_log "$tag" "verified: pod is no longer RUNNING"
    return 0
  fi
  pod_stop_log "$tag" "not verified stopped (stop failed, pod still RUNNING, or status unknown)"
  local safe=0 why=""
  if [ -z "${POD_PENDING_FLAG:-}" ] || [ ! -e "$POD_PENDING_FLAG" ]; then
    safe=1; why="no output awaits upload"
  elif [ "${POD_OUTPUTS_ON_NETWORK:-0}" = "1" ]; then
    safe=1; why="outputs are on a network volume, which outlives the pod"
  elif [ "$force" = "force" ]; then
    safe=1; why="forced by the $tag; the output listed in $POD_PENDING_FLAG may be lost"
  fi
  if [ "$safe" != "1" ]; then
    pod_stop_log "$tag" "NOT removing: an output awaits upload ($(cat "$POD_PENDING_FLAG" 2>/dev/null)); STOP OR REMOVE THE POD BY HAND"
    return 1
  fi
  pod_stop_log "$tag" "removing pod $RUNPOD_POD_ID ($why)"
  out=$(timeout -k 10 "${POD_STOP_TIMEOUT:-120}" runpodctl remove pod "$RUNPOD_POD_ID" 2>&1); rc=$?
  pod_stop_log "$tag" "remove rc=$rc: $(printf '%s' "$out" | tr '\n' ' ' | cut -c1-300)"
  if [ "$rc" -ne 0 ]; then
    pod_stop_log "$tag" "remove FAILED; STOP OR REMOVE THE POD BY HAND"
    return 1
  fi
  return 0
}
