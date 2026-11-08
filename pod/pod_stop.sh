# Stop the RunPod pod this run is billed on, and if it cannot be stopped, remove it without losing
# outputs. Sourced by pod/run_post_submission.sh for its preflight, its EXIT trap and its detached
# watchdog: one stop path for both. Bash, coreutils, rclone and runpodctl only, no python, so it
# works whatever state the python environment is in. No log write can stop it: every write ignores
# its own failure, and command output goes to the log only when the log is writable.
#
#   pod_verify_outputs_safe   preflight, before any GPU step: outputs are safe only if the upload
#                             target passes a round trip (a small file uploaded and read back with
#                             the same content), or NETWORK_VOLUME=1 and the mount table shows the
#                             output root on a network filesystem mounted at /workspace (the flag
#                             alone is not trusted). The verified state goes to POD_STATE_FILE.
#   pod_stop TAG              writes the output manifest (every completed npz with its size and
#                             sha256, and the step in progress), uploads it, then
#                             `runpodctl stop pod`. If the stop is not verified (refused -- RunPod has
#                             documented stop as unavailable for pods with a network volume --, hung,
#                             or status unknown), it escalates once, never retrying the stop: upload
#                             the outputs (or rely on the verified network volume), `runpodctl remove
#                             pod`, and verify absence: no row of `runpodctl get pod` carries the id.
#
# `runpodctl get pod` reports DesiredStatus and, in its legacy form, filters the id on the client:
# an absent pod prints only the header and exits 0. So a status is read only from a row carrying the
# pod id; before removal a missing row means "unknown", after removal it is the proof of absence
# (the preflight required a RUNNING row, so the key can see the pod).
#
# A stopped pod still bills its volume storage. Retrieve the outputs, then terminate the pod
# (runpodctl remove pod <id>) and delete any network volume that is no longer needed.
#
# Environment (the launcher sets these; POD_STATE_FILE overrides them with the verified state):
#   RUNPOD_POD_ID, POD_UPLOAD (rclone target), POD_OUTPUT_ROOT (results/post_submission)
#   POD_STOP_LOG (default /tmp/pod_stop.log), POD_STATE_FILE, POD_PENDING_FLAG, POD_COMMIT
#   POD_UPLOAD_PAIRS   "local=remote" subdirectory pairs under POD_OUTPUT_ROOT and POD_UPLOAD
#   POD_STOP_TIMEOUT (120 s), POD_GET_TIMEOUT (60 s), POD_VERIFY_TRIES (6), POD_VERIFY_SLEEP (20 s),
#   POD_UPLOAD_TIMEOUT (1800 s per directory), POD_MANIFEST_TIMEOUT (300 s), POD_MOUNTS_FILE (/proc/mounts),
#   POD_NETWORK_MOUNT (/workspace)

pod_stop_log() {
  { printf '%s [%s] %s\n' "$(date -u +%FT%TZ)" "$1" "$2" >> "${POD_STOP_LOG:-/tmp/pod_stop.log}"; } 2>/dev/null || true
}

pod_sink() {   # where command output may go: the log if writable, else nowhere
  if { : >> "${POD_STOP_LOG:-/tmp/pod_stop.log}"; } 2>/dev/null; then
    printf '%s' "${POD_STOP_LOG:-/tmp/pod_stop.log}"
  else
    printf '/dev/null'
  fi
}

pod_state() {   # the state the preflight verified (output root, upload target, network volume)
  if [ -n "${POD_STATE_FILE:-}" ] && [ -f "$POD_STATE_FILE" ]; then
    . "$POD_STATE_FILE" 2>/dev/null || true
  fi
}

pod_upload_pairs() {
  printf '%s\n' ${POD_UPLOAD_PAIRS:-paraphrase/acts=paraphrase/acts paraphrase/acts_validated_only=paraphrase/acts_validated_only paraphrase/floor=paraphrase/floor qwen25_validated/acts=qwen25_validated behaviour=behaviour}
}

pod_mount_of() {   # $1 path -> "mountpoint fstype" of the longest mount containing it
  local p
  p=$(realpath -m "$1" 2>/dev/null || printf '%s' "$1")
  awk -v p="$p" '{ mp = $2; gsub(/\\040/, " ", mp)
                   if ((p == mp || index(p, (mp == "/" ? "/" : mp "/")) == 1) && length(mp) > best) {
                     best = length(mp); out = mp " " $3 } }
                 END { if (out != "") print out }' "${POD_MOUNTS_FILE:-/proc/mounts}" 2>/dev/null
}

pod_verify_outputs_safe() {   # 0 if the outputs are safe; writes POD_STATE_FILE
  local net=0 up=0 m mp fs probe got
  if [ "${NETWORK_VOLUME:-0}" = "1" ]; then
    m=$(pod_mount_of "${POD_OUTPUT_ROOT:-/nonexistent}")
    mp=${m% *}; fs=${m##* }
    case "$fs" in
      nfs*|cifs|smb*|ceph*|fuse|fuse.*|9p|lustre|glusterfs*|mfs*)
        if [ "$mp" = "${POD_NETWORK_MOUNT:-/workspace}" ]; then
          net=1; echo "  outputs on a verified network volume: $mp ($fs)"
        else
          echo "  NETWORK_VOLUME=1, but the output root's mount is $mp ($fs), not ${POD_NETWORK_MOUNT:-/workspace}"
        fi ;;
      *) echo "  NETWORK_VOLUME=1, but the output root is on ${mp:-?} (${fs:-unknown}), not a network filesystem" ;;
    esac
  fi
  if [ -n "${POD_UPLOAD:-}" ]; then
    if command -v rclone >/dev/null 2>&1; then
      probe=$(mktemp -d 2>/dev/null || echo /tmp)/upload-probe.txt
      printf 'probe %s %s %s\n' "$(date -u +%FT%TZ)" "${RUNPOD_POD_ID:-local}" "$RANDOM$RANDOM" > "$probe"
      if timeout -k 10 300 rclone copy "$probe" "$POD_UPLOAD/.probe" >/dev/null 2>&1 \
         && got=$(timeout -k 10 120 rclone cat "$POD_UPLOAD/.probe/upload-probe.txt" 2>/dev/null) \
         && [ "$got" = "$(cat "$probe")" ]; then
        up=1; echo "  upload target verified by a round trip: $POD_UPLOAD"
      else
        echo "  upload target $POD_UPLOAD failed the round trip (copy, read back, compare)"
      fi
      timeout -k 10 120 rclone deletefile "$POD_UPLOAD/.probe/upload-probe.txt" >/dev/null 2>&1 || true
    else
      echo "  POD_UPLOAD is set but rclone is not installed"
    fi
  fi
  {
    printf 'POD_OUTPUTS_ON_NETWORK=%s\nPOD_UPLOAD_VERIFIED=%s\n' "$net" "$up"
    printf 'POD_UPLOAD=%q\nPOD_OUTPUT_ROOT=%q\nPOD_COMMIT=%q\n' "${POD_UPLOAD:-}" "${POD_OUTPUT_ROOT:-}" "${POD_COMMIT:-}"
  } > "${POD_STATE_FILE:-/tmp/post_state.env}" 2>/dev/null || true
  export POD_OUTPUTS_ON_NETWORK=$net POD_UPLOAD_VERIFIED=$up
  [ "$net" = 1 ] || [ "$up" = 1 ]
}

POD_STOP_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" 2>/dev/null && pwd)

pod_write_manifest() {   # $1 tag: every completed output with bytes and sha256, then upload it
  # Built by pod_manifest.sh under a timeout, on the local log disk first, so neither a slow hash nor
  # a hung output filesystem can delay the stop; a timeout leaves a partial manifest marked as such.
  local tag=$1 rc n dir man
  dir=$(dirname "${POD_STOP_LOG:-/tmp/pod_stop.log}")
  man=$dir/post_output_manifest.tsv
  if [ -z "${POD_OUTPUT_ROOT:-}" ]; then
    pod_stop_log "$tag" "no output root yet; no manifest"
    return 0
  fi
  timeout -k 5 "${POD_MANIFEST_TIMEOUT:-300}" bash "$POD_STOP_DIR/pod_manifest.sh" "$tag" "$man.tmp" \
    >/dev/null 2>&1
  rc=$?
  if [ "$rc" -ne 0 ]; then
    { printf '# TRUNCATED: the manifest builder ended with status %s (timeout %s s)\n' "$rc" \
        "${POD_MANIFEST_TIMEOUT:-300}" >> "$man.tmp"; } 2>/dev/null || true
  fi
  mv -f "$man.tmp" "$man" 2>/dev/null || true
  n=$(grep -cv '^#\|^path' "$man" 2>/dev/null || true)
  pod_stop_log "$tag" "manifest $man: ${n:-0} completed output files (builder status $rc)"
  timeout -k 5 30 cp -f "$man" "$POD_OUTPUT_ROOT/" >/dev/null 2>&1 \
    || pod_stop_log "$tag" "could not copy the manifest beside the outputs"
  if [ -n "${POD_UPLOAD:-}" ] && command -v rclone >/dev/null 2>&1; then
    if timeout -k 10 120 rclone copy "$man" "$POD_UPLOAD/manifest" >> "$(pod_sink)" 2>&1; then
      pod_stop_log "$tag" "manifest uploaded to $POD_UPLOAD/manifest"
    else
      pod_stop_log "$tag" "manifest upload failed"
    fi
  fi
}

pod_upload_outputs() {   # $1 tag -> 0 when every output directory is uploaded
  local tag=$1 pair d rc bad=0 n=0
  if [ -z "${POD_UPLOAD:-}" ] || ! command -v rclone >/dev/null 2>&1; then
    pod_stop_log "$tag" "no upload target or no rclone"
    return 1
  fi
  for pair in $(pod_upload_pairs); do
    d=${POD_OUTPUT_ROOT:-/nonexistent}/${pair%%=*}
    timeout -k 2 10 test -d "$d" || continue          # a hung filesystem must not hang the stop
    timeout -k 30 "${POD_UPLOAD_TIMEOUT:-1800}" rclone copy --checksum --exclude '*.tmp.npz' \
      "$d" "$POD_UPLOAD/${pair#*=}" >> "$(pod_sink)" 2>&1
    rc=$?
    pod_stop_log "$tag" "upload ${pair%%=*} -> $POD_UPLOAD/${pair#*=}: rc=$rc"
    [ "$rc" -eq 0 ] || bad=1
    n=$((n + 1))
  done
  [ "$n" -gt 0 ] || pod_stop_log "$tag" "no output directory exists yet: nothing to upload"
  return "$bad"
}

pod_get() {   # sets POD_GET_RC, POD_GET_OUT and POD_ROW (the row carrying the id); no subshell
  local out
  out=$(timeout -k 10 "${POD_GET_TIMEOUT:-60}" runpodctl get pod "$RUNPOD_POD_ID" 2>&1)
  POD_GET_RC=$?
  POD_GET_OUT=$(printf '%s' "$out" | tr '\n' ' ' | cut -c1-200)
  POD_ROW=$(printf '%s\n' "$out" | grep -F -- "$RUNPOD_POD_ID")
}

pod_still_running() {   # 0 = RUNNING or unknown (not verified), 1 = confirmed not RUNNING
  local tag=$1 row i
  pod_stop_log "$tag" "verifying the stop with runpodctl get pod (DesiredStatus; only a row with the id counts)"
  for i in $(seq 1 "${POD_VERIFY_TRIES:-6}"); do
    pod_get; row=$POD_ROW
    if [ "$POD_GET_RC" -ne 0 ]; then
      pod_stop_log "$tag" "check $i: get pod rc=$POD_GET_RC, status unknown: $POD_GET_OUT"
    elif [ -z "$row" ]; then
      pod_stop_log "$tag" "check $i: no row for $RUNPOD_POD_ID, status unknown: $POD_GET_OUT"
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

pod_absent() {   # 0 = verified absent: get pod succeeds and no row carries the id
  local tag=$1 row i
  for i in $(seq 1 "${POD_VERIFY_TRIES:-6}"); do
    pod_get; row=$POD_ROW
    if [ "$POD_GET_RC" -eq 0 ] && [ -z "$row" ]; then
      pod_stop_log "$tag" "absence check $i: no row carries $RUNPOD_POD_ID: removal verified"
      return 0
    fi
    pod_stop_log "$tag" "absence check $i: rc=$POD_GET_RC, row: ${row:-none}"
    [ "$i" -lt "${POD_VERIFY_TRIES:-6}" ] && sleep "${POD_VERIFY_SLEEP:-20}"
  done
  return 1
}

pod_remove_path() {   # $1 tag: secure the outputs, remove the pod, verify absence
  local tag=$1 out rc attempt
  if [ "${POD_OUTPUTS_ON_NETWORK:-0}" = "1" ]; then
    pod_stop_log "$tag" "outputs are on the verified network volume, which outlives the pod: no upload needed"
  elif pod_upload_outputs "$tag"; then
    pod_stop_log "$tag" "every output directory uploaded"
  else
    pod_stop_log "$tag" "UPLOAD FAILED: the outputs in the manifest are not confirmed off the pod; removing anyway, since keeping the pod only adds cost (the watchdog would remove it later)"
  fi
  for attempt in 1 2; do
    pod_stop_log "$tag" "runpodctl remove pod $RUNPOD_POD_ID (attempt $attempt)"
    out=$(timeout -k 10 "${POD_STOP_TIMEOUT:-120}" runpodctl remove pod "$RUNPOD_POD_ID" 2>&1); rc=$?
    pod_stop_log "$tag" "remove rc=$rc: $(printf '%s' "$out" | tr '\n' ' ' | cut -c1-300)"
    if pod_absent "$tag"; then
      return 0
    fi
  done
  pod_stop_log "$tag" "REMOVAL NOT VERIFIED: a row still carries $RUNPOD_POD_ID; REMOVE THE POD BY HAND"
  return 1
}

pod_stop() {   # $1 caller tag (autostop | watchdog); a second argument is accepted and ignored
  local tag=${1:-stop} out rc
  set +e
  trap '' PIPE
  pod_state
  if [ -z "${RUNPOD_POD_ID:-}" ] || ! command -v runpodctl >/dev/null 2>&1; then
    pod_stop_log "$tag" "cannot stop: RUNPOD_POD_ID or runpodctl missing; STOP THE POD BY HAND"
    return 1
  fi
  pod_write_manifest "$tag"
  pod_stop_log "$tag" "runpodctl stop pod $RUNPOD_POD_ID (timeout ${POD_STOP_TIMEOUT:-120} s)"
  out=$(timeout -k 10 "${POD_STOP_TIMEOUT:-120}" runpodctl stop pod "$RUNPOD_POD_ID" 2>&1); rc=$?
  { [ "$rc" -eq 124 ] || [ "$rc" -eq 137 ]; } && pod_stop_log "$tag" "stop timed out"
  pod_stop_log "$tag" "stop rc=$rc: $(printf '%s' "$out" | tr '\n' ' ' | cut -c1-300)"
  if ! pod_still_running "$tag"; then
    pod_stop_log "$tag" "verified: pod is no longer RUNNING"
    return 0
  fi
  pod_stop_log "$tag" "stop not verified (refused, hung or status unknown): escalating to the remove path, no stop retry"
  pod_remove_path "$tag"
}
