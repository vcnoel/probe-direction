# Stop the RunPod pod this run is billed on, and if it cannot be stopped, remove it without losing
# outputs. Sourced by pod/run_post_submission.sh for its preflight, its EXIT trap and its detached
# watchdog: one stop path for both. Bash, coreutils, rclone and runpodctl; python (standard library,
# isolated mode) only to parse a JSON status, with a bash fallback, so the stop works whatever state
# the run's python environment is in. No log write can stop it: every write ignores its own failure,
# and command output goes to the log only when the log is writable.
#
#   pod_import_env            RUNPOD_POD_ID and RUNPOD_API_KEY from /proc/1/environ when the shell
#                             has none (an SSH session on a pod carries no RUNPOD_* variables);
#                             values already exported win, so an account key exported as
#                             RUNPOD_API_KEY replaces the pod-injected one.
#   pod_status                the pod's desiredStatus, read in this order until one source answers:
#                             `runpodctl pod get` (current runpodctl: JSON), `runpodctl get pod`
#                             (legacy: a table, or JSON), then the REST API (GET
#                             rest.runpod.io/v1/pods/ID with RUNPOD_API_KEY). Parsed by
#                             pod/pod_status.py: RUNNING, EXITED, TERMINATED, ABSENT (the source
#                             answered and nothing carries the id), MULTIPLE or UNKNOWN. Any output
#                             starting with "Error" is UNKNOWN whatever the exit status (runpodctl
#                             1.14 prints "Error: Unauthorized" and exits 0).
#   pod_verify_outputs_safe   preflight, before any GPU step: outputs are safe only if the upload
#                             target passes a round trip (a small file uploaded and read back with
#                             the same content), or NETWORK_VOLUME=1 and the mount table shows the
#                             output root on a network filesystem mounted at /workspace (the flag
#                             alone is not trusted; RunPod's network volume is fuse with source
#                             mfs#..., or fuse.mfs). The verified state goes to POD_STATE_FILE.
#   pod_stop TAG              under a lock in /tmp (waited on for at most POD_STOP_LOCK_WAIT, 600 s):
#                             the watchdog first sends TERM to the launcher and its running step and
#                             waits briefly; then the output manifest (every completed output with
#                             bytes and sha256, and the step in progress) is written and uploaded;
#                             if a step was in progress and the outputs are not on a network volume,
#                             the outputs are uploaded and every manifest line confirmed before the
#                             stop; then one stop. If the stop is not verified (refused -- RunPod
#                             has documented stop as unavailable for pods with a network volume --,
#                             hung, or status unknown), it escalates once, never retrying the stop:
#                             upload the outputs and confirm every manifest line on the target (or
#                             rely on the verified network volume), remove the pod, and count the
#                             removal as verified only when remove returned 0 and the pod then reads
#                             ABSENT or TERMINATED.
#
# ABSENT counts as "no longer running" only when a marker in /tmp shows that this container already
# issued a stop or a removal (the launcher clears it at start); otherwise it means the key cannot
# see the pod, which is "unknown". A behaviour directory is never uploaded while the leak flag
# exists (SAD item text was found).
#
# A stopped pod still bills its volume storage. Retrieve the outputs, then terminate the pod
# (runpodctl remove pod <id>) and delete any network volume that is no longer needed.
#
# Environment (the launcher sets these; POD_STATE_FILE adds the verified state, exported):
#   RUNPOD_POD_ID, RUNPOD_API_KEY, POD_UPLOAD (rclone target), POD_OUTPUT_ROOT (results/post_submission)
#   POD_STOP_LOG (default /tmp/pod_stop.log), POD_STATE_FILE, POD_PENDING_FLAG, POD_COMMIT,
#   POD_LEAK_FLAG, POD_LAUNCHER_PID, POD_STOP_LOCK (/tmp/pod_stop.lock),
#   POD_STOP_MARKER (/tmp/pod_stop_marker), POD_INIT_ENVIRON (/proc/1/environ)
#   POD_UPLOAD_PAIRS   "local=remote" subdirectory pairs under POD_OUTPUT_ROOT and POD_UPLOAD
#   POD_STOP_TIMEOUT (120 s), POD_GET_TIMEOUT (60 s), POD_VERIFY_TRIES (6), POD_VERIFY_SLEEP (20 s),
#   POD_UPLOAD_TIMEOUT (1800 s per directory; 900 s before a stop), POD_MANIFEST_TIMEOUT (300 s),
#   POD_STOP_LOCK_WAIT (600 s), POD_TERM_WAIT (30 s), POD_MOUNTS_FILE (/proc/mounts),
#   POD_NETWORK_MOUNT (/workspace), POD_REST_URL (https://rest.runpod.io/v1), POD_STATUS_PY

pod_stop_log() {   # to the stop log; to stderr if the log cannot be written
  local line
  line=$(printf '%s [%s] %s' "$(date -u +%FT%TZ)" "$1" "$2")
  { printf '%s\n' "$line" >> "${POD_STOP_LOG:-/tmp/pod_stop.log}"; } 2>/dev/null \
    || printf '%s\n' "$line" >&2 2>/dev/null || true    # order matters: stdout to stderr first
}

pod_sink() {   # where command output may go: the log if writable, else nowhere
  if { : >> "${POD_STOP_LOG:-/tmp/pod_stop.log}"; } 2>/dev/null; then
    printf '%s' "${POD_STOP_LOG:-/tmp/pod_stop.log}"
  else
    printf '/dev/null'
  fi
}

pod_state() {   # the state the preflight verified, exported so that child processes see it too
  if [ -n "${POD_STATE_FILE:-}" ] && [ -f "$POD_STATE_FILE" ]; then
    set -a
    . "$POD_STATE_FILE" 2>/dev/null || true
    set +a
  fi
}

pod_import_env() {   # RUNPOD_POD_ID and RUNPOD_API_KEY from PID 1's environment; exported values win
  local f=${POD_INIT_ENVIRON:-/proc/1/environ} v
  [ -r "$f" ] || return 0
  if [ -z "${RUNPOD_POD_ID:-}" ]; then
    v=$(tr '\0' '\n' < "$f" 2>/dev/null | sed -n 's/^RUNPOD_POD_ID=//p' | head -1)
    if [ -n "$v" ]; then export RUNPOD_POD_ID=$v; echo "  RUNPOD_POD_ID imported from $f"; fi
  fi
  if [ -z "${RUNPOD_API_KEY:-}" ]; then
    v=$(tr '\0' '\n' < "$f" 2>/dev/null | sed -n 's/^RUNPOD_API_KEY=//p' | head -1)
    if [ -n "$v" ]; then export RUNPOD_API_KEY=$v; echo "  RUNPOD_API_KEY imported from $f (the pod-injected key)"; fi
  elif tr '\0' '\n' < "$f" 2>/dev/null | grep -q '^RUNPOD_API_KEY='; then
    echo "  RUNPOD_API_KEY from this environment kept (the pod-injected key in $f is not used)"
  fi
  return 0
}

pod_upload_pairs() {
  printf '%s\n' ${POD_UPLOAD_PAIRS:-paraphrase/acts=paraphrase/acts paraphrase/acts_validated_only=paraphrase/acts_validated_only paraphrase/floor=paraphrase/floor qwen25_validated/acts=qwen25_validated behaviour=behaviour}
}

pod_leak_flag() {
  printf '%s' "${POD_LEAK_FLAG:-$(dirname "${POD_STOP_LOG:-/tmp/pod_stop.log}")/post_leak_detected}"
}

pod_marker() { printf '%s' "${POD_STOP_MARKER:-/tmp/pod_stop_marker}"; }

pod_mount_of() {   # $1 path -> "mountpoint<TAB>fstype<TAB>source" of the longest mount containing it
  local p
  p=$(realpath -m "$1" 2>/dev/null || printf '%s' "$1")
  awk -v p="$p" '{ mp = $2; gsub(/\\040/, " ", mp)
                   if ((p == mp || index(p, (mp == "/" ? "/" : mp "/")) == 1) && length(mp) > best) {
                     best = length(mp); out = mp "\t" $3 "\t" $1 } }
                 END { if (out != "") print out }' "${POD_MOUNTS_FILE:-/proc/mounts}" 2>/dev/null
}

pod_network_fs() {   # $1 fstype, $2 source -> 0 for a network filesystem
  case "$1" in
    nfs*|cifs|smb*|ceph*|9p|lustre|glusterfs*|mfs|fuse.mfs) return 0 ;;
    fuse) case "$2" in mfs#*) return 0 ;; esac ;;          # RunPod network volume (MooseFS)
  esac
  return 1
}

pod_verify_outputs_safe() {   # 0 if the outputs are safe; writes POD_STATE_FILE
  local net=0 up=0 m mp fs src probe got
  if [ "${NETWORK_VOLUME:-0}" = "1" ]; then
    m=$(pod_mount_of "${POD_OUTPUT_ROOT:-/nonexistent}")
    IFS=$'\t' read -r mp fs src <<< "$m"
    if pod_network_fs "$fs" "$src"; then
      if [ "$mp" = "${POD_NETWORK_MOUNT:-/workspace}" ]; then
        net=1; echo "  outputs on a verified network volume: $mp ($fs, ${src%%[*})"
      else
        echo "  NETWORK_VOLUME=1, but the output root's mount is $mp ($fs), not ${POD_NETWORK_MOUNT:-/workspace}"
      fi
    else
      echo "  NETWORK_VOLUME=1, but the output root is on ${mp:-?} (${fs:-unknown}, source ${src:-?}), not a network filesystem"
    fi
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

# ----------------------------------------------------------------------------- pod state
pod_python() {   # a python that runs, resolved once; empty if none (the bash parser takes over)
  if [ -z "${POD_STATUS_PY_OK+x}" ]; then
    POD_STATUS_PY_OK=""
    local c
    for c in ${POD_STATUS_PY:-python3 python}; do
      [ "$c" = none ] && break
      if command -v "$c" >/dev/null 2>&1 && timeout -k 5 20 "$c" -I -c 'import json' >/dev/null 2>&1; then
        POD_STATUS_PY_OK=$c; break
      fi
    done
  fi
  printf '%s' "$POD_STATUS_PY_OK"
}

pod_parse_sh() {   # $1 source $2 status, POD_RAW -> "STATE<TAB>FORMAT<TAB>detail" (no python)
  local raw=$POD_RAW id=$RUNPOD_POD_ID n rows st
  if [ "$1" = rest ]; then
    [ "$2" = 404 ] && { printf 'ABSENT\tjson\tHTTP 404'; return; }
    [ "$2" = 200 ] || { printf 'UNKNOWN\tother\tHTTP %s' "$2"; return; }
  fi
  if printf '%s\n' "$raw" | grep -qiE '^[[:space:]]*error\b'; then
    printf 'UNKNOWN\tother\t%s' "$(printf '%s' "$raw" | head -1 | cut -c1-120)"; return
  fi
  case "$(printf '%s' "$raw" | tr -d '[:space:]' | cut -c1)" in
    '{'|'[')
      if printf '%s' "$raw" | grep -qE '"code"[[:space:]]*:[[:space:]]*"not_found"'; then
        printf 'ABSENT\tjson\terror code not_found'; return
      fi
      if printf '%s' "$raw" | grep -q '"error"'; then printf 'UNKNOWN\tjson\terror object'; return; fi
      [ "$1" = rest ] || [ "$2" -eq 0 ] || { printf 'UNKNOWN\tjson\texit status %s' "$2"; return; }
      n=$(printf '%s' "$raw" | grep -oE "\"id\"[[:space:]]*:[[:space:]]*\"$id\"" | wc -l)
      [ "$n" -eq 0 ] && { printf 'ABSENT\tjson\tno object carries the id'; return; }
      [ "$n" -gt 1 ] && { printf 'MULTIPLE\tjson\t%s objects carry the id' "$n"; return; }
      st=$(printf '%s' "$raw" | grep -oE '"desiredStatus"[[:space:]]*:[[:space:]]*"[A-Za-z]+"' | sed 's/.*"\([A-Za-z]*\)"$/\1/' | sort -u)
      case "$st" in RUNNING|EXITED|TERMINATED) printf '%s\tjson\tdesiredStatus %s' "$st" "$st" ;;
                    *) printf 'UNKNOWN\tjson\tdesiredStatus not attributable' ;; esac
      return ;;
  esac
  [ "$2" -eq 0 ] || { printf 'UNKNOWN\tother\texit status %s' "$2"; return; }
  if [ -z "$(printf '%s' "$raw" | tr -d '[:space:]')" ]; then printf 'ABSENT\tempty\tempty output'; return; fi
  rows=$(printf '%s\n' "$raw" | grep -wF -- "$id")
  if [ -z "$rows" ]; then
    if printf '%s\n' "$raw" | grep -v '^[[:space:]]*$' | grep -qvE '\bID\b.*STATUS|ID.*Status'; then
      printf 'UNKNOWN\tother\t%s' "$(printf '%s' "$raw" | head -1 | cut -c1-120)"
    else
      printf 'ABSENT\ttable\theader only'
    fi
    return
  fi
  [ "$(printf '%s\n' "$rows" | wc -l)" -gt 1 ] && { printf 'MULTIPLE\ttable\tseveral rows carry the id'; return; }
  for st in RUNNING EXITED TERMINATED; do
    printf '%s' "$rows" | grep -qw "$st" && { printf '%s\ttable\t%s' "$st" "$(printf '%s' "$rows" | cut -c1-160)"; return; }
  done
  printf 'UNKNOWN\ttable\t%s' "$(printf '%s' "$rows" | cut -c1-160)"
}

pod_parse() {   # $1 source $2 status, POD_RAW -> POD_STATUS POD_FORMAT POD_DETAIL
  local py res="" rest
  py=$(pod_python)
  if [ -n "$py" ]; then
    res=$(printf '%s' "$POD_RAW" | timeout -k 5 30 "$py" -I "$POD_STOP_DIR/pod_status.py" \
            "$RUNPOD_POD_ID" "$1" "$2" 2>/dev/null | tr -d '\r')
  fi
  [ -n "$res" ] || res=$(pod_parse_sh "$1" "$2")
  POD_STATUS=${res%%$'\t'*}; rest=${res#*$'\t'}
  POD_FORMAT=${rest%%$'\t'*}; POD_DETAIL=${rest#*$'\t'}
  case "$POD_STATUS" in RUNNING|EXITED|TERMINATED|ABSENT|MULTIPLE|UNKNOWN) ;; *) POD_STATUS=UNKNOWN ;; esac
}

pod_status() {   # -> POD_STATUS, POD_FORMAT, POD_DETAIL, POD_SOURCE, POD_RAW_ALL, POD_CLI; no subshell
  local rc out code
  POD_RAW_ALL=""
  # 1. current runpodctl (JSON); an older runpodctl answers 'Error: unknown command "pod"'
  POD_RAW=$(timeout -k 10 "${POD_GET_TIMEOUT:-60}" runpodctl pod get "$RUNPOD_POD_ID" 2>&1); rc=$?
  POD_RAW_ALL="runpodctl pod get $RUNPOD_POD_ID (rc=$rc):"$'\n'"$POD_RAW"
  pod_parse cli "$rc"; POD_SOURCE="runpodctl pod get"
  [ "$POD_FORMAT" = json ] && POD_CLI=new
  printf '%s' "$POD_RAW" | grep -qi 'unknown command' && POD_CLI=legacy
  case "$POD_STATUS" in RUNNING|EXITED|TERMINATED|ABSENT|MULTIPLE) return 0 ;; esac
  # 2. legacy runpodctl (a table, or JSON)
  POD_RAW=$(timeout -k 10 "${POD_GET_TIMEOUT:-60}" runpodctl get pod "$RUNPOD_POD_ID" 2>&1); rc=$?
  POD_RAW_ALL="$POD_RAW_ALL"$'\n'"runpodctl get pod $RUNPOD_POD_ID (rc=$rc):"$'\n'"$POD_RAW"
  pod_parse legacy "$rc"; POD_SOURCE="runpodctl get pod"
  case "$POD_STATUS" in RUNNING|EXITED|TERMINATED|ABSENT|MULTIPLE) return 0 ;; esac
  # 3. REST API with RUNPOD_API_KEY (the key goes to curl on stdin, never on the command line)
  if [ -n "${RUNPOD_API_KEY:-}" ] && command -v curl >/dev/null 2>&1; then
    out=$(printf 'header = "Authorization: Bearer %s"\n' "$RUNPOD_API_KEY" \
          | timeout -k 10 "${POD_GET_TIMEOUT:-60}" curl -sS -K - -w '\n%{http_code}' \
              "${POD_REST_URL:-https://rest.runpod.io/v1}/pods/$RUNPOD_POD_ID" 2>&1)
    code=${out##*$'\n'}; POD_RAW=${out%$'\n'*}
    case "$code" in [0-9][0-9][0-9]) ;; *) POD_RAW=$out; code=000 ;; esac
    POD_RAW_ALL="$POD_RAW_ALL"$'\n'"REST GET pods/$RUNPOD_POD_ID (HTTP $code):"$'\n'"$(printf '%s' "$POD_RAW" | cut -c1-300)"
    pod_parse rest "$code"; POD_SOURCE="REST API"
  fi
  return 0
}

pod_cmd() {   # $1 stop|remove -> POD_CMD_RC, POD_CMD_OUT, POD_CMD_FORM; output "Error..." is a failure
  local new legacy
  case "$1" in stop) new="pod stop"; legacy="stop pod" ;; *) new="pod delete"; legacy="remove pod" ;; esac
  if [ "${POD_CLI:-}" != legacy ]; then
    POD_CMD_FORM="runpodctl $new"
    POD_CMD_OUT=$(timeout -k 10 "${POD_STOP_TIMEOUT:-120}" runpodctl $new "$RUNPOD_POD_ID" 2>&1); POD_CMD_RC=$?
    if ! printf '%s' "$POD_CMD_OUT" | grep -qi 'unknown command'; then
      printf '%s\n' "$POD_CMD_OUT" | grep -qiE '^[[:space:]]*error\b' && [ "$POD_CMD_RC" -eq 0 ] && POD_CMD_RC=1
      return 0
    fi
  fi
  POD_CMD_FORM="runpodctl $legacy"
  POD_CMD_OUT=$(timeout -k 10 "${POD_STOP_TIMEOUT:-120}" runpodctl $legacy "$RUNPOD_POD_ID" 2>&1); POD_CMD_RC=$?
  printf '%s\n' "$POD_CMD_OUT" | grep -qiE '^[[:space:]]*error\b' && [ "$POD_CMD_RC" -eq 0 ] && POD_CMD_RC=1
  return 0
}

pod_mark() {   # $1 tag $2 action: this container issued a stop or a removal
  { printf '%s %s %s\n' "$(date -u +%FT%TZ)" "$1" "$2" >> "$(pod_marker)"; } 2>/dev/null || true
}

# ----------------------------------------------------------------------- outputs
pod_write_manifest() {   # $1 tag: every completed output with bytes and sha256, then upload it
  # Built by pod_manifest.sh under a timeout, on the local log disk first, so neither a slow hash nor
  # a hung output filesystem can delay the stop; a timeout leaves a partial manifest marked as such.
  local tag=$1 rc n dir man
  dir=$(dirname "${POD_STOP_LOG:-/tmp/pod_stop.log}")
  { : >> "$dir/post_output_manifest.tsv.tmp"; } 2>/dev/null || dir=/tmp     # unwritable log dir
  man=$dir/post_output_manifest.tsv
  POD_MANIFEST=$man
  if [ -z "${POD_OUTPUT_ROOT:-}" ]; then
    pod_stop_log "$tag" "no output root yet; no manifest"
    POD_MANIFEST=""
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

pod_upload_outputs() {   # $1 tag [$2 timeout per directory] -> 0 when every output directory is uploaded
  local tag=$1 to=${2:-${POD_UPLOAD_TIMEOUT:-1800}} pair d rc bad=0 n=0
  if [ -z "${POD_UPLOAD:-}" ] || ! command -v rclone >/dev/null 2>&1; then
    pod_stop_log "$tag" "no upload target or no rclone"
    return 1
  fi
  for pair in $(pod_upload_pairs); do
    if [ "${pair%%=*}" = behaviour ] && [ -e "$(pod_leak_flag)" ]; then
      pod_stop_log "$tag" "behaviour not uploaded: the leak flag exists (SAD item text was found)"
      continue
    fi
    d=${POD_OUTPUT_ROOT:-/nonexistent}/${pair%%=*}
    timeout -k 2 10 test -d "$d" || continue          # a hung filesystem must not hang the stop
    timeout -k 30 "$to" rclone copy --checksum --exclude '*.tmp.npz' \
      "$d" "$POD_UPLOAD/${pair#*=}" >> "$(pod_sink)" 2>&1
    rc=$?
    pod_stop_log "$tag" "upload ${pair%%=*} -> $POD_UPLOAD/${pair#*=}: rc=$rc"
    [ "$rc" -eq 0 ] || bad=1
    n=$((n + 1))
  done
  [ "$n" -gt 0 ] || pod_stop_log "$tag" "no output directory exists yet: nothing to upload"
  return "$bad"
}

pod_confirm_upload() {   # $1 tag -> 0 when every manifest line is on the target with its size (and
  # its sha256 where the backend reports one); a truncated manifest is never fully confirmed
  local tag=$1 man=${POD_MANIFEST:-} pair loc rem listing hashes hmode path bytes sha base
  local ok=0 bad=0 skipped=0 trunc=0
  if [ -z "$man" ] || [ ! -f "$man" ]; then
    pod_stop_log "$tag" "no manifest: nothing to confirm the upload against"
    return 1
  fi
  [ -z "${POD_UPLOAD:-}" ] && { pod_stop_log "$tag" "no upload target: nothing confirmed"; return 1; }
  grep -q '^# TRUNCATED' "$man" && trunc=1
  for pair in $(pod_upload_pairs); do
    loc=${pair%%=*}; rem=${pair#*=}
    awk -F'\t' -v d="$loc" 'index($1, d "/") == 1 && substr($1, length(d) + 2) !~ /\// { f = 1 } END { exit !f }' "$man" || continue
    if [ "$loc" = behaviour ] && [ -e "$(pod_leak_flag)" ]; then
      skipped=$(awk -F'\t' -v d="$loc" 'index($1, d "/") == 1 { n++ } END { print n + 0 }' "$man")
      continue
    fi
    listing=$(timeout -k 10 300 rclone lsf --files-only --format ps --separator ';' "$POD_UPLOAD/$rem" 2>/dev/null) || listing=""
    if hashes=$(timeout -k 10 900 rclone hashsum sha256 "$POD_UPLOAD/$rem" 2>/dev/null) && [ -n "$hashes" ]; then
      hmode=sha256
    else
      hashes=""; hmode="size only (no sha256 from the backend)"
    fi
    while IFS=$'\t' read -r path bytes sha; do
      case "$path" in ''|'#'*|path) continue ;; esac
      [ "${path%/*}" = "$loc" ] || continue
      base=${path##*/}
      if printf '%s\n' "$listing" | grep -qxF "$base;$bytes" \
         && { [ -z "$hashes" ] || printf '%s\n' "$hashes" | grep -qxF "$sha  $base"; }; then
        ok=$((ok + 1))
      else
        bad=$((bad + 1)); pod_stop_log "$tag" "NOT CONFIRMED on the target: $path ($bytes bytes)"
      fi
    done < "$man"
    pod_stop_log "$tag" "confirmed $loc against $POD_UPLOAD/$rem: $hmode"
  done
  pod_stop_log "$tag" "upload confirmation: $ok manifest lines confirmed, $bad not confirmed, $skipped behaviour lines withheld (leak flag)$([ "$trunc" = 1 ] && echo '; the manifest is TRUNCATED, so outputs beyond it are unconfirmed')"
  [ "$bad" -eq 0 ] && [ "$trunc" -eq 0 ]
}

# ------------------------------------------------------------------- verification
pod_still_running() {   # 0 = RUNNING or unknown (not verified), 1 = confirmed no longer running
  local tag=$1 i
  pod_stop_log "$tag" "verifying the stop (desiredStatus; ABSENT counts only after this container's own stop or removal)"
  for i in $(seq 1 "${POD_VERIFY_TRIES:-6}"); do
    pod_status
    case "$POD_STATUS" in
      EXITED|TERMINATED)
        pod_stop_log "$tag" "check $i: $POD_STATUS ($POD_SOURCE: $POD_DETAIL)"; return 1 ;;
      ABSENT)
        if [ -s "$(pod_marker)" ] && [ "${POD_SEEN_ABSENT:-0}" != 1 ]; then
          pod_stop_log "$tag" "check $i: ABSENT after this container's $(tail -1 "$(pod_marker)" | cut -d' ' -f3) ($POD_SOURCE: $POD_DETAIL)"
          return 1
        fi
        pod_stop_log "$tag" "check $i: ABSENT, but $([ "${POD_SEEN_ABSENT:-0}" = 1 ] && echo "it read ABSENT before this container issued anything" || echo "this container issued no stop or removal"): status unknown ($POD_SOURCE: $POD_DETAIL)" ;;
      RUNNING)
        pod_stop_log "$tag" "check $i: RUNNING ($POD_SOURCE: $POD_DETAIL)" ;;
      *)
        pod_stop_log "$tag" "check $i: $POD_STATUS, status unknown ($POD_SOURCE: $POD_DETAIL)" ;;
    esac
    [ "$i" -lt "${POD_VERIFY_TRIES:-6}" ] && sleep "${POD_VERIFY_SLEEP:-20}"
  done
  return 0
}

pod_absent() {   # $1 tag $2 remove status -> 0 = verified: remove returned 0 AND the pod is gone
  local tag=$1 rrc=$2 i
  for i in $(seq 1 "${POD_VERIFY_TRIES:-6}"); do
    pod_status
    case "$POD_STATUS" in
      ABSENT|TERMINATED)
        if [ "$rrc" -eq 0 ]; then
          pod_stop_log "$tag" "absence check $i: $POD_STATUS after remove rc=0 ($POD_SOURCE: $POD_DETAIL): removal verified"
          return 0
        fi
        pod_stop_log "$tag" "absence check $i: $POD_STATUS, but remove returned $rrc: not counted as verified"
        return 1 ;;
    esac
    pod_stop_log "$tag" "absence check $i: $POD_STATUS ($POD_SOURCE: $POD_DETAIL)"
    [ "$rrc" -eq 0 ] || return 1
    [ "$i" -lt "${POD_VERIFY_TRIES:-6}" ] && sleep "${POD_VERIFY_SLEEP:-20}"
  done
  return 1
}

pod_remove_path() {   # $1 tag: secure the outputs, remove the pod, verify removal
  local tag=$1 attempt
  if [ "${POD_OUTPUTS_ON_NETWORK:-0}" = "1" ]; then
    pod_stop_log "$tag" "outputs are on the verified network volume, which outlives the pod: no upload needed"
  elif pod_upload_outputs "$tag" && pod_confirm_upload "$tag"; then
    pod_stop_log "$tag" "every manifest line confirmed on the upload target"
  else
    pod_stop_log "$tag" "UPLOAD NOT CONFIRMED: some outputs in the manifest are not confirmed off the pod; removing anyway, since keeping the pod only adds cost (the watchdog would remove it later)"
  fi
  for attempt in 1 2; do
    pod_mark "$tag" remove
    pod_cmd remove
    pod_stop_log "$tag" "$POD_CMD_FORM $RUNPOD_POD_ID (attempt $attempt): rc=$POD_CMD_RC: $(printf '%s' "$POD_CMD_OUT" | tr '\n' ' ' | cut -c1-300)"
    if pod_absent "$tag" "$POD_CMD_RC"; then
      return 0
    fi
  done
  pod_stop_log "$tag" "REMOVAL NOT VERIFIED (remove never returned 0 with the pod then gone): CHECK AND REMOVE THE POD BY HAND"
  return 1
}

# ------------------------------------------------------------------- lock and launcher
pod_lock() {   # $1 tag: one stop at a time; a lock older than POD_STOP_LOCK_WAIT is taken over
  local lock=${POD_STOP_LOCK:-/tmp/pod_stop.lock} waited=0 max=${POD_STOP_LOCK_WAIT:-600}
  while ! mkdir "$lock" 2>/dev/null; do
    if [ "$waited" -ge "$max" ]; then
      pod_stop_log "$1" "stop lock still held after $max s ($(cat "$lock/owner" 2>/dev/null)): taking it over"
      rm -rf "$lock" 2>/dev/null; mkdir "$lock" 2>/dev/null || true
      break
    fi
    [ "$waited" -eq 0 ] && pod_stop_log "$1" "waiting for the stop lock ($(cat "$lock/owner" 2>/dev/null)), at most $max s"
    sleep 5; waited=$((waited + 5))
  done
  { printf '%s %s\n' "${BASHPID:-$$}" "$1" > "$lock/owner"; } 2>/dev/null || true
}

pod_unlock() {
  local lock=${POD_STOP_LOCK:-/tmp/pod_stop.lock}
  [ "$(cut -d' ' -f1 "$lock/owner" 2>/dev/null)" = "${BASHPID:-$$}" ] && rm -rf "$lock" 2>/dev/null
  return 0
}

pod_term_launcher() {   # $1 tag: TERM the launcher's running step and the launcher, wait briefly
  local lp=${POD_LAUNCHER_PID:-} me=${BASHPID:-$$} c comm i
  if [ -z "$lp" ] || ! kill -0 "$lp" 2>/dev/null; then
    pod_stop_log "$1" "launcher not running (pid ${lp:-unknown})"
    return 0
  fi
  for c in $(pgrep -P "$lp" 2>/dev/null || ps -ef 2>/dev/null | awk -v p="$lp" '$3 == p { print $2 }'); do
    [ "$c" = "$me" ] && continue
    comm=$(cat "/proc/$c/comm" 2>/dev/null || ps -o comm= -p "$c" 2>/dev/null)
    case "$comm" in tee|*/tee) continue ;; esac          # the launcher's log
    kill -TERM "$c" 2>/dev/null || true
  done
  kill -TERM "$lp" 2>/dev/null || true
  pod_stop_log "$1" "sent TERM to the launcher (pid $lp) and its running step"
  for i in $(seq 1 "${POD_TERM_WAIT:-30}"); do
    kill -0 "$lp" 2>/dev/null || break
    sleep 1
  done
  if kill -0 "$lp" 2>/dev/null; then
    pod_stop_log "$1" "launcher still running after ${POD_TERM_WAIT:-30} s (its EXIT trap may be waiting for this lock): going on"
  else
    pod_stop_log "$1" "launcher exited"
  fi
}

# ----------------------------------------------------------------------------- stop
pod_stop() {   # $1 caller tag (autostop | watchdog); a second argument is accepted and ignored
  local tag=${1:-stop} rc
  set +e
  trap '' PIPE
  pod_state
  [ -n "${RUNPOD_POD_ID:-}" ] || pod_import_env >/dev/null
  if [ -z "${RUNPOD_POD_ID:-}" ] || ! command -v runpodctl >/dev/null 2>&1; then
    pod_stop_log "$tag" "cannot stop: RUNPOD_POD_ID or runpodctl missing; STOP THE POD BY HAND"
    return 1
  fi
  pod_lock "$tag"
  pod_stop_locked "$tag"
  rc=$?
  pod_unlock
  return "$rc"
}

pod_stop_locked() {
  local tag=$1
  [ "$tag" = watchdog ] && pod_term_launcher "$tag"
  pod_status
  pod_stop_log "$tag" "state before the stop: $POD_STATUS ($POD_SOURCE: $POD_DETAIL)"
  if { [ "$POD_STATUS" = ABSENT ] || [ "$POD_STATUS" = TERMINATED ]; } && [ -s "$(pod_marker)" ]; then
    pod_stop_log "$tag" "this container already issued: $(tail -1 "$(pod_marker)"); nothing more to do"
    return 0
  fi
  POD_SEEN_ABSENT=0
  if [ "$POD_STATUS" = ABSENT ]; then        # the key may not see the pod: absence proves nothing now
    POD_SEEN_ABSENT=1
    pod_stop_log "$tag" "the pod reads ABSENT before this container issued anything: a later ABSENT will not count as stopped"
  fi
  pod_write_manifest "$tag"
  if [ "${POD_OUTPUTS_ON_NETWORK:-0}" != "1" ] && [ -f "${POD_PENDING_FLAG:-/nonexistent}" ] && [ -n "${POD_UPLOAD:-}" ]; then
    pod_stop_log "$tag" "a step was in progress ($(cat "$POD_PENDING_FLAG" 2>/dev/null)) and the outputs are not on a network volume: uploading before the stop"
    if pod_upload_outputs "$tag" "${POD_PRESTOP_UPLOAD_TIMEOUT:-900}" && pod_confirm_upload "$tag"; then
      pod_stop_log "$tag" "every manifest line confirmed on the upload target before the stop"
    else
      pod_stop_log "$tag" "upload before the stop NOT CONFIRMED; stopping anyway"
    fi
  fi
  pod_mark "$tag" stop
  pod_cmd stop
  { [ "$POD_CMD_RC" -eq 124 ] || [ "$POD_CMD_RC" -eq 137 ]; } && pod_stop_log "$tag" "stop timed out"
  pod_stop_log "$tag" "$POD_CMD_FORM $RUNPOD_POD_ID: rc=$POD_CMD_RC: $(printf '%s' "$POD_CMD_OUT" | tr '\n' ' ' | cut -c1-300)"
  if ! pod_still_running "$tag"; then
    pod_stop_log "$tag" "verified: pod is no longer RUNNING"
    return 0
  fi
  pod_stop_log "$tag" "stop not verified (refused, hung or status unknown): escalating to the remove path, no stop retry"
  pod_remove_path "$tag"
}
