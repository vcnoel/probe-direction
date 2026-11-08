#!/usr/bin/env bash
# Write the output manifest: every completed output (npz and json, partial .tmp.npz writes excluded)
# under POD_OUTPUT_ROOT's upload directories, with bytes and sha256, plus the step in progress.
# Called by pod_stop.sh under `timeout`, so a hung filesystem cannot delay the stop. Lines are
# appended as they are computed, so a timeout leaves a usable partial manifest, which the caller
# marks as truncated.
#
#   pod_manifest.sh TAG OUTFILE        (environment as for pod_stop.sh)
tag=$1 out=$2
root=${POD_OUTPUT_ROOT:-}
{
  printf '# outputs at %s, written by %s; pod %s; commit %s; step in progress: %s\n' \
    "$(date -u +%FT%TZ)" "$tag" "${RUNPOD_POD_ID:-?}" "${POD_COMMIT:-?}" \
    "$(cat "${POD_PENDING_FLAG:-/nonexistent}" 2>/dev/null || echo none)"
  printf 'path\tbytes\tsha256\n'
} > "$out"
[ -n "$root" ] && [ -d "$root" ] || exit 0
for pair in ${POD_UPLOAD_PAIRS:-paraphrase/acts=paraphrase/acts paraphrase/acts_validated_only=paraphrase/acts_validated_only paraphrase/floor=paraphrase/floor qwen25_validated/acts=qwen25_validated behaviour=behaviour}; do
  d=$root/${pair%%=*}
  [ -d "$d" ] || continue
  for f in "$d"/*.npz "$d"/*.json; do
    [ -f "$f" ] || continue
    case "$f" in *.tmp.npz|*.tmp) continue ;; esac
    printf '%s\t%s\t%s\n' "${f#"$root"/}" "$(stat -c %s "$f" 2>/dev/null || wc -c < "$f")" \
      "$(sha256sum "$f" 2>/dev/null | cut -d' ' -f1)" >> "$out"
  done
done
