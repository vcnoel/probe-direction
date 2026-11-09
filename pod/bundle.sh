#!/usr/bin/env bash
# Copy step for a pod: a tarball of the committed tree at HEAD, with LF line endings.
#
#     bash pod/bundle.sh [out.tar]        # then copy out.tar to the pod and untar it there
#
# git archive writes blobs through the checkout conversion, so it is run with core.autocrlf=false;
# .gitattributes sets eol=lf for *.sh and *.py besides. The archive is then checked in python
# (MSYS bash drops CR inside $(...), so a shell check could pass a CRLF file): any CR byte in a
# script refuses the bundle. Untracked and modified files are never bundled; the launcher refuses
# a modified tree anyway.
set -euo pipefail
cd "$(dirname "$0")/.."
out=${1:-post_bundle.tar}
if [ -z "${PY:-}" ]; then           # the first python that runs (python3 may be a store alias)
  for c in python3 python; do "$c" -c 1 >/dev/null 2>&1 && { PY=$c; break; }; done
fi
git -c core.autocrlf=false archive --format=tar -o "$out" HEAD
"$PY" - "$out" <<'PY'
import sys, tarfile
bad, n = [], 0
with tarfile.open(sys.argv[1]) as t:
    for m in t.getmembers():
        if m.isfile() and m.name.endswith((".sh", ".py")):
            n += 1
            if b"\r" in t.extractfile(m).read():
                bad.append(m.name)
for b in bad:
    print(f"CR byte in {b}", file=sys.stderr)
if bad:
    raise SystemExit(f"refusing the bundle: {len(bad)} scripts carry CR bytes")
print(f"{n} scripts checked: LF only")
PY
echo "bundle $out of commit $(git rev-parse HEAD)"
