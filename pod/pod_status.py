"""Read one RunPod pod's state from whatever the CLI or the REST API printed. Standard library only.

    pod_status.py POD_ID SOURCE STATUS < raw_output

SOURCE is `cli` (runpodctl pod get, JSON by default), `legacy` (runpodctl get pod: a table in the
releases that keep it, possibly JSON in others) or `rest` (GET rest.runpod.io/v1/pods/ID, STATUS is
the HTTP code). STATUS is the command's exit status otherwise. Prints one line, never raises:

    STATE<TAB>FORMAT<TAB>detail

STATE is RUNNING, EXITED or TERMINATED (desiredStatus of the one object or row carrying the id),
ABSENT (the source answered and nothing carries the id: a not_found error, an empty listing, a
header-only table, an empty JSON list, a REST 404), MULTIPLE (more than one object or row carries
the id) or UNKNOWN (an error, unparseable output, or a status that is none of the above). FORMAT
is json, table, empty or other. Formats follow the runpodctl README: data on stdout, errors on
stderr as one flat JSON object with a `code` field (`not_found` for a missing resource) and a
non-zero exit; the legacy commands print `Error: <msg>` and exit 1.
"""

from __future__ import annotations

import json
import re
import sys

STATES = ("RUNNING", "EXITED", "TERMINATED")


def _docs(text: str):
    """Every JSON document in the text: the whole text, or one per line (stdout plus stderr)."""
    s = text.strip()
    if not s or s[0] not in "[{":
        out = []
    else:
        try:
            return [json.loads(s)]
        except ValueError:
            out = []
    for line in s.splitlines():
        line = line.strip()
        if line[:1] in ("{", "["):
            try:
                out.append(json.loads(line))
            except ValueError:
                pass
    return out


def _carriers(node, pid, found):
    """Objects whose id is the pod id; objects already matched are not searched further."""
    if isinstance(node, dict):
        if str(node.get("id", "")) == pid:
            found.append(node)
            return
        for v in node.values():
            if isinstance(v, (dict, list)):
                _carriers(v, pid, found)
    elif isinstance(node, list):
        for v in node:
            _carriers(v, pid, found)


def _state_of(obj) -> str:
    for key in ("desiredStatus", "desired_status", "DesiredStatus"):
        v = obj.get(key)
        if isinstance(v, str):
            return v.upper() if v.upper() in STATES else "UNKNOWN"
    return "UNKNOWN"


def parse(pid: str, source: str, status: int, raw: str):
    docs = _docs(raw)
    if source == "rest":
        if status == 404:
            return "ABSENT", "json", "HTTP 404"
        if status != 200:
            return "UNKNOWN", "other", f"HTTP {status}"
    if docs:
        for d in docs:
            if isinstance(d, dict) and "error" in d and str(d.get("id", "")) != pid:
                code = str(d.get("code", ""))
                if code == "not_found" or d.get("status") == 404:
                    return "ABSENT", "json", f"error code {code or 404}"
                return "UNKNOWN", "json", f"error code {code or '?'} status {d.get('status', '?')}"
        if status != 0 and source != "rest":
            return "UNKNOWN", "json", f"exit status {status}"
        found: list = []
        for d in docs:
            _carriers(d, pid, found)
        if not found:
            return "ABSENT", "json", "no object carries the id"
        if len(found) > 1:
            return "MULTIPLE", "json", f"{len(found)} objects carry the id"
        st = _state_of(found[0])
        return st, "json", f"desiredStatus {found[0].get('desiredStatus', '?')}"
    lines = [ln for ln in raw.splitlines() if ln.strip()]
    if status != 0:
        return "UNKNOWN", "other", f"exit status {status}: {(lines or [''])[0][:120]}"
    if not lines:
        return "ABSENT", "empty", "empty output"
    if any(re.match(r"\s*error\b", ln, re.I) for ln in lines):
        return "UNKNOWN", "other", lines[0][:120]
    rows = [ln for ln in lines if pid in ln.split()]
    if not rows:
        header = all(re.search(r"\bID\b", ln) and re.search(r"STATUS", ln, re.I) for ln in lines)
        if header:
            return "ABSENT", "table", "header only"
        return "UNKNOWN", "other", lines[0][:120]
    if len(rows) > 1:
        return "MULTIPLE", "table", f"{len(rows)} rows carry the id"
    tokens = set(rows[0].upper().split())
    hit = [s for s in STATES if s in tokens]
    if len(hit) == 1:
        return hit[0], "table", rows[0].strip()[:160]
    return "UNKNOWN", "table", rows[0].strip()[:160]


def main() -> None:
    try:
        pid, source, status = sys.argv[1], sys.argv[2], int(sys.argv[3])
        raw = sys.stdin.buffer.read().decode("utf-8", "replace")
        st, fmt, detail = parse(pid, source, status, raw)
    except Exception as e:   # never raise: the stop path reads whatever is printed
        st, fmt, detail = "UNKNOWN", "other", f"parser error {type(e).__name__}"
    print(f"{st}\t{fmt}\t{' '.join(detail.split())}")


if __name__ == "__main__":
    main()
