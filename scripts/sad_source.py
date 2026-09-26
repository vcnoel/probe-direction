"""Read the SAD stages_oversight split from a password-protected local archive.

The SAD authors require that question and answer texts never appear in plain text anywhere
scrapable, including private repositories, because scraped benchmark text ends up in pretraining
corpora and invalidates the benchmark for everyone. This repository previously committed the
400-item split as plain JSON; that file has been purged from history and is now read from an
encrypted archive held outside the working tree, mirroring the scheme SAD itself uses.

Nothing here writes plaintext to disk. The items are decrypted into memory on demand.

Configure with environment variables if the defaults do not apply:

    EAS_SAD_ARCHIVE   path to the password-protected zip
    EAS_SAD_PASSWORD  password for it

    from sad_source import load_sad
    items = load_sad()          # list of {"question_id", "question", "type"}
"""

from __future__ import annotations

import io
import json
import os
import zipfile
from pathlib import Path

# No default location: the archive is licence-restricted, is never committed here, and lived at an
# absolute path on one machine until this was a release blocker. Unset is resolved at use, not here,
# so importing this module for its helpers does not require the benchmark to be present.
DEFAULT_ARCHIVE = Path(os.environ.get("EAS_SAD_ARCHIVE", ""))
# No default. The SAD archive is encrypted by its authors to keep its question text out of
# training corpora, and a public repository that ships the password defeats exactly that. The
# password comes from the SAD project; set EAS_SAD_PASSWORD from there.
DEFAULT_PASSWORD = os.environ.get("EAS_SAD_PASSWORD", "")
MEMBER = "sad_stages_oversight_400.json"


def load_sad(archive: str | Path | None = None, password: str | None = None):
    """-> list of item dicts, decrypted in memory."""
    path = Path(archive) if archive else DEFAULT_ARCHIVE
    if not str(path):
        raise SystemExit(
            "EAS_SAD_ARCHIVE is not set. The SAD benchmark is licence-restricted and is "
            "not redistributed with this repository; see the 'Regenerating results' "
            "section of README.md.")
    if not path.exists():
        raise FileNotFoundError(
            f"SAD archive not found at {path}. Set EAS_SAD_ARCHIVE, or recreate it with\n"
            f"  zip -P <password> {path.name} sad_stages_oversight_400.json\n"
            f"The plaintext split must not be committed to this repository.")
    pwd = (password or DEFAULT_PASSWORD)
    if not pwd:
        raise SystemExit(
            "EAS_SAD_PASSWORD is not set. The SAD archive is encrypted by its authors to "
            "keep its question text out of training corpora; obtain the password from "
            "the SAD project.")
    pwd = pwd.encode()
    with zipfile.ZipFile(path) as z:
        name = MEMBER if MEMBER in z.namelist() else z.namelist()[0]
        raw = z.read(name, pwd=pwd)
    return json.loads(raw.decode("utf-8"))


def load_sad_frame():
    """-> pandas DataFrame, for call sites that used pd.read_json on the old file."""
    import pandas as pd
    return pd.DataFrame(load_sad())


def load_sad_bodies():
    """-> list of raw item bodies with the Qwen chat template stripped.

    The archived split is already rendered with Qwen's chat template, which makes it unusable
    for other families: feeding Qwen control tokens to Gemma or Llama measures the wrong thing.
    Recovering the body lets each family apply its own template, which is what the reference
    implementation does. Note the recovered body is only as faithful as the rendering allowed;
    where an exact body is needed, the SAD repository's own struct files are the source.
    """
    out = []
    for it in load_sad():
        q = it["question"].replace("\r\n", "\n").replace("\r", "\n")
        body = q
        if "<|im_start|>user" in q:
            body = q.split("<|im_start|>user", 1)[1]
            body = body.split("<|im_end|>", 1)[0]
        out.append({**it, "body": body.strip("\n")})
    return out


if __name__ == "__main__":
    items = load_sad()
    print(f"{len(items)} items; types: "
          f"{ {t: sum(i['type'] == t for i in items) for t in {i['type'] for i in items}} }")
    b = load_sad_bodies()[0]["body"]
    print(f"first body starts: {b[:60]!r}")
