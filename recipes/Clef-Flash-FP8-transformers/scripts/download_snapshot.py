#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Download a pinned Hugging Face snapshot into the standard HF cache layout, verified.

Standard library only. Writes exactly what huggingface_hub would:

    $HF_HOME/hub/models--<org>--<name>/blobs/<blob id>
    $HF_HOME/hub/models--<org>--<name>/snapshots/<revision>/<path> -> ../../blobs/<blob id>

so the files are shared with every other tool that reads the HF cache. It does
not write refs/main: that ref belongs to whoever last downloaded `main`, and the
recipe addresses the snapshot by its revision directory instead.

Every file is verified before it is linked into the snapshot: LFS files by
SHA-256, small files by their git blob id (sha1 of "blob <len>\\0" + bytes),
which is what the Hub's tree API reports for them.

A short response body is the normal case on a slow link, not a failure, so each
file is resumed with a Range request from whatever is already on disk. Only
attempts that add no bytes count against --stall-attempts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

CHUNK = 8 * 1024 * 1024


def repo_dir(hf_home: Path, repo: str) -> Path:
    return hf_home / "hub" / ("models--" + repo.replace("/", "--"))


def digest_file(path: Path, entry: dict) -> str:
    """The id the manifest records for this file: SHA-256 for LFS files, the git blob sha1 otherwise."""

    if "sha256" in entry:
        h = hashlib.sha256()
    else:
        h = hashlib.sha1()
        h.update(b"blob %d\0" % path.stat().st_size)
    with path.open("rb") as f:
        while block := f.read(CHUNK):
            h.update(block)
    return h.hexdigest()


def expected_id(entry: dict) -> str:
    return entry.get("sha256") or entry["blob"]


def check(path: Path, entry: dict) -> str | None:
    """None if ``path`` is the file the manifest names, else why not."""

    if not path.is_file():
        return "missing"
    size = path.stat().st_size
    if size != entry["bytes"]:
        return f"size {size}, expected {entry['bytes']}"
    got = digest_file(path, entry)
    if got != expected_id(entry):
        return f"digest {got[:16]}, expected {expected_id(entry)[:16]}"
    return None


def fetch(url: str, partial: Path, total: int, token: str, stall_attempts: int) -> None:
    """Fill ``partial`` to ``total`` bytes, resuming with Range from what is on disk."""

    stalls = 0
    while True:
        have = partial.stat().st_size if partial.exists() else 0
        if have == total:
            return
        if have > total:
            partial.unlink()                       # larger than the file: not a prefix of it
            continue
        headers = {"User-Agent": "dgx-spark-recipes/1"}
        if token:
            headers["Authorization"] = f"Bearer {token}"
        if have:
            headers["Range"] = f"bytes={have}-"
        before = have
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=60) as resp:
                mode = "ab"
                if have and resp.status != 206:     # the server ignored Range: start over
                    mode, have = "wb", 0
                with partial.open(mode) as out:
                    while block := resp.read(CHUNK):
                        out.write(block)
        except urllib.error.HTTPError as exc:
            if exc.code in (401, 403, 404):
                raise SystemExit(f"{url}: HTTP {exc.code} (gated, private or wrong revision?)")
            if exc.code == 416:                    # range past the end: re-check the size on disk
                continue
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as exc:
            print(f"  {partial.name}: {exc}; resuming", file=sys.stderr, flush=True)
        now = partial.stat().st_size if partial.exists() else 0
        # Progress is bytes on disk beyond what was there before this attempt. A
        # server that ignores Range and truncates would otherwise resend the same
        # opening bytes forever and look like progress every time.
        if now > before:
            stalls = 0
        else:
            stalls += 1
            if stalls >= stall_attempts:
                raise SystemExit(f"{partial.name}: no progress in {stalls} attempts at {now} of {total} bytes")
            time.sleep(min(30, 2 ** stalls))


def link(snapshot: Path, rel: str, blob: Path) -> None:
    dest = snapshot / rel
    dest.parent.mkdir(parents=True, exist_ok=True)
    target = os.path.relpath(blob, dest.parent)
    if dest.is_symlink() and os.readlink(dest) == target:
        return
    if dest.is_symlink() or dest.exists():
        dest.unlink()
    dest.symlink_to(target)


def one(entry: dict, args, root: Path, snapshot: Path, token: str) -> str:
    blob = root / "blobs" / entry["blob"]
    rel = entry["path"]
    if args.verify_only:
        problem = check(snapshot / rel, entry)
        if problem:
            raise SystemExit(f"{rel}: {problem}")
        return f"ok      {rel}"
    if blob.is_file() and blob.stat().st_size == entry["bytes"] and check(blob, entry) is None:
        link(snapshot, rel, blob)
        return f"present {rel}"
    partial = blob.with_name(blob.name + ".incomplete")
    url = (f"{args.endpoint}/{args.repo}/resolve/{args.revision}/"
           + urllib.parse.quote(rel))
    fetch(url, partial, entry["bytes"], token, args.stall_attempts)
    problem = check(partial, entry)
    if problem:
        partial.unlink()                           # wrong bytes: the next run starts this file over
        raise SystemExit(f"{rel}: downloaded file failed verification ({problem})")
    partial.rename(blob)
    link(snapshot, rel, blob)
    return f"fetched {rel} ({entry['bytes'] / 2**30:.2f} GiB)"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--hf-home", type=Path, default=Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface")))
    p.add_argument("--endpoint", default=os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/"))
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--stall-attempts", type=int, default=8)
    p.add_argument("--verify-only", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    m = json.loads(args.manifest.read_text())
    args.repo, args.revision = m["repo"], m["revision"]
    root = repo_dir(args.hf_home, args.repo)
    snapshot = root / "snapshots" / args.revision
    print(f"repo      {args.repo} @ {args.revision}")
    print(f"snapshot  {snapshot}")
    print(f"files     {len(m['files'])}, {m['total_bytes'] / 2**30:.2f} GiB")
    if args.dry_run:
        return 0
    (root / "blobs").mkdir(parents=True, exist_ok=True)
    snapshot.mkdir(parents=True, exist_ok=True)
    token = os.environ.get("HF_TOKEN", "")

    # Largest first, so the long transfers overlap instead of trailing at the end.
    files = sorted(m["files"], key=lambda e: -e["bytes"])
    failed = 0
    with ThreadPoolExecutor(max(1, args.workers)) as pool:
        futures = {pool.submit(one, e, args, root, snapshot, token): e for e in files}
        for fut in as_completed(futures):
            try:
                print(fut.result(), flush=True)
            except SystemExit as exc:
                failed += 1
                print(f"FAILED  {exc}", file=sys.stderr, flush=True)
    if failed:
        print(f"{failed} file(s) failed", file=sys.stderr)
        return 1
    print("verified" if args.verify_only else "complete and verified")
    return 0


if __name__ == "__main__":
    sys.exit(main())
