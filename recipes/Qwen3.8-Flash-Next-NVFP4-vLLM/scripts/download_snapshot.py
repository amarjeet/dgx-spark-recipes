#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Copyright (C) 2026 amarjeet
#
# Written for this port. Part of the same AGPL-3.0-or-later combined work
# as the files it sits beside, which derive from MiaAI-Lab/
# Qwen3.8-Flash-Next-Single-DGX-Spark, Copyright (C) 2026 MiaAI Lab.
"""Download a pinned Hugging Face revision into the Hugging Face cache.

Pure stdlib on purpose: the repo's recipes assume `python3` and nothing else,
and `huggingface_hub` is not installed on this host. Upstream shells out to
`snapshot_download` inside the container image, which makes fetching 98.66 GiB
wait on a 20 GB image pull; this does not.

What it writes is the ordinary cache layout, so vLLM (and anything else using
`huggingface_hub`) resolves the model from it with no special handling:

    $HF_HOME/hub/models--<org>--<name>/
        refs/main                     the pinned commit
        blobs/<etag>                  file contents, stored once
        snapshots/<commit>/<path>     relative symlink into blobs/

The ETag is the sha256 for an LFS object and the git blob sha1 for a regular
file -- the same names `huggingface_hub` would choose, so a later `hf download`
sees a warm cache rather than re-fetching.

Downloads resume: a partial blob is kept at `<etag>.incomplete` and continued
with a Range request. Every file is verified by size and SHA-256 before it is
linked into the snapshot, so an interrupted transfer can never present itself
as a complete checkpoint.

Usage:
  ./scripts/download_snapshot.py --manifest manifests/nvfp4.json
  ./scripts/download_snapshot.py --manifest manifests/nvfp4.json --verify-only
  ./scripts/download_snapshot.py --manifest manifests/nvfp4.json --workers 8
"""
import argparse
import concurrent.futures
import hashlib
import json
import os
import shutil
import sys
import threading
import time
import urllib.error
import urllib.request

CHUNK = 8 * 1024 * 1024
_print_lock = threading.Lock()


def say(*args):
    with _print_lock:
        print(*args, flush=True)


def human(n: float) -> str:
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(n) < 1024 or unit == "TiB":
            return f"{n:.1f} {unit}"
        n /= 1024


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(CHUNK), b""):
            digest.update(block)
    return digest.hexdigest()


def open_url(url: str, token: str, offset: int = 0):
    headers = {"User-Agent": "dgx-spark-recipes/1.0"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    if offset:
        headers["Range"] = f"bytes={offset}-"
    return urllib.request.urlopen(
        urllib.request.Request(url, headers=headers), timeout=120)


def fetch_blob(url: str, blob: str, expect_bytes: int, expect_sha: str,
               token: str, label: str) -> None:
    """Download `url` to `blob`, resuming and verifying. No-op if already good."""
    if os.path.exists(blob):
        if os.path.getsize(blob) == expect_bytes:
            return
        # A complete-looking blob of the wrong size is corrupt, not resumable.
        os.remove(blob)

    partial = f"{blob}.incomplete"
    offset = os.path.getsize(partial) if os.path.exists(partial) else 0
    if offset > expect_bytes:
        os.remove(partial)
        offset = 0

    if offset < expect_bytes:
        started = time.monotonic()
        response = open_url(url, token, offset)
        # A server that ignores Range replies 200 and restarts the body at 0;
        # appending it to what we already have would silently corrupt the blob.
        if offset and response.status != 206:
            offset = 0
        mode = "ab" if offset else "wb"
        with response, open(partial, mode) as out:
            shutil.copyfileobj(response, out, CHUNK)
        done = os.path.getsize(partial)
        rate = (done - offset) / max(time.monotonic() - started, 1e-9)
        say(f"  fetched {label}  {human(done)}  ({human(rate)}/s)")

    size = os.path.getsize(partial)
    if size != expect_bytes:
        raise IOError(f"{label}: got {size} bytes, manifest says {expect_bytes}")
    actual = sha256_file(partial)
    if actual != expect_sha:
        os.remove(partial)
        raise IOError(f"{label}: sha256 {actual} != {expect_sha} (removed)")
    os.replace(partial, blob)


def link_into_snapshot(blob: str, target: str) -> None:
    """Point `target` at `blob` with a relative symlink, as the hub cache does."""
    os.makedirs(os.path.dirname(target), exist_ok=True)
    relative = os.path.relpath(blob, os.path.dirname(target))
    if os.path.islink(target):
        if os.readlink(target) == relative:
            return
        os.remove(target)
    elif os.path.exists(target):
        os.remove(target)
    os.symlink(relative, target)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--hf-home",
                        default=os.environ.get("HF_HOME",
                                               os.path.expanduser("~/.cache/huggingface")))
    parser.add_argument("--workers", type=int, default=4,
                        help="parallel file downloads (the Hub throttles above ~8)")
    parser.add_argument("--verify-only", action="store_true",
                        help="re-hash what is on disk; never touch the network")
    args = parser.parse_args()

    manifest = json.load(open(args.manifest))
    repo, revision = manifest["repo"], manifest["revision"]
    org, name = repo.split("/", 1)
    root = os.path.join(args.hf_home, "hub", f"models--{org}--{name}")
    blobs = os.path.join(root, "blobs")
    snapshot = os.path.join(root, "snapshots", revision)
    token = os.environ.get("HF_TOKEN", "")

    say(f"repo      : {repo}")
    say(f"revision  : {revision}")
    say(f"cache     : {root}")
    say(f"total     : {human(manifest['total_bytes'])} in {len(manifest['files'])} files")

    if args.verify_only:
        failures = []
        for item in manifest["files"]:
            path = os.path.join(snapshot, item["path"])
            if not os.path.exists(path):
                failures.append(f"{item['path']}: missing")
                continue
            size = os.path.getsize(path)
            if size != item["bytes"]:
                failures.append(f"{item['path']}: {size} bytes != {item['bytes']}")
                continue
            actual = sha256_file(path)
            if actual != item["sha256"]:
                failures.append(f"{item['path']}: sha256 mismatch")
            else:
                say(f"  OK   {item['path']}")
        if failures:
            for line in failures:
                say(f"  FAIL {line}")
            return 1
        say("\nall files verified")
        return 0

    os.makedirs(blobs, exist_ok=True)
    os.makedirs(snapshot, exist_ok=True)

    # Refuse to start a 98 GiB transfer that cannot finish. Count only what is
    # still missing, so a resumed run is not blocked by the full figure.
    remaining = sum(item["bytes"] for item in manifest["files"]
                    if not os.path.exists(os.path.join(blobs, item["etag"])))
    free = shutil.disk_usage(blobs).free
    reserve = manifest.get("reserve_bytes", 0)
    say(f"remaining : {human(remaining)}  (free {human(free)}, reserve {human(reserve)})")
    if free < remaining + reserve:
        say(f"error: need {human(remaining + reserve)} free, have {human(free)}")
        return 1

    errors = []

    def one(item):
        blob = os.path.join(blobs, item["etag"])
        url = f"https://huggingface.co/{repo}/resolve/{revision}/{item['path']}"
        try:
            fetch_blob(url, blob, item["bytes"], item["sha256"], token, item["path"])
            link_into_snapshot(blob, os.path.join(snapshot, item["path"]))
        except (urllib.error.URLError, IOError, OSError) as exc:
            errors.append(f"{item['path']}: {exc}")

    # Largest first: the tail of a parallel download is one straggler shard, and
    # starting the big ones early keeps every worker busy to the end.
    ordered = sorted(manifest["files"], key=lambda f: -f["bytes"])
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(one, ordered))

    if errors:
        say("\nfailed:")
        for line in errors:
            say(f"  {line}")
        return 1

    # refs/main is how huggingface_hub resolves a bare repo id offline, which is
    # what start.sh relies on when it passes MODEL_ID to vLLM.
    refs = os.path.join(root, "refs")
    os.makedirs(refs, exist_ok=True)
    with open(os.path.join(refs, "main"), "w") as handle:
        handle.write(revision)

    say(f"\ncomplete: {snapshot}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
