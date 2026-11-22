#!/usr/bin/env python3
"""Write a pinned manifest for a Hugging Face repo revision.

Reads only the Hub's public metadata API (no weights): every file's size and
SHA-256 at one exact commit (the LFS oid, or hashed here for the few small
regular files).
download_snapshot.py consumes the result.

Usage: ./scripts/make_manifest.py REPO REVISION OUT.json
"""
import hashlib
import json
import sys
import urllib.request

repo, revision, out = sys.argv[1:4]
url = f"https://huggingface.co/api/models/{repo}/tree/{revision}?recursive=1"


def sha256_of(path):
    """Regular (non-LFS) files are small; the Hub gives no sha256, so hash them."""
    with urllib.request.urlopen(f"https://huggingface.co/{repo}/resolve/{revision}/{path}") as r:
        return hashlib.sha256(r.read()).hexdigest()


files, total = [], 0
for f in sorted(json.load(urllib.request.urlopen(url)), key=lambda f: f["path"]):
    if f["type"] != "file":
        continue
    lfs = f.get("lfs")
    size = lfs["size"] if lfs else f["size"]
    files.append({
        "path": f["path"],
        "bytes": size,
        # An LFS object's name in the hub cache is its sha256; a regular file's
        # is its git blob sha1. Same names huggingface_hub would choose.
        "sha256": lfs["oid"] if lfs else sha256_of(f["path"]),
        "etag": lfs["oid"] if lfs else f["oid"],
        "lfs": bool(lfs),
    })
    total += size
json.dump({"schema_version": 1, "repo": repo, "revision": revision,
           "total_bytes": total, "reserve_bytes": 10 * 2**30, "files": files},
          open(out, "w"), indent=2)
print(f"{repo} @ {revision[:12]}: {len(files)} files, {total} bytes")
