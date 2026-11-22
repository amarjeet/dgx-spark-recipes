#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Write a pinned manifest for a Hugging Face repo revision, from the Hub's metadata only (no weights).

Each file gets its size and `blob`, the name huggingface_hub gives it in the cache: the SHA-256 for an
LFS file (also recorded as `sha256`), the git blob id for a small one. download_snapshot.py verifies both.

Usage: ./scripts/make_manifest.py REPO REVISION OUT.json [quantization label] [--include PATH ...]

--include keeps only the named files. HF_TOKEN, when set, is sent: a gated repo hides its LFS
SHA-256s from anyone whose account has not accepted its terms, and this refuses such a listing.
"""
import json
import os
import sys
import urllib.request

argv, include = [], []
args = iter(sys.argv[1:])
for a in args:
    if a == "--include":
        include.append(next(args))
    else:
        argv.append(a)
if len(argv) < 3:
    sys.exit(__doc__)
repo, revision, out = argv[:3]
label = argv[3] if len(argv) > 3 else ""
url = f"https://huggingface.co/api/models/{repo}/tree/{revision}?recursive=true&expand=true"
headers = {"User-Agent": "dgx-spark-recipes/1"}
if os.environ.get("HF_TOKEN"):
    headers["Authorization"] = f"Bearer {os.environ['HF_TOKEN']}"
files, total = [], 0
for f in sorted(json.load(urllib.request.urlopen(urllib.request.Request(url, headers=headers))), key=lambda f: f["path"]):
    if f["type"] != "file" or (include and f["path"] not in include):
        continue
    lfs = f.get("lfs")
    size = lfs["size"] if lfs else f["size"]
    entry = {"path": f["path"], "bytes": size, "blob": lfs["oid"] if lfs else f["oid"]}
    if lfs:
        if not all(c in "0123456789abcdef" for c in lfs["oid"]) or len(lfs["oid"]) != 64:
            sys.exit(f"{repo}: the Hub withholds the SHA-256 of {f['path']} (gated: accept its terms with the account of HF_TOKEN)")
        entry["sha256"] = lfs["oid"]
    files.append(entry)
    total += size
missing = sorted(set(include) - {e["path"] for e in files})
if missing:
    sys.exit(f"{repo} @ {revision[:12]} has no {', '.join(missing)}")
doc = {"schema_version": 1, "repo": repo, "revision": revision}
if label:
    doc["quantization"] = label
doc.update(total_bytes=total, files=files)
with open(out, "w") as fh:
    json.dump(doc, fh, indent=2)
    fh.write("\n")
print(f"{repo} @ {revision[:12]}: {len(files)} files, {total} bytes ({total / 2**30:.2f} GiB)")
