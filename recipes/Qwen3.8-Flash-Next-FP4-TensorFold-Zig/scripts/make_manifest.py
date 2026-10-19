#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Write a pinned manifest for a Hugging Face repo revision, from the Hub's metadata only (no weights).

Each file gets its size and `blob`, the name huggingface_hub gives it in the cache: the SHA-256 for an
LFS file (also recorded as `sha256`), the git blob id for a small one. download_snapshot.py verifies both.

Usage: ./scripts/make_manifest.py REPO REVISION OUT.json [quantization label]
"""
import json
import sys
import urllib.request

repo, revision, out = sys.argv[1:4]
label = sys.argv[4] if len(sys.argv) > 4 else ""
url = f"https://huggingface.co/api/models/{repo}/tree/{revision}?recursive=true&expand=true"
files, total = [], 0
for f in sorted(json.load(urllib.request.urlopen(url)), key=lambda f: f["path"]):
    if f["type"] != "file":
        continue
    lfs = f.get("lfs")
    size = lfs["size"] if lfs else f["size"]
    entry = {"path": f["path"], "bytes": size, "blob": lfs["oid"] if lfs else f["oid"]}
    if lfs:
        entry["sha256"] = lfs["oid"]
    files.append(entry)
    total += size
doc = {"schema_version": 1, "repo": repo, "revision": revision}
if label:
    doc["quantization"] = label
doc.update(total_bytes=total, files=files)
with open(out, "w") as fh:
    json.dump(doc, fh, indent=2)
    fh.write("\n")
print(f"{repo} @ {revision[:12]}: {len(files)} files, {total} bytes ({total / 2**30:.2f} GiB)")
