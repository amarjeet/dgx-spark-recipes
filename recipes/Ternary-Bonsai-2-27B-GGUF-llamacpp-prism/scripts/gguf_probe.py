#!/usr/bin/env python3
"""Read a GGUF file's metadata header, with stdlib only.

Exists for one preflight check that cannot be done any other way: proving a
file really is a rotated Bonsai pack before a load is attempted.

The failure this guards against is not loud. The PrismML fork's PQ2_0 and
PTQ1_0 type ids sit past upstream's GGML_TYPE_COUNT, so stock llama.cpp
refuses those two outright -- fine. But the same family also ships a plain
Q2_0 band whose type id upstream already knows, and upstream loads it without
a warning and emits fluent nonsense, because it has no Hadamard activation
runtime. The only honest discriminator is the prism.hadamard.* metadata, which
declares the rotation the weights are stored in.

It also prints the architecture numbers the recipe's memory budget is derived
from, so those can be checked against the file rather than trusted from a
comment.

Usage:
  ./scripts/gguf_probe.py MODEL.gguf                     # summary
  ./scripts/gguf_probe.py MODEL.gguf --json              # all scalar keys
  ./scripts/gguf_probe.py MODEL.gguf --require-hadamard  # exit 1 if absent
  ./scripts/gguf_probe.py MODEL.gguf --key qwen35.block_count
"""
import argparse
import json
import struct
import sys
from pathlib import Path

# GGUF scalar type ids -> (struct format, size). Types 8 (string) and 9 (array)
# are handled separately.
SCALARS = {
    0: ("<B", 1), 1: ("<b", 1), 2: ("<H", 2), 3: ("<h", 2),
    4: ("<I", 4), 5: ("<i", 4), 6: ("<f", 4), 7: ("<?", 1),
    10: ("<Q", 8), 11: ("<q", 8), 12: ("<d", 8),
}
TYPE_STRING = 8
TYPE_ARRAY = 9

# Arrays longer than this are summarised rather than materialised: the token
# list alone is a quarter of a million strings.
ARRAY_INLINE_LIMIT = 16


class Reader:
    def __init__(self, handle):
        self.handle = handle

    def take(self, count: int) -> bytes:
        data = self.handle.read(count)
        if len(data) != count:
            raise ValueError("unexpected end of file while reading metadata")
        return data

    def scalar(self, type_id: int):
        fmt, size = SCALARS[type_id]
        return struct.unpack(fmt, self.take(size))[0]

    def string(self) -> str:
        (length,) = struct.unpack("<Q", self.take(8))
        return self.take(length).decode("utf-8", "replace")

    def value(self, type_id: int):
        if type_id == TYPE_STRING:
            return self.string()
        if type_id == TYPE_ARRAY:
            (elem_type,) = struct.unpack("<I", self.take(4))
            (count,) = struct.unpack("<Q", self.take(8))
            if count <= ARRAY_INLINE_LIMIT:
                return [self.value(elem_type) for _ in range(count)]
            # Skip the payload. Fixed-width elements can be skipped by
            # arithmetic; strings have to be walked.
            if elem_type in SCALARS:
                self.handle.seek(SCALARS[elem_type][1] * count, 1)
            elif elem_type == TYPE_STRING:
                for _ in range(count):
                    self.string()
            else:
                raise ValueError(f"cannot skip array of type {elem_type}")
            return {"_array": True, "type": elem_type, "len": count}
        if type_id in SCALARS:
            return self.scalar(type_id)
        raise ValueError(f"unknown GGUF value type {type_id}")


def read_metadata(path: Path) -> dict:
    with path.open("rb") as handle:
        reader = Reader(handle)
        if reader.take(4) != b"GGUF":
            raise ValueError(f"not a GGUF file: {path}")
        version, = struct.unpack("<I", reader.take(4))
        n_tensors, = struct.unpack("<Q", reader.take(8))
        n_kv, = struct.unpack("<Q", reader.take(8))
        meta = {}
        for _ in range(n_kv):
            key = reader.string()
            type_id, = struct.unpack("<I", reader.take(4))
            meta[key] = reader.value(type_id)
    meta["_gguf_version"] = version
    meta["_n_tensors"] = n_tensors
    meta["_n_kv"] = n_kv
    return meta


def kv_bytes_per_token(meta: dict) -> int | None:
    """KV cache cost per token at F16, from the file's own numbers.

    Only the full-attention layers hold a growing cache; the linear-attention
    layers keep a fixed-size recurrent state instead. This is the arithmetic
    behind the recipe's 64 KiB/token, restated against the actual file so a
    different checkpoint cannot quietly invalidate it.
    """
    arch = meta.get("general.architecture")
    if not arch:
        return None
    try:
        blocks = meta[f"{arch}.block_count"]
        interval = meta[f"{arch}.full_attention_interval"]
        heads_kv = meta[f"{arch}.attention.head_count_kv"]
        key_len = meta[f"{arch}.attention.key_length"]
        value_len = meta[f"{arch}.attention.value_length"]
    except KeyError:
        return None
    full_layers = blocks // interval
    return full_layers * heads_kv * (key_len + value_len) * 2


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("model", type=Path)
    parser.add_argument("--json", action="store_true",
                        help="dump every scalar metadata key as JSON")
    parser.add_argument("--key", action="append", default=[],
                        help="print one key's value (repeatable)")
    parser.add_argument("--require-hadamard", action="store_true",
                        help="exit non-zero unless prism.hadamard.* is present")
    args = parser.parse_args()

    if not args.model.is_file():
        print(f"no such file: {args.model}", file=sys.stderr)
        return 1
    try:
        meta = read_metadata(args.model)
    except ValueError as exc:
        print(f"{args.model}: {exc}", file=sys.stderr)
        return 1

    if args.key:
        missing = False
        for key in args.key:
            if key in meta:
                print(meta[key])
            else:
                print(f"{key}: absent", file=sys.stderr)
                missing = True
        return 1 if missing else 0

    if args.json:
        printable = {k: v for k, v in meta.items() if not isinstance(v, dict)}
        print(json.dumps(printable, indent=2, sort_keys=True, default=str))
        return 0

    if args.require_hadamard:
        version = meta.get("prism.hadamard.version")
        if version is None:
            print(f"{args.model.name}: no prism.hadamard.* metadata -- this is not a "
                  "rotated Bonsai pack", file=sys.stderr)
            return 1
        print(f"prism.hadamard.version={version} "
              f"transform={meta.get('prism.hadamard.transform')} "
              f"block_size={meta.get('prism.hadamard.block_size')}")
        return 0

    arch = meta.get("general.architecture", "?")
    print(f"file          : {args.model.name}")
    print(f"architecture  : {arch}")
    print(f"file_type     : {meta.get('general.file_type')}")
    print(f"tensors       : {meta.get('_n_tensors')}")
    print(f"n_ctx_train   : {meta.get(f'{arch}.context_length')}")
    print(f"blocks        : {meta.get(f'{arch}.block_count')}")
    print(f"full_attn_int : {meta.get(f'{arch}.full_attention_interval')}")
    print(f"head_count_kv : {meta.get(f'{arch}.attention.head_count_kv')}")
    print(f"key/value_len : {meta.get(f'{arch}.attention.key_length')}"
          f"/{meta.get(f'{arch}.attention.value_length')}")
    per_token = kv_bytes_per_token(meta)
    if per_token:
        print(f"kv per token  : {per_token} B ({per_token / 1024:.0f} KiB) at F16")
    print(f"hadamard      : {meta.get('prism.hadamard.transform', 'ABSENT')}")
    print(f"chat template : {'present' if 'tokenizer.chat_template' in meta else 'ABSENT'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
