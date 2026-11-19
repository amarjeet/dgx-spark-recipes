#!/usr/bin/env python3
"""For each hard task: doing nothing must fail, and the reference fix must pass."""
import os
import sys
import tempfile

sys.dont_write_bytecode = True  # keep __pycache__ out of the recipe directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import harness  # noqa: E402
from tasks import TASKS  # noqa: E402
from tasks_hard import SOLUTIONS  # noqa: E402


def materialise(t, root, solve):
    for rel, c in {**t["files"], **t["hidden"]}.items():
        p = os.path.join(root, rel)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        open(p, "wb" if isinstance(c, bytes) else "w").write(c)
    if not solve:
        return
    for rel, patches in SOLUTIONS[t["name"]].items():
        p = os.path.join(root, rel)
        if isinstance(patches, tuple):
            patches = [patches]
        for old, new in patches:
            if old == "__WRITE__":
                open(p, "w").write(new)
                continue
            text = open(p).read()
            assert text.count(old) == 1, f"{t['name']}: patch target not unique in {rel}: {old[:40]!r}"
            open(p, "w").write(text.replace(old, new))


def main():
    base = os.path.join(harness.STATE_DIR, "pi-eval", "validate")
    os.makedirs(base, exist_ok=True)
    ok = True
    for t in [t for t in TASKS if t["suite"] == "hard"]:
        res = []
        for solve in (False, True):
            root = tempfile.mkdtemp(dir=base)
            materialise(t, root, solve)
            sb = harness.Sandbox(root)
            out, err = sb.bash(t["verify"], 180)
            sb.close()
            res.append((err, out))
        good = res[0][0] and not res[1][0]
        ok &= good
        print(f"{t['name']:20s} no-op fails: {res[0][0]!s:5s} reference passes: {not res[1][0]!s:5s} {'OK' if good else 'BROKEN'}")
        if not good:
            print("   no-op tail:", res[0][1][-300:].replace("\n", " | "))
            print("   reference tail:", res[1][1][-300:].replace("\n", " | "))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
