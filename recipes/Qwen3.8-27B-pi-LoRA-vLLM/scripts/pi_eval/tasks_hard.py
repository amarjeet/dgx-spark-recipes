"""Harder pi-style tasks. Each is built so a first attempt, or a fix made without
running anything, tends to fail. SOLUTIONS holds a reference fix per task, used
only to prove each task is solvable (validate.py)."""
import csv
import io
import random
from decimal import Decimal

SOLUTIONS = {}


def register(task, d):
    H = "hard"

    # 1. The traceback points at report.py; the bug is in the parser.
    task(
        "misleading_keyerror",
        "tests/test_report.py fails with a KeyError. Fix the underlying cause rather than working around it.",
        {
            "kvparse.py": d('''
                def parse_line(line):
                    """Parse 'Key Name: value' into (key, value).

                    Keys are normalised to snake_case: lowercase, and spaces and dashes become
                    underscores. Values are stripped. Only the first ':' separates key from value.
                    """
                    key, _, value = line.partition(":")
                    key = key.strip().lower().replace(" ", "_")
                    return key, value.strip()


                def parse(text):
                    return dict(parse_line(l) for l in text.splitlines() if l.strip() and not l.startswith("#"))
                '''),
            "report.py": d('''
                from kvparse import parse


                def summary(text):
                    d = parse(text)
                    return f"{d['host_name']} up {d['up_time']}s, load {d['load_avg']}"
                '''),
            "tests/test_report.py": d('''
                from report import summary

                def test_summary():
                    text = "Host Name: web1\\nUp-Time: 3600\\nLoad-Avg: 0.5\\n"
                    assert summary(text) == "web1 up 3600s, load 0.5"
                '''),
        },
        {"hidden_tests/test_h.py": d('''
            from kvparse import parse
            from report import summary
            def test_parser():
                assert parse("X-Forwarded-For: a\\nUrl: http://x:80/y\\n") == {"x_forwarded_for": "a", "url": "http://x:80/y"}
            def test_report():
                assert summary("Host Name: db\\nUp-Time: 5\\nLoad-Avg: 1.25") == "db up 5s, load 1.25"
            ''')},
        "python -m pytest -q tests hidden_tests", H)
    SOLUTIONS["misleading_keyerror"] = {"kvparse.py": ('.replace(" ", "_")', '.replace(" ", "_").replace("-", "_")')}

    # 2. Three independent bugs; fixing the obvious one is not enough.
    task(
        "three_bugs",
        "tests/test_stats.py is failing. Make all the tests pass without changing the tests.",
        {
            "stats.py": d('''
                import math


                def mean(xs):
                    return sum(xs) / len(xs)


                def median(xs):
                    s = sorted(xs)
                    n = len(s)
                    return s[n // 2]


                def sample_variance(xs):
                    m = mean(xs)
                    return sum((x - m) ** 2 for x in xs) / len(xs)


                def percentile(xs, p):
                    """Nearest-rank percentile: the smallest value such that at least p% of the values are <= it."""
                    s = sorted(xs)
                    k = int(len(s) * p / 100)
                    return s[k]
                '''),
            "tests/test_stats.py": d('''
                import pytest
                from stats import mean, median, sample_variance, percentile

                def test_mean():
                    assert mean([1, 2, 3, 4]) == 2.5

                def test_median():
                    assert median([3, 1, 2]) == 2
                    assert median([4, 1, 3, 2]) == 2.5

                def test_variance():
                    assert sample_variance([2, 4, 4, 4, 5, 5, 7, 9]) == pytest.approx(32 / 7)

                def test_percentile():
                    xs = [15, 20, 35, 40, 50]
                    assert [percentile(xs, p) for p in (5, 30, 40, 50, 100)] == [15, 20, 20, 35, 50]
                '''),
        },
        {"hidden_tests/test_h.py": d('''
            import pytest
            from stats import median, sample_variance, percentile
            def test_more():
                assert median([5]) == 5
                assert median([1, 2, 3, 4, 5, 6]) == 3.5
                assert sample_variance([1, 2, 3, 4]) == pytest.approx(5 / 3)
                assert percentile(list(range(1, 11)), 90) == 9
                assert percentile(list(range(1, 11)), 91) == 10
                assert percentile([7], 1) == 7
            ''')},
        "python -m pytest -q tests hidden_tests", H)
    SOLUTIONS["three_bugs"] = {"stats.py": [
        ("    return s[n // 2]\n", "    if n % 2:\n        return s[n // 2]\n    return (s[n // 2 - 1] + s[n // 2]) / 2\n"),
        ("/ len(xs)\n\n\ndef percentile", "/ (len(xs) - 1)\n\n\ndef percentile"),
        ("    k = int(len(s) * p / 100)\n", "    k = max(math.ceil(len(s) * p / 100) - 1, 0)\n"),
    ]}

    # 3. Circular import.
    task(
        "circular_import",
        "Importing the shapes package fails. Fix it so `python -m pytest -q` passes, keeping the registry's type check.",
        {
            "shapes/__init__.py": "",
            "shapes/base.py": d('''
                from shapes.registry import register


                class Shape:
                    def area(self):
                        raise NotImplementedError


                @register
                class Unit(Shape):
                    def area(self):
                        return 1.0
                '''),
            "shapes/registry.py": d('''
                from shapes.base import Shape

                REGISTRY = {}


                def register(cls):
                    if not issubclass(cls, Shape):
                        raise TypeError(f"{cls.__name__} is not a Shape")
                    REGISTRY[cls.__name__] = cls
                    return cls
                '''),
            "tests/test_shapes.py": d('''
                from shapes.base import Unit
                from shapes.registry import REGISTRY

                def test_registered():
                    assert REGISTRY["Unit"] is Unit
                    assert Unit().area() == 1.0
                '''),
        },
        {"hidden_tests/check.sh": d('''
            set -e
            python -m pytest -q tests
            python -c 'import shapes.registry as r; import shapes.base as b; assert r.REGISTRY["Unit"] is b.Unit'
            python -c 'import shapes.base as b; import shapes.registry as r; assert r.REGISTRY["Unit"] is b.Unit'
            python -c '
            from shapes.registry import register
            try:
                register(int)
            except TypeError:
                pass
            else:
                raise SystemExit("type check lost")'
            ''')},
        "bash hidden_tests/check.sh", H)
    SOLUTIONS["circular_import"] = {"shapes/registry.py": [
        ("from shapes.base import Shape\n\n", ""),
        ("def register(cls):\n", "def register(cls):\n    from shapes.base import Shape\n\n"),
    ]}

    # 4. Performance, with an order trap for the obvious Counter rewrite.
    task(
        "slow_duplicates",
        "tests/test_dupes.py takes far too long. Make find_duplicates fast without changing its behaviour.",
        {
            "dupes.py": d('''
                def find_duplicates(items):
                    """Return each value that occurs more than once, once, ordered by where it first repeats."""
                    out = []
                    for i, x in enumerate(items):
                        if x in items[:i] and x not in out:
                            out.append(x)
                    return out
                '''),
            "tests/test_dupes.py": d('''
                import time
                from dupes import find_duplicates

                def test_basic():
                    assert find_duplicates([1, 2, 1, 3, 2, 1]) == [1, 2]

                def test_fast():
                    xs = list(range(60000)) * 2
                    t = time.perf_counter()
                    assert find_duplicates(xs) == list(range(60000))
                    assert time.perf_counter() - t < 2
                '''),
        },
        {"hidden_tests/test_h.py": d('''
            import time
            from dupes import find_duplicates
            def test_order():
                assert find_duplicates(["b", "a", "a", "b"]) == ["a", "b"]
                assert find_duplicates([3, 1, 3, 2, 1, 1, 2]) == [3, 1, 2]
                assert find_duplicates([]) == []
            def test_fast():
                xs = [i % 50000 for i in range(200000)]
                t = time.perf_counter()
                assert len(find_duplicates(xs)) == 50000
                assert time.perf_counter() - t < 2
            ''')},
        "timeout 120 python -m pytest -q tests hidden_tests", H)
    SOLUTIONS["slow_duplicates"] = {"dupes.py": [
        ("    out = []\n    for i, x in enumerate(items):\n        if x in items[:i] and x not in out:\n            out.append(x)\n",
         "    seen, added, out = set(), set(), []\n    for x in items:\n        if x in seen and x not in added:\n            added.add(x)\n            out.append(x)\n        seen.add(x)\n"),
    ]}

    # 5. Two separate sources of shared state.
    task(
        "shared_state",
        "Customers report seeing items and discount codes from other people's carts. Find and fix the cause in cart.py.",
        {
            "cart.py": d('''
                class Cart:
                    items = []

                    def __init__(self, owner, discounts=[]):
                        self.owner = owner
                        self.discounts = discounts

                    def add(self, sku, qty=1):
                        self.items.append((sku, qty))

                    def add_discount(self, code):
                        self.discounts.append(code)

                    def summary(self):
                        return {"owner": self.owner, "items": list(self.items), "discounts": list(self.discounts)}
                '''),
        },
        {"hidden_tests/test_h.py": d('''
            from cart import Cart
            def test_isolated():
                a, b = Cart("a"), Cart("b")
                a.add("x"); a.add_discount("SAVE")
                assert b.summary() == {"owner": "b", "items": [], "discounts": []}
                assert a.summary() == {"owner": "a", "items": [("x", 1)], "discounts": ["SAVE"]}
                c = Cart("c", discounts=["VIP"])
                assert c.summary()["discounts"] == ["VIP"]
                assert Cart("d").summary()["items"] == []
            ''')},
        "python -m pytest -q hidden_tests", H)
    SOLUTIONS["shared_state"] = {"cart.py": [
        ("    items = []\n\n", ""),
        ("discounts=[]):\n        self.owner = owner\n        self.discounts = discounts\n",
         "discounts=None):\n        self.owner = owner\n        self.items = []\n        self.discounts = list(discounts or [])\n"),
    ]}

    # 6. Mixed encodings; forcing latin-1 breaks a UTF-8-only name.
    task(
        "encoding_fallback",
        "`load_names('data/names_latin1.txt')` crashes with UnicodeDecodeError. Make load_names in names.py "
        "read both UTF-8 and Latin-1 files correctly.",
        {
            "names.py": d('''
                def load_names(path):
                    with open(path) as f:
                        return [line.strip() for line in f if line.strip()]
                '''),
            "data/names_utf8.txt": "José\nZoë\n".encode("utf-8"),
            "data/names_latin1.txt": "José\nZoë\n".encode("latin-1"),
        },
        {
            "hidden_tests/test_h.py": d('''
                from names import load_names
                def test_both():
                    assert load_names("data/names_utf8.txt") == ["José", "Zoë"]
                    assert load_names("data/names_latin1.txt") == ["José", "Zoë"]
                    assert load_names("hidden_tests/names_more.txt") == ["Łukasz", "Søren", "日本"]
                '''),
            "hidden_tests/names_more.txt": "Łukasz\nSøren\n日本\n".encode("utf-8"),
        },
        "python -m pytest -q hidden_tests", H)
    SOLUTIONS["encoding_fallback"] = {"names.py": [
        ("    with open(path) as f:\n        return [line.strip() for line in f if line.strip()]\n",
         "    raw = open(path, 'rb').read()\n    try:\n        text = raw.decode('utf-8')\n    except UnicodeDecodeError:\n        text = raw.decode('latin-1')\n    return [l.strip() for l in text.splitlines() if l.strip()]\n"),
    ]}

    # 7. Data report: duplicates, refunds, quoted thousands separators, ties.
    rng = random.Random(7)
    customers = [f"C{i:03d}" for i in range(1, 13)]
    rows, seen = [], []
    for i in range(160):
        txn = f"T{i:04d}"
        cust = rng.choice(customers)
        amt = Decimal(rng.randrange(500, 250000)) / 100
        typ = "refund" if rng.random() < 0.2 else "charge"
        rows.append((txn, cust, amt, typ))
        seen.append((txn, cust, amt, typ))
        if rng.random() < 0.15:
            rows.append((txn, cust, amt, typ))  # exact duplicate row
    # Force a tie for third place, broken by customer id.
    net = {}
    for txn, cust, amt, typ in seen:
        net[cust] = net.get(cust, Decimal(0)) + (amt if typ == "charge" else -amt)
    ranked = sorted(net.items(), key=lambda kv: (-kv[1], kv[0]))
    third, fourth = ranked[2], ranked[3]
    gap = third[1] - fourth[1]
    if gap:
        rows.append((f"T9999", fourth[0], gap, "charge"))
        net[fourth[0]] += gap
    ranked = sorted(net.items(), key=lambda kv: (-kv[1], kv[0]))
    expected = "\n".join(f"{c},{v:.2f}" for c, v in ranked[:3])
    rng.shuffle(rows)
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(["txn_id", "customer", "amount", "type"])
    for txn, cust, amt, typ in rows:
        a = f"{amt:,.2f}"
        w.writerow([txn, cust, a, typ])
    task(
        "dedupe_report",
        "data/transactions.csv lists transactions. Some rows are duplicated (same txn_id); count each txn_id once. "
        "A customer's net total is charges minus refunds. Write the top 3 customers by net total (highest first; "
        "break ties by customer id ascending) to answer.csv as `customer,net` lines with exactly 2 decimals and no header.",
        {"data/transactions.csv": buf.getvalue()},
        {"hidden_tests/check.sh": f"test \"$(sed -e 's/\\r$//' answer.csv | sed -e '/^$/d')\" = \"$(printf '{expected.replace(chr(10), chr(92) + 'n')}')\"\n"},
        "bash hidden_tests/check.sh", H)
    SOLUTIONS["dedupe_report"] = {"answer.csv": ("__WRITE__", expected + "\n")}

    # 8. .env parser with several edge cases.
    task(
        "env_parser",
        "parse_env in envparse.py does not follow its docstring and tests/test_envparse.py fails. Fix parse_env.",
        {
            "envparse.py": d('''
                def parse_env(text):
                    """Parse .env text into a dict.

                    - blank lines and lines whose first non-space character is '#' are ignored
                    - an optional leading `export ` is ignored
                    - KEY=VALUE; VALUE may itself contain '='
                    - whitespace around KEY and around an unquoted VALUE is stripped
                    - VALUE may be wrapped in matching single or double quotes, which are removed;
                      a quoted VALUE is kept exactly, including any '#'
                    - an unquoted VALUE ends at an inline comment that starts with ' #'
                    """
                    out = {}
                    for line in text.splitlines():
                        if not line or line.startswith("#"):
                            continue
                        key, value = line.split("=")
                        out[key] = value.strip('"')
                    return out
                '''),
            "tests/test_envparse.py": d('''
                from envparse import parse_env

                def test_basic():
                    assert parse_env("A=1\\n\\n# c\\nB=two\\n") == {"A": "1", "B": "two"}

                def test_equals_in_value():
                    assert parse_env("URL=postgres://u:p@h/db?x=1") == {"URL": "postgres://u:p@h/db?x=1"}

                def test_export_and_spaces():
                    assert parse_env("export  K = v  ") == {"K": "v"}
                '''),
        },
        {"hidden_tests/test_h.py": d('''
            from envparse import parse_env
            def test_all():
                text = "\\n".join([
                    "   # indented comment",
                    "export A=1",
                    "URL=postgres://u:p@h/db?x=1",
                    "S='a # not a comment'",
                    'D="x=y"',
                    "B=2 # trailing comment",
                    "H=abc#def",
                    "EMPTY=",
                    "  SP = val  ",
                    "Q='  padded  '",
                ])
                assert parse_env(text) == {
                    "A": "1", "URL": "postgres://u:p@h/db?x=1", "S": "a # not a comment",
                    "D": "x=y", "B": "2", "H": "abc#def", "EMPTY": "", "SP": "val", "Q": "  padded  ",
                }
            ''')},
        "python -m pytest -q tests hidden_tests", H)
    SOLUTIONS["env_parser"] = {"envparse.py": [
        ("""    out = {}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        key, value = line.split("=")
        out[key] = value.strip('"')
    return out
""", """    out = {}
    for line in text.splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        if s.startswith("export "):
            s = s[len("export "):]
        key, _, value = s.partition("=")
        key, value = key.strip(), value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\\"":
            value = value[1:-1]
        else:
            i = value.find(" #")
            if i >= 0:
                value = value[:i].rstrip()
        out[key] = value
    return out
"""),
    ]}
