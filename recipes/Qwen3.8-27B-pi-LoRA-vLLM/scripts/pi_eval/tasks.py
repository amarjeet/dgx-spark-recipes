"""Synthetic pi-style tasks. Each task: files (the repo the agent sees), a prompt,
hidden files copied in only after the agent stops, and a verify command run in
the sandbox (exit 0 = solved)."""
import textwrap


def d(s: str) -> str:
    return textwrap.dedent(s).lstrip("\n")


TASKS = []


def task(name, prompt, files, hidden, verify, suite="easy"):
    TASKS.append(dict(name=name, prompt=prompt, files=files, hidden=hidden, verify=verify, suite=suite))


# 1. ImportError from a typo'd module path
task(
    "import_error",
    "The test suite fails with an ImportError. Fix it so `python -m pytest -q` passes.",
    {
        "inventory/__init__.py": "",
        "inventory/models.py": d('''
            from dataclasses import dataclass

            @dataclass
            class Item:
                sku: str
                qty: int
                price: float
            '''),
        "inventory/store.py": d('''
            from inventory.model import Item


            class Store:
                def __init__(self):
                    self.items = {}

                def add(self, sku, qty, price):
                    if sku in self.items:
                        self.items[sku].qty += qty
                    else:
                        self.items[sku] = Item(sku, qty, price)

                def value(self):
                    return sum(i.qty * i.price for i in self.items.values())
            '''),
        "tests/test_store.py": d('''
            from inventory.store import Store

            def test_value():
                s = Store()
                s.add("a", 2, 1.5)
                s.add("a", 1, 1.5)
                s.add("b", 1, 10.0)
                assert s.value() == 14.5
            '''),
    },
    {"hidden_tests/test_h.py": d('''
        from inventory.store import Store
        from inventory.models import Item
        def test_item_type():
            s = Store(); s.add("x", 1, 2.0)
            assert isinstance(s.items["x"], Item)
        ''')},
    "python -m pytest -q tests hidden_tests",
)

# 2. off-by-one in 1-indexed pagination
task(
    "paginate_off_by_one",
    "Users report that page 1 from `paginate()` in pager.py skips the first items. Fix the bug.",
    {
        "pager.py": d('''
            def paginate(items, page, per_page=10):
                """Return the items on `page` (1-indexed), `per_page` items per page."""
                if page < 1:
                    raise ValueError("page must be >= 1")
                start = page * per_page
                return items[start:start + per_page]
            '''),
    },
    {"hidden_tests/test_h.py": d('''
        import pytest
        from pager import paginate
        def test_pages():
            xs = list(range(25))
            assert paginate(xs, 1) == list(range(10))
            assert paginate(xs, 3) == [20, 21, 22, 23, 24]
            assert paginate(xs, 4) == []
            assert paginate(xs, 2, per_page=4) == [4, 5, 6, 7]
        def test_bad_page():
            with pytest.raises(ValueError):
                paginate([1], 0)
        ''')},
    "python -m pytest -q hidden_tests",
)

# 3. implement a function from its docstring
task(
    "implement_slugify",
    "Implement `slugify` in text_utils.py according to its docstring.",
    {
        "text_utils.py": d('''
            def slugify(text: str) -> str:
                """Turn `text` into a URL slug.

                - lowercase
                - accented letters become their ASCII base letter (é -> e)
                - every run of characters that are not a-z or 0-9 becomes a single "-"
                - no leading or trailing "-"

                >>> slugify("  Héllo, World!  ")
                'hello-world'
                """
                raise NotImplementedError
            '''),
    },
    {"hidden_tests/test_h.py": d('''
        from text_utils import slugify
        def test_slug():
            assert slugify("  Héllo, World!  ") == "hello-world"
            assert slugify("Crème Brûlée -- 2024") == "creme-brulee-2024"
            assert slugify("a__b..c") == "a-b-c"
            assert slugify("---") == ""
            assert slugify("ÀÉÎÕÜ ç") == "aeiou-c"
        ''')},
    "python -m pytest -q hidden_tests",
)

# 4. config schema mismatch, constrained fix
task(
    "config_keyerror",
    "`python app.py` crashes with a KeyError. Make it work with the existing config.json "
    "(do not edit config.json). It should print `connecting to db.internal:5432`.",
    {
        "config.json": '{\n  "database": {"host": "db.internal", "port": 5432},\n  "debug": false\n}\n',
        "app.py": d('''
            import json

            def load():
                with open("config.json") as f:
                    return json.load(f)

            def main():
                cfg = load()
                print(f"connecting to {cfg['db_host']}:{cfg['db_port']}")

            if __name__ == "__main__":
                main()
            '''),
    },
    {"hidden_tests/check.sh": d('''
        set -e
        test "$(python app.py)" = "connecting to db.internal:5432"
        python -c 'import json;c=json.load(open("config.json"));assert c=={"database":{"host":"db.internal","port":5432},"debug":False}'
        ''')},
    "bash hidden_tests/check.sh",
)

# 5. rename across files
task(
    "rename_function",
    "Rename the function `calc_total` to `compute_total` everywhere in this project. Tests must still pass.",
    {
        "billing/__init__.py": "",
        "billing/core.py": d('''
            def calc_total(lines, tax=0.0):
                subtotal = sum(q * p for q, p in lines)
                return round(subtotal * (1 + tax), 2)
            '''),
        "billing/invoice.py": d('''
            from billing.core import calc_total

            def invoice_text(name, lines):
                return f"{name}: {calc_total(lines, tax=0.2):.2f}"
            '''),
        "billing/report.py": d('''
            from billing import core

            def daily(batches):
                return [core.calc_total(b) for b in batches]
            '''),
        "tests/test_billing.py": d('''
            from billing.core import calc_total
            from billing.invoice import invoice_text
            from billing.report import daily

            def test_all():
                assert calc_total([(2, 5.0)]) == 10.0
                assert invoice_text("acme", [(1, 10.0)]) == "acme: 12.00"
                assert daily([[(1, 1.0)], [(2, 2.0)]]) == [1.0, 4.0]
            '''),
    },
    {"hidden_tests/check.sh": d('''
        set -e
        if grep -rn --include='*.py' 'calc_total' billing tests; then exit 1; fi
        python -m pytest -q tests
        python -c 'from billing.core import compute_total; assert compute_total([(1,2.0)])==2.0'
        ''')},
    "bash hidden_tests/check.sh",
)

# 6. terminal: find the largest file
_big = {f"data/{a}/{b}/f{i}.bin": "x" * (100 + 37 * i + 11 * len(a + b)) for a in ("raw", "cache") for b in ("x", "y") for i in range(4)}
_big["data/raw/y/archive_2023.bin"] = "y" * 900
_big["data/.hidden/big.tmp"] = "z" * 950
task(
    "largest_file",
    "Find the largest file under data/ by size in bytes, including hidden files and directories. "
    "Write its path relative to the current directory (e.g. data/a/b.txt) into answer.txt, and nothing else.",
    _big,
    {"hidden_tests/check.sh": "test \"$(tr -d '[:space:]' < answer.txt | sed 's#^\\./##')\" = data/.hidden/big.tmp\n"},
    "bash hidden_tests/check.sh",
)

# 7. terminal: structured log counting with decoys
_logs = {}
_n = 0
for k in range(3):
    lines = []
    for i in range(40):
        lvl = ["INFO", "WARN", "ERROR", "DEBUG"][(i * 7 + k) % 4]
        msg = "upstream ERROR ignored" if i % 9 == 0 else ("retry ERRORS=0" if i % 11 == 0 else "ok")
        if lvl == "ERROR":
            _n += 1
        lines.append(f"2024-05-0{k + 1}T10:{i:02d}:00 svc{k} {lvl} {msg}")
    _logs[f"logs/app{k}.log"] = "\n".join(lines) + "\n"
_logs["logs/README.txt"] = "ERROR ERROR ERROR this file is not a log\n"
task(
    "count_errors",
    "Count how many lines across logs/*.log have level ERROR. The level is the third "
    "whitespace-separated field. Write just the number into answer.txt.",
    _logs,
    {"hidden_tests/check.sh": f"test \"$(tr -d '[:space:]' < answer.txt)\" = {_n}\n"},
    "bash hidden_tests/check.sh",
)

# 8. CSV with quoted commas
task(
    "csv_quoted",
    "`load_people()` in people.py breaks on rows whose fields contain commas inside quotes. "
    "Fix it. Ages should be ints.",
    {
        "people.csv": 'name,city,age\n"Smith, Jane","Portland, OR",41\nBob,Austin,29\n"O\'Neil, Pat",Boston,35\n',
        "people.py": d('''
            def load_people(path="people.csv"):
                rows = []
                with open(path) as f:
                    header = f.readline().strip().split(",")
                    for line in f:
                        values = line.strip().split(",")
                        row = dict(zip(header, values))
                        row["age"] = int(row["age"])
                        rows.append(row)
                return rows
            '''),
    },
    {"hidden_tests/test_h.py": d('''
        from people import load_people
        def test_load():
            assert load_people() == [
                {"name": "Smith, Jane", "city": "Portland, OR", "age": 41},
                {"name": "Bob", "city": "Austin", "age": 29},
                {"name": "O'Neil, Pat", "city": "Boston", "age": 35},
            ]
        ''')},
    "python -m pytest -q hidden_tests",
)

# 9. add a CLI flag
task(
    "cli_top_flag",
    "Add a `--top N` option to counter.py. With it, the script prints the N most frequent words as "
    "`word count` lines, sorted by count descending, then word ascending. Without it, keep the current behaviour.",
    {
        "counter.py": d('''
            import argparse
            import re
            from collections import Counter


            def count_words(text):
                return Counter(re.findall(r"[a-z']+", text.lower()))


            def main():
                ap = argparse.ArgumentParser()
                ap.add_argument("path")
                args = ap.parse_args()
                with open(args.path) as f:
                    counts = count_words(f.read())
                print(sum(counts.values()))


            if __name__ == "__main__":
                main()
            '''),
        "sample.txt": "the cat and the hat. The bat and a cat! zebra\n",
    },
    {"hidden_tests/check.sh": d('''
        set -e
        test "$(python counter.py sample.txt)" = 11
        test "$(python counter.py sample.txt --top 3)" = "$(printf 'the 3\\nand 2\\ncat 2')"
        test "$(python counter.py --top 1 sample.txt)" = "the 3"
        ''')},
    "bash hidden_tests/check.sh",
)

# 10. failing test, fix the code not the test
task(
    "fix_code_not_test",
    "One test in tests/ is failing. Fix the code, not the test.",
    {
        "dates.py": d('''
            import re

            _ISO = re.compile(r"^(\\d{4})-(\\d{2})-(\\d{2})$")


            def parse_date(s):
                """Parse YYYY-MM-DD into (year, month, day). Raise ValueError if invalid."""
                m = _ISO.match(s)
                if not m:
                    raise ValueError(f"bad date: {s!r}")
                y, mo, d = map(int, m.groups())
                if not 1 <= mo <= 12 or not 1 <= d <= 31:
                    raise ValueError(f"bad date: {s!r}")
                return y, mo, d
            '''),
        "tests/test_dates.py": d('''
            import pytest
            from dates import parse_date

            def test_ok():
                assert parse_date("2024-02-29") == (2024, 2, 29)

            def test_invalid_day_for_month():
                for s in ["2023-02-29", "2024-02-30", "2024-04-31"]:
                    with pytest.raises(ValueError):
                        parse_date(s)
            '''),
    },
    {"hidden_tests/check.sh": d('''
        set -e
        python -m pytest -q tests
        grep -q '"2023-02-29", "2024-02-30", "2024-04-31"' tests/test_dates.py
        python -c 'from dates import parse_date; assert parse_date("2000-02-29")==(2000,2,29)
        import sys
        try: parse_date("1900-02-29"); sys.exit(1)
        except ValueError: pass'
        ''')},
    "bash hidden_tests/check.sh",
)


from tasks_hard import register as _register_hard  # noqa: E402
_register_hard(task, d)
