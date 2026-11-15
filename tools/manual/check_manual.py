"""Execute the Python examples of docs/manual.md in order, in one namespace (like notebook
cells), inside a temporary directory. Blocks preceded by a line starting with
``<!-- not run`` are skipped. Exit code 1 if any block fails.

    python tools/manual/check_manual.py [--from N]
"""
import os
import re
import sys
import tempfile
import time
import traceback

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MANUAL = os.path.join(ROOT, "docs", "manual.md")


def blocks(text):
    lines = text.splitlines()
    out, heading, i = [], "", 0
    while i < len(lines):
        ln = lines[i]
        if ln.startswith("## "):
            heading = ln[3:].strip()
        if ln.strip() == "```python":
            prev = next((lines[k] for k in range(i - 1, -1, -1) if lines[k].strip()), "")
            j = i + 1
            while lines[j].strip() != "```":
                j += 1
            out.append(dict(heading=heading, code="\n".join(lines[i + 1:j]), skip=prev.startswith("<!-- not run"),
                            line=i + 1))
            i = j
        i += 1
    return out


def main():
    start = int(sys.argv[sys.argv.index("--from") + 1]) if "--from" in sys.argv else 0
    bl = blocks(open(MANUAL).read())
    ns = {"__name__": "__manual__"}
    failed = []
    cwd = os.getcwd()
    with tempfile.TemporaryDirectory() as tmp:
        os.chdir(tmp)
        try:
            for k, b in enumerate(bl):
                tag = f"[{k:2d}] line {b['line']:4d} {b['heading'][:50]}"
                if b["skip"] or k < start:
                    print(f"{tag}: skipped")
                    continue
                t0 = time.perf_counter()
                try:
                    exec(compile(b["code"], f"manual block {k} (line {b['line']})", "exec"), ns)
                    print(f"{tag}: ok ({time.perf_counter() - t0:.1f} s)", flush=True)
                except Exception:
                    failed.append(k)
                    print(f"{tag}: FAILED\n{traceback.format_exc()}", flush=True)
        finally:
            os.chdir(cwd)
    print(f"{len(bl)} blocks, {sum(b['skip'] for b in bl)} skipped, {len(failed)} failed {failed}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
