"""Execute the Python examples of the manual and the tutorials, in order, one namespace per file
(like notebook cells), inside a temporary directory. Blocks preceded by a line starting with
``<!-- not run`` are skipped. Exit code 1 if any block fails.

    python tools/manual/check_manual.py                         # docs/manual.md
    python tools/manual/check_manual.py docs/tutorials/*.md     # the tutorials
    python tools/manual/check_manual.py --all                   # both
    python tools/manual/check_manual.py FILE --from N           # start at block N (of one file)
    python tools/manual/check_manual.py FILE --update-output    # also write each block's printed
                                                                # output below it (tutorials)

With ``--update-output`` the text printed by a block is written into the ```text block that
follows the line ``Output (from the test run):`` after it (inserted if missing), so the
tutorials show real, reproducible results.
"""
import contextlib
import glob
import io
import os
import sys
import tempfile
import time
import traceback

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MANUAL = os.path.join(ROOT, "docs", "manual.md")
TUTORIALS = os.path.join(ROOT, "docs", "tutorials")
MARK = "Output (from the test run):"
MAX_LINES = 40


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
                            line=i + 1, end=j))
            i = j
        i += 1
    return out


class _Tee(io.StringIO):
    def __init__(self, echo):
        super().__init__()
        self.echo = echo

    def write(self, s):
        self.echo.write(s)
        return super().write(s)


def _clean(text):
    lines = [ln.rstrip() for ln in text.strip("\n").splitlines()]
    if len(lines) > MAX_LINES:
        lines = lines[:MAX_LINES - 1] + [f"... ({len(lines) - MAX_LINES + 1} more lines)"]
    return lines


def write_outputs(path, outputs):
    """Insert or replace the output block after each executed code block (outputs: {end line: text})."""
    lines = open(path).read().splitlines()
    for end in sorted(outputs, reverse=True):              # bottom up: line numbers stay valid
        body = _clean(outputs[end])
        k = end + 1
        while k < len(lines) and not lines[k].strip():
            k += 1
        if k < len(lines) and lines[k].strip() == MARK:
            s = k + 1
            while s < len(lines) and lines[s].strip() != "```text":
                s += 1
            e = s + 1
            while e < len(lines) and lines[e].strip() != "```":
                e += 1
            if not body:
                del lines[end + 1:e + 1]
            else:
                lines[s + 1:e] = body
        elif body:
            lines[end + 1:end + 1] = ["", MARK, "", "```text"] + body + ["```"]
    open(path, "w").write("\n".join(lines) + "\n")


def run_file(path, start=0, update=False):
    bl = blocks(open(path).read())
    ns = {"__name__": "__manual__"}
    failed, outputs = [], {}
    cwd = os.getcwd()
    name = os.path.relpath(path, ROOT)
    with tempfile.TemporaryDirectory() as tmp:
        os.chdir(tmp)
        try:
            for k, b in enumerate(bl):
                tag = f"{name} [{k:2d}] line {b['line']:4d} {b['heading'][:40]}"
                if b["skip"] or k < start:
                    print(f"{tag}: skipped")
                    continue
                t0 = time.perf_counter()
                buf = _Tee(sys.stdout)
                try:
                    with contextlib.redirect_stdout(buf):
                        exec(compile(b["code"], f"{name} block {k} (line {b['line']})", "exec"), ns)
                    outputs[b["end"]] = buf.getvalue()
                    print(f"{tag}: ok ({time.perf_counter() - t0:.1f} s)", flush=True)
                except Exception:
                    failed.append(k)
                    print(f"{tag}: FAILED\n{traceback.format_exc()}", flush=True)
        finally:
            os.chdir(cwd)
    if update and not failed:
        write_outputs(path, outputs)
    print(f"{name}: {len(bl)} blocks, {sum(b['skip'] for b in bl)} skipped, {len(failed)} failed {failed}")
    return failed


def main():
    args = sys.argv[1:]
    start = 0
    if "--from" in args:
        i = args.index("--from")
        start = int(args[i + 1])
        del args[i:i + 2]
    update = "--update-output" in args
    files = [a for a in args if not a.startswith("--")]
    if "--all" in args:
        files = [MANUAL] + sorted(glob.glob(os.path.join(TUTORIALS, "[0-9]*.md")))
    files = files or [MANUAL]
    bad = {}
    for f in files:
        fl = run_file(os.path.abspath(f), start, update)
        if fl:
            bad[f] = fl
    if len(files) > 1:
        print(f"{len(files)} files, failures: {bad if bad else 'none'}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
