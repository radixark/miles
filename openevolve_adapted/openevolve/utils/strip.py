#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
strip_comments.py
Remove Python '#' comments while preserving strings & shebang.
- Removes full-line and trailing '#' comments.
- Preserves shebang if it's the very first line.
- Does NOT remove docstrings or any string literals.
- Robust to syntax errors (no AST/CST parsing).
Usage:
  python strip_comments.py path/to/file.py > out.py
  cat in.py | python strip_comments.py - > out.py
  python strip_comments.py --test
"""

from __future__ import annotations
import sys
import argparse
import difflib
import textwrap


def strip_text(src: str) -> str:
    try:
        return _strip_text(src)
    except Exception as e:
        raise RuntimeError(f"strip_text failed: {e}\nSource:\n{src}") from e

def _strip_text(src: str) -> str:
    """
    Remove Python comments (lines/inline starting with '#') while preserving strings.
    - Works on text with possible syntax errors (pure text scan).
    - Keeps shebang ('#!') if it is the very first line.
    - Does NOT remove docstrings or any string literals.
    """
    out_chunks = []
    i, n = 0, len(src)

    # Preserve the first shebang line verbatim
    if src.startswith("#!"):
        j = src.find("\n")
        if j == -1:
            return src  # only shebang present
        out_chunks.append(src[:j + 1])
        i = j + 1

    in_single = False       # inside '...'
    in_double = False       # inside "..."
    in_triple: str | None = None  # "'''" or '"""'
    escape = False

    line_buf: list[str] = []       # current line content (without trailing comment)
    drop_line_entirely = False     # True if current line is pure comment

    def flush_line() -> bool:
        """Flush current line buffer. Return True if line is kept (emit newline), False if dropped."""
        nonlocal line_buf, drop_line_entirely
        if drop_line_entirely:
            line_buf = []
            drop_line_entirely = False
            return False
        s = "".join(line_buf).rstrip(" \t")
        line_buf = []
        if s == "":
            return True  # keep empty line (emit newline)
        out_chunks.append(s)
        return True

    while i < n:
        ch = src[i]

        # newline ends the current line
        if ch == "\n":
            kept = flush_line()
            if kept:
                out_chunks.append("\n")
            i += 1
            continue

        if in_triple:
            line_buf.append(ch)
            # close triple only with the same quote kind
            if ch == in_triple[0] and i + 2 < n and src[i:i + 3] == in_triple:
                line_buf.append(src[i + 1])
                line_buf.append(src[i + 2])
                i += 3
                in_triple = None
                continue
            i += 1
            continue

        if in_single:
            line_buf.append(ch)
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == "'":
                in_single = False
            i += 1
            continue

        if in_double:
            line_buf.append(ch)
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_double = False
            i += 1
            continue

        # not in any string
        if ch == "#":
            # if only whitespace so far, whole line is a comment line -> drop line entirely
            if "".join(line_buf).strip() == "":
                drop_line_entirely = True
                while i < n and src[i] != "\n":
                    i += 1
                continue
            # otherwise it's a trailing comment: skip until EOL
            while i < n and src[i] != "\n":
                i += 1
            continue

        # entering strings?
        if ch == "'":
            if i + 2 < n and src[i:i + 3] == "'''":
                in_triple = "'''"
                line_buf.extend(("'", "'", "'"))
                i += 3
                continue
            in_single = True
            line_buf.append(ch)
            i += 1
            continue

        if ch == '"':
            if i + 2 < n and src[i:i + 3] == '"""':
                in_triple = '"""'
                line_buf.extend(('"', '"', '"'))
                i += 3
                continue
            in_double = True
            line_buf.append(ch)
            i += 1
            continue

        # normal char
        line_buf.append(ch)
        i += 1

    # flush last line (if any content and not dropped), do not add extra newline
    if line_buf or drop_line_entirely:
        flush_line()

    return "".join(out_chunks)


# ---------------------- Built-in tests ---------------------- #
def _check(name: str, src: str, expected: str) -> bool:
    got = strip_text(src)
    ok = got == expected
    print(f"[{name}] {'OK' if ok else 'FAIL'}")
    if not ok:
        diff = "\n".join(
            difflib.unified_diff(
                expected.splitlines(), got.splitlines(),
                fromfile="expected", tofile="got", lineterm=""
            )
        )
        print(diff)
    return ok


def _run_tests() -> int:
    ok_all = True

    # 1) basic: remove comments, keep docstring
    src = textwrap.dedent("""\
        # module comment
        import os  # inline
        def f(x):  # trailing after header
            \"\"\"func doc\"\"\"
            return x + 1  # trailing
    """)
    exp = textwrap.dedent("""\
        import os
        def f(x):
            \"\"\"func doc\"\"\"
            return x + 1
    """)
    ok_all &= _check("basic", src, exp)

    # 2) hash inside strings preserved
    src = 's = "not a # comment"  # here\nprint(s)  # show\n'
    exp = 's = "not a # comment"\nprint(s)\n'
    ok_all &= _check("hash_in_string", src, exp)

    # 3) triple quotes as data (not docstring)
    src = textwrap.dedent("""\
        def g():
            x = \"\"\"data # not comment
            still data\"\"\"
            return x  # ok
    """)
    exp = textwrap.dedent("""\
        def g():
            x = \"\"\"data # not comment
            still data\"\"\"
            return x
    """)
    ok_all &= _check("triple_data", src, exp)

    # 4) shebang preserved
    src = '#!/usr/bin/env python3\n# cmt\nprint("hi") # tail\n'
    exp = '#!/usr/bin/env python3\nprint("hi")\n'
    ok_all &= _check("shebang", src, exp)

    # 5) syntax error tolerated (only strip comments)
    src = "def h(:\n    return  # trailing\n"
    exp = "def h(:\n    return\n"
    ok_all &= _check("syntax_error_ok", src, exp)

    print("\nALL PASSED" if ok_all else "\nSOME TESTS FAILED")
    return 0 if ok_all else 1


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description="Strip Python '#' comments while preserving strings and shebang.")
    ap.add_argument("path", nargs="?", default="-", help="Input file path or '-' for stdin (default: '-')")
    ap.add_argument("--test", action="store_true", help="Run built-in tests and exit")
    args = ap.parse_args(argv)

    if args.test:
        return _run_tests()

    if args.path == "-":
        src = sys.stdin.read()
    else:
        with open(args.path, "r", encoding="utf-8") as f:
            src = f.read()

    sys.stdout.write(strip_text(src))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
