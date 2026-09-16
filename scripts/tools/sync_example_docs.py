#!/usr/bin/env python3
"""Mirror the README files under examples/ into the Examples tab of the docs site.

examples/ is the single source of truth. Every README.md outside examples/experimental/
becomes one page under docs/examples/, and the Examples tab in docs/docs.json is
regenerated from the same tree, so the site cannot drift from the repository.

Usage:
    python scripts/tools/sync_example_docs.py            # regenerate
    python scripts/tools/sync_example_docs.py --check    # fail if anything is stale

The generated pages are Mintlify MDX, which is stricter than GitHub-flavored Markdown.
Two constructs in a README break the build and are rewritten here: unescaped braces
(parsed as JSX expressions) and non-self-closing void tags such as <img> and <br>.
Relative links and images are rewritten to the site page when the target is mirrored,
and to GitHub otherwise.

docs/docs.json is round-tripped through json.dumps on every run, so this script owns
that file's formatting (indent=1); hand-edits to other tabs keep their content but are
renormalized to that style.
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
EXAMPLES = REPO / "examples"
DOCS = REPO / "docs"
DOCS_JSON = DOCS / "docs.json"
OUT_DIR = DOCS / "examples"

BRANCH = "main"
GITHUB_TREE = f"https://github.com/radixark/miles/tree/{BRANCH}"
GITHUB_BLOB = f"https://github.com/radixark/miles/blob/{BRANCH}"
GITHUB_RAW = f"https://raw.githubusercontent.com/radixark/miles/{BRANCH}"

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".gif", ".svg", ".webp"}
MAX_DESCRIPTION = 160

# Tags Mintlify renders as JSX. Anything else in prose (<tool_call>, <search>, ...) is a
# model token, not markup, and is escaped so acorn never sees it.
HTML_TAGS = {
    "a",
    "b",
    "blockquote",
    "br",
    "code",
    "details",
    "div",
    "em",
    "hr",
    "i",
    "img",
    "kbd",
    "li",
    "ol",
    "p",
    "pre",
    "span",
    "strong",
    "sub",
    "summary",
    "sup",
    "table",
    "tbody",
    "td",
    "th",
    "thead",
    "tr",
    "ul",
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "h6",
    "caption",
    "center",
    "dd",
    "del",
    "dl",
    "dt",
    "figcaption",
    "figure",
    "ins",
    "mark",
    "picture",
    "s",
    "small",
    "source",
    "u",
    "video",
    # Mintlify's own component, which <details> blocks are rewritten into.
    "accordion",
}
VOID_TAGS = {"br", "hr", "img", "input"}

# - **[fully_async](./fully_async)**: Demonstrates fully asynchronous rollout generation.
INDEX_BULLET = re.compile(r"^\s*[-*]\s+\*\*\[([^\]]+)\]\(([^)]+)\)\*\*:\s*(.+?)\s*$")
# ## [Infra Features](./infra_features) — a section directory registered as a heading link.
INDEX_HEADING = re.compile(r"^#{2,}\s+\[([^\]]+)\]\(([^)]+)\)\s*$")

# Directories rendered as label-only sidebar sections. Their own README becomes the
# section's first row, labeled "Overview" — never a group `root`, which would make the
# header itself navigate (banned by pre-commit; see docs/README.md).
SECTION_DIRS = ("infra_features",)
MD_LINK = re.compile(r"(!?)\[([^\]]*)\]\(\s*([^)\s]+)(\s+\"[^\"]*\")?\s*\)")
IMG_LINK = re.compile(r"!\[([^\]]*)\]\(\s*([^)\s]+)(\s+\"[^\"]*\")?\s*\)")
# Link text may carry one nested image ([![badge](img)](target)), which MD_LINK would
# mis-parse as text "![badge" with href "img".
OUTER_LINK = re.compile(r"(?<!!)\[((?:[^\[\]\n]|!\[[^\]]*\]\([^)]*\))*)\]\(\s*([^)\s]+)(\s+\"[^\"]*\")?\s*\)")
HTML_TAG = re.compile(r"<(/?)([A-Za-z][A-Za-z0-9:-]*)((?:\"[^\"]*\"|'[^']*'|[^>\"'])*?)(/?)>")
IMG_TAG = re.compile(r"<img\b((?:\"[^\"]*\"|'[^']*'|[^>\"'])*)/?>", re.IGNORECASE)
ATTR = re.compile(r"([A-Za-z-]+)\s*=\s*\"([^\"]*)\"|([A-Za-z-]+)\s*=\s*'([^']*)'")
DETAILS = re.compile(r"<details[^>]*>\s*(?:<summary[^>]*>(.*?)</summary>)?(.*?)</details>", re.DOTALL | re.IGNORECASE)
# Content between these markers stays on GitHub but is left out of the docs site.
EXCLUDE_BLOCK = re.compile(r"<!--\s*docs:exclude:start\s*-->.*?<!--\s*docs:exclude:end\s*-->", re.DOTALL)
HTML_COMMENT = re.compile(r"<!--.*?-->", re.DOTALL)
FENCE = re.compile(r"^\s*(```+|~~~+)")


class SyncError(Exception):
    pass


def discover_pages():
    """Map each mirrored directory (relative to examples/, "" for the root) to its README."""
    pages = {}
    for readme in sorted(EXAMPLES.rglob("README.md")):
        rel = readme.parent.relative_to(EXAMPLES)
        parts = [] if rel == Path(".") else list(rel.parts)
        if "experimental" in parts or "__pycache__" in parts:
            continue
        pages["/".join(parts)] = readme
    return pages


def slug_for(rel_dir):
    if not rel_dir:
        return "index"
    return "/".join(part.replace("_", "-").lower() for part in rel_dir.split("/"))


def site_url(rel_dir):
    return "/examples" if not rel_dir else f"/examples/{slug_for(rel_dir)}"


def repo_dir_of(rel_dir):
    return "examples" if not rel_dir else f"examples/{rel_dir}"


def parse_index(index_readme):
    """Read examples/README.md, the registry for every mirrored page.

    Returns (descriptions, registered): one-line descriptions keyed by directory in
    bullet order, and the set of every directory the index mentions — as a bullet or,
    for a section directory like infra_features, as a section-heading link.
    """
    descriptions, registered = {}, set()
    for line in index_readme.read_text().splitlines():
        bullet = INDEX_BULLET.match(line)
        heading = None if bullet else INDEX_HEADING.match(line)
        m = bullet or heading
        if not m:
            continue
        target = m.group(2).split("#")[0].strip().rstrip("/")
        rel = os.path.normpath(os.path.join("examples", target))
        if not rel.startswith("examples/"):
            continue
        rel_dir = rel[len("examples/") :]
        registered.add(rel_dir)
        if bullet:
            descriptions[rel_dir] = bullet.group(3)
    return descriptions, registered


def first_sentence(text):
    """First prose sentence of a README, used when the index has no bullet for a page."""
    body = []
    in_fence = False
    for line in text.splitlines():
        if FENCE.match(line):
            in_fence = not in_fence
            continue
        if in_fence or line.startswith("#"):
            continue
        if line.strip():
            body.append(line.strip())
        elif body:
            break
    para = " ".join(body)
    para = MD_LINK.sub(lambda m: m.group(2) or m.group(3), para)
    para = re.sub(r"[*`_]", "", para)
    match = re.search(r"^(.+?[.!?])(\s|$)", para)
    return (match.group(1) if match else para).strip()


def mask_code(text):
    """Replace fenced blocks, inline code and display math with placeholders."""
    stash = []

    def keep(chunk):
        stash.append(chunk)
        return f"\x00{len(stash) - 1}\x00"

    out_lines = []
    fence = None
    buffer = []
    for line in text.split("\n"):
        m = FENCE.match(line)
        if fence is None and m:
            fence = m.group(1)
            buffer = [line]
        elif fence is not None:
            buffer.append(line)
            if m and line.strip().startswith(fence):
                out_lines.append(keep("\n".join(buffer)))
                fence = None
        else:
            out_lines.append(line)
    if fence is not None:
        out_lines.append(keep("\n".join(buffer)))
    text = "\n".join(out_lines)

    text = re.sub(r"\$\$.*?\$\$", lambda m: keep(m.group(0)), text, flags=re.DOTALL)
    # A code span may wrap a line but not a paragraph, so an unpaired backtick cannot
    # swallow the rest of the document into the stash.
    text = re.sub(r"(`+)((?:(?!\n\n)[^`])+?)\1", lambda m: keep(m.group(0)), text)
    # Inline math, bounded to one line so a stray "$5" in prose stays inert. The two
    # alternatives are disjoint (an escape, or anything but a backslash), so an
    # unterminated span cannot trigger exponential backtracking.
    text = re.sub(r"(?<!\$)\$(?!\$)(?:\\[^\n]|[^\\$\n])+\$(?!\$)", lambda m: keep(m.group(0)), text)
    return text, stash


def unmask_code(text, stash):
    # A stashed chunk can itself contain a placeholder (inline math around a code
    # span, say), so expand until none remain.
    while "\x00" in text:
        text = re.sub(r"\x00(\d+)\x00", lambda m: stash[int(m.group(1))], text)
    return text


def resolve_link(href, cur_dir, mirrored, broken):
    """Rewrite a repo-relative href to a site page or a GitHub URL. None keeps it as is."""
    if not href or href.startswith(("#", "/")) or re.match(r"^[A-Za-z][A-Za-z0-9+.-]*:", href):
        return None
    path, sep, fragment = href.partition("#")
    if not path:
        return None
    rel = os.path.normpath(os.path.join(repo_dir_of(cur_dir), path))
    if rel.startswith(".."):
        broken.append(f"{repo_dir_of(cur_dir)}/README.md -> {href} (escapes the repository)")
        return None
    suffix = "#" + fragment if sep else ""

    target = rel[: -len("/README.md")] if rel.endswith("/README.md") else rel
    if target in mirrored:
        return site_url(mirrored[target]) + suffix

    absolute = REPO / rel
    # A README pointing at a docs source file should point at the published page instead.
    if absolute.is_file() and rel.startswith("docs/") and absolute.suffix in {".md", ".mdx"}:
        page = rel[len("docs/") : -len(absolute.suffix)]
        return f"/{page[: -len('/index')] if page.endswith('/index') else page}{suffix}"
    if absolute.is_dir():
        return f"{GITHUB_TREE}/{rel}{suffix}"
    if absolute.is_file():
        if absolute.suffix.lower() in IMAGE_SUFFIXES:
            return f"{GITHUB_RAW}/{rel}"
        return f"{GITHUB_BLOB}/{rel}{suffix}"
    broken.append(f"{repo_dir_of(cur_dir)}/README.md -> {href}")
    return None


def convert_img_tags(text, cur_dir, mirrored, broken):
    """<img src=... alt=...> is not self-closing in most READMEs; markdown carries it fine."""

    def replace(match):
        attrs = dict()
        for m in ATTR.finditer(match.group(1)):
            key = (m.group(1) or m.group(3)).lower()
            attrs[key] = m.group(2) if m.group(2) is not None else m.group(4)
        src = attrs.get("src", "")
        resolved = resolve_link(src, cur_dir, mirrored, broken)
        # A row of side-by-side images relies on the width attributes we just dropped, so
        # give each one its own paragraph rather than letting them run into each other.
        return f"\n\n![{attrs.get('alt', '')}]({resolved or src})\n\n"

    text = IMG_TAG.sub(replace, text)
    # The wrapper <p align="center"> around those images carries no meaning in markdown.
    text = re.sub(r"</?p\b[^>]*>", "\n", text)
    text = re.sub(r"^[ \t]+$", "", text, flags=re.MULTILINE)
    return re.sub(r"\n{3,}", "\n\n", text)


def convert_details_blocks(text):
    """A <details> block renders empty on the docs site; Mintlify's Accordion keeps it."""

    def replace(match):
        summary, inner = match.group(1) or "Details", match.group(2)
        # Braces would be escaped to \{ later and render literally inside the attribute.
        title = re.sub(r"[*`_{}]|<[^>]+>", "", summary).strip().replace('"', "'")
        return f'\n\n<Accordion title="{title}">\n\n{inner.strip()}\n\n</Accordion>\n\n'

    return DETAILS.sub(replace, text)


def escape_html(text):
    """Self-close void tags; escape anything that is a model token rather than markup."""

    def replace(match):
        closing, name, attrs, self_closed = match.groups()
        if name.lower() not in HTML_TAGS:
            return "&lt;" + match.group(0)[1:]
        if name.lower() in VOID_TAGS and not self_closed and not closing:
            return f"<{name}{attrs.rstrip()} />"
        return match.group(0)

    return HTML_TAG.sub(replace, text)


def convert(readme_text, rel_dir, mirrored, broken):
    readme_text = EXCLUDE_BLOCK.sub("", readme_text)
    # Mask before anything else, including the title scan: a "# comment" inside a
    # fenced block must not be mistaken for the page's level-1 heading.
    masked, stash = mask_code(readme_text)
    lines = masked.split("\n")
    title = None
    for i, line in enumerate(lines):
        if line.startswith("# "):
            title = unmask_code(line[2:].strip(), stash)
            del lines[i]
            break
    if title is None:
        raise SyncError(f"{repo_dir_of(rel_dir)}/README.md has no level-1 heading to use as the page title")
    body = "\n".join(lines)

    # GitHub-only annotations; also raw comments are not valid MDX.
    body = HTML_COMMENT.sub("", body)
    body = convert_img_tags(body, rel_dir, mirrored, broken)

    def rewrite(bang):
        def replace(match):
            text, href, hint = match.groups()
            resolved = resolve_link(href, rel_dir, mirrored, broken)
            return f"{bang}[{text}]({resolved or href}{hint or ''})"

        return replace

    body = IMG_LINK.sub(rewrite("!"), body)
    body = OUTER_LINK.sub(rewrite(""), body)
    body = convert_details_blocks(body)
    body = escape_html(body)
    body = body.replace("{", "\\{").replace("}", "\\}")
    body = re.sub(r"\n{3,}", "\n\n", body)
    body = unmask_code(body, stash)
    return title, body.strip("\n")


def render_page(title, description, rel_dir, body):
    source = f"{repo_dir_of(rel_dir)}/README.md"
    # Tab and section landing pages keep their full title (h1 / SEO) but show a short,
    # uniform "Overview" label in the sidebar (see the convention in docs/README.md).
    landing = rel_dir == "" or rel_dir in SECTION_DIRS
    sidebar = 'sidebarTitle: "Overview"\n' if landing else ""
    return (
        "---\n"
        f"title: {json.dumps(title, ensure_ascii=False)}\n"
        f"{sidebar}"
        f"description: {json.dumps(description, ensure_ascii=False)}\n"
        f"# Generated from {source} by scripts/tools/sync_example_docs.py. Edit that README, not this file.\n"
        "---\n"
        f"{body}\n"
    )


def build_pages():
    pages = discover_pages()
    if "" not in pages:
        raise SyncError("examples/README.md is missing; it is the source of the Examples index page")
    mirrored = {repo_dir_of(rel): rel for rel in pages}
    descriptions, registered = parse_index(pages[""])

    unregistered = sorted(rel_dir for rel_dir in pages if rel_dir and rel_dir not in registered)
    if unregistered:
        raise SyncError(
            "mirrored but not listed in examples/README.md — add a bullet for:\n  "
            + "\n  ".join(repo_dir_of(d) for d in unregistered)
        )

    broken, rendered, slug_owner = [], {}, {}
    for rel_dir, readme in sorted(pages.items()):
        text = readme.read_text()
        title, body = convert(text, rel_dir, mirrored, broken)
        description = descriptions.get(rel_dir)
        if description is None:
            # Derived from the README's own first sentence, which Mintlify already renders
            # under the title as the description — drop the duplicate from the body. The
            # comparison strips markdown the same way first_sentence does, so a link or
            # emphasis in the opening sentence does not defeat the dedup.
            description = first_sentence(text)
            lead = re.match(r"\s*(.+?[.!?])(\s|$)", body, re.DOTALL)
            if lead:
                normalized = MD_LINK.sub(lambda m: m.group(2) or m.group(3), lead.group(1))
                if re.sub(r"[*`_]", "", normalized).strip() == description:
                    body = body[lead.end(1) :].lstrip()
        if not description:
            raise SyncError(f"{repo_dir_of(rel_dir)}/README.md has no description; add a bullet in examples/README.md")
        if len(description) > MAX_DESCRIPTION:
            raise SyncError(
                f"description for {repo_dir_of(rel_dir)} is {len(description)} characters, "
                f"over the {MAX_DESCRIPTION} the docs site allows; shorten it at the source"
            )
        out_path = OUT_DIR / f"{slug_for(rel_dir)}.md"
        if out_path in rendered:
            raise SyncError(
                f"slug collision: {repo_dir_of(slug_owner[out_path])} and {repo_dir_of(rel_dir)} "
                f"both map to {out_path.relative_to(REPO)}"
            )
        slug_owner[out_path] = rel_dir
        rendered[out_path] = render_page(title, description, rel_dir, body)

    if broken:
        raise SyncError("READMEs link to paths that do not exist:\n  " + "\n  ".join(sorted(set(broken))))
    # dicts preserve insertion order, so this is the bullet order of examples/README.md.
    return pages, rendered, list(descriptions)


def build_navigation(pages, bullet_order):
    """Examples tab, mirroring the directory layout: top-level recipes, then infra_features.

    Sidebar order follows the bullet order in examples/README.md — the index README owns
    ordering along with titles and descriptions. Directories without a bullet sort last,
    alphabetically.

    Emits the tab's `pages` list directly: the index page first, then label-only
    sections whose landing page is their first row — no group `root` (see docs/README.md).
    """
    rank = {rel_dir: i for i, rel_dir in enumerate(bullet_order)}
    recipes, infra = [], []
    for rel_dir in sorted(pages, key=lambda d: (rank.get(d, len(rank)), d)):
        if not rel_dir:
            continue
        page = f"examples/{slug_for(rel_dir)}"
        (infra if rel_dir.startswith("infra_features") else recipes).append(page)
    entries = ["examples/index"]
    entries.append({"group": "Recipes", "pages": recipes})
    if infra:
        infra_root = "examples/infra-features"
        children = [p for p in infra if p != infra_root]
        entries.append({"group": "Infra Features", "pages": [infra_root] + children})
    return entries


def examples_tab(config):
    for tab in config["navigation"]["tabs"]:
        if tab.get("tab") == "Examples":
            return tab
    raise SyncError('docs.json has no "Examples" tab')


def render_docs_json(config):
    return json.dumps(config, indent=1, ensure_ascii=False) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", action="store_true", help="fail instead of writing when something is stale")
    args = parser.parse_args()

    try:
        pages, rendered, bullet_order = build_pages()
    except SyncError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    config = json.loads(DOCS_JSON.read_text())
    try:
        tab = examples_tab(config)
    except SyncError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    tab.pop("groups", None)
    tab["pages"] = build_navigation(pages, bullet_order)
    docs_json_text = render_docs_json(config)

    existing = {p for p in OUT_DIR.rglob("*.md")} if OUT_DIR.exists() else set()
    stale = sorted(existing - set(rendered))
    changed = sorted(p for p, text in rendered.items() if not p.exists() or p.read_text() != text)
    json_changed = DOCS_JSON.read_text() != docs_json_text

    if args.check:
        problems = [f"stale: {p.relative_to(REPO)}" for p in stale]
        problems += [f"out of date: {p.relative_to(REPO)}" for p in changed]
        if json_changed:
            problems.append("out of date: docs/docs.json")
        if problems:
            print("error: docs/examples is out of sync with examples/:", file=sys.stderr)
            for problem in problems:
                print(f"  {problem}", file=sys.stderr)
            print("run: python scripts/tools/sync_example_docs.py", file=sys.stderr)
            return 1
        return 0

    for path in stale:
        path.unlink()
    for path, text in rendered.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    if json_changed:
        DOCS_JSON.write_text(docs_json_text)

    for path in stale:
        print(f"removed {path.relative_to(REPO)}")
    for path in changed:
        print(f"wrote {path.relative_to(REPO)}")
    if json_changed:
        print("wrote docs/docs.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
