"""Pin the behavior of scripts/tools/sync_example_docs.py, the examples -> docs mirror.

The generator lives in scripts/tools because it writes files; these tests import it the
way tests/fast/examples/infra_features/p2p_weight_transfer/test_run.py imports run.py.
Each converter case here is a hazard the docs build either rejects (unescaped braces,
non-self-closed void tags) or silently drops (<details> content, corrupted inline math),
so the suite is what keeps README edits from breaking the published site.
"""

import importlib.util
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT = REPO_ROOT / "scripts" / "tools" / "sync_example_docs.py"


@pytest.fixture(scope="module")
def sync() -> ModuleType:
    spec = importlib.util.spec_from_file_location("sync_example_docs", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def convert(sync, readme, rel_dir="fully_async", mirrored=None, broken=None):
    return sync.convert(readme, rel_dir, mirrored or {}, [] if broken is None else broken)


class TestTitle:
    def test_fenced_comment_is_not_the_title(self, sync):
        readme = "```bash\n# Download the model\nhf download x\n```\n\n# Real Title\n\nBody.\n"
        title, body = convert(sync, readme)
        assert title == "Real Title"
        assert "# Download the model" in body  # and the fence keeps its line

    def test_missing_title_is_an_error(self, sync):
        with pytest.raises(sync.SyncError, match="no level-1 heading"):
            convert(sync, "just prose\n")


class TestMasking:
    def test_inline_math_braces_survive(self, sync):
        _, body = convert(sync, "# T\n\nRoles $\\pi_{\\text{old}}$ and $\\pi_{\\text{new}}$.\n")
        assert "$\\pi_{\\text{old}}$" in body
        assert "\\{" not in body

    def test_display_math_survives(self, sync):
        _, body = convert(sync, "# T\n\n$$\nL_{\\text{PPO}}(\\theta)\n$$\n")
        assert "L_{\\text{PPO}}" in body

    def test_prose_braces_are_escaped(self, sync):
        _, body = convert(sync, "# T\n\nParses <tool_call>{...}</tool_call> tokens.\n")
        assert "\\{" in body
        assert "&lt;tool_call>" in body

    def test_unterminated_math_span_is_linear(self, sync):
        # Disjoint alternatives in the inline-math pattern; a pathological line must
        # fail to match in linear time rather than backtrack exponentially (CodeQL 133).
        readme = "# T\n\nPrice $" + "\\a" * 5000 + " end.\n"
        _, body = convert(sync, readme)
        assert "Price $" in body

    def test_unpaired_backtick_stays_within_its_paragraph(self, sync):
        readme = "# T\n\nBroken `tick here.\n\nNext [run](./run.py) paragraph.\n"
        broken = []
        _, body = convert(sync, readme, broken=broken)
        # The link in the following paragraph is still seen by the rewriter.
        assert any("run.py" in b for b in broken)


class TestLinks:
    def test_mirrored_readme_resolves_to_site_page(self, sync):
        mirrored = {"examples/fully_async": "fully_async"}
        _, body = convert(sync, "# T\n\nSee [it](./README.md).\n", mirrored=mirrored)
        assert "](/examples/fully-async)" in body

    def test_badge_in_link_rewrites_both_hrefs(self, sync):
        mirrored = {"examples/fully_async": "fully_async"}
        broken = []
        _, body = convert(sync, "# T\n\n[![badge](./pic.png)](./README.md) end.\n", mirrored=mirrored, broken=broken)
        assert "](/examples/fully-async)" in body  # outer link
        assert any("pic.png" in b for b in broken)  # inner image checked too

    def test_repo_escaping_link_is_reported(self, sync):
        broken = []
        convert(sync, "# T\n\nSee [up](../../../outside.md).\n", broken=broken)
        assert any("escapes the repository" in b for b in broken)


class TestHtml:
    def test_common_tags_stay_markup(self, sync):
        _, body = convert(sync, '# T\n\n<h3 align="center">Hello</h3>\n')
        assert "<h3" in body and "&lt;h3" not in body

    def test_details_becomes_accordion_and_title_drops_braces(self, sync):
        readme = "# T\n\n<details>\n<summary>API {v2}</summary>\n\ncontent\n\n</details>\n"
        _, body = convert(sync, readme)
        assert '<Accordion title="API v2">' in body
        assert "content" in body

    def test_void_img_tag_becomes_markdown_image(self, sync):
        _, body = convert(
            sync, '# T\n\n<p align="center">\n<img src="https://x.test/a.png" alt="A" width="800">\n</p>\n'
        )
        assert "![A](https://x.test/a.png)" in body
        assert "<img" not in body


class TestIndex:
    def test_first_sentence_strips_markdown(self, sync):
        text = "# T\n\nThis **mirrors** [stuff](./run.py) nicely. Second sentence.\n"
        assert sync.first_sentence(text) == "This mirrors stuff nicely."

    def test_parse_index_reads_bullets_and_heading_links(self, sync, tmp_path):
        index = tmp_path / "README.md"
        index.write_text(
            "# Examples\n\n"
            "- **[fully_async](./fully_async)**: Async rollout.\n"
            "## [Infra Features](./infra_features)\n"
        )
        descriptions, registered = sync.parse_index(index)
        assert descriptions == {"fully_async": "Async rollout."}
        assert registered == {"fully_async", "infra_features"}

    def test_navigation_follows_bullet_order(self, sync):
        pages = {"": None, "zeta": None, "alpha": None, "unlisted_b": None, "unlisted_a": None}
        entries = sync.build_navigation(pages, ["zeta", "alpha"])
        assert entries == [
            "examples/index",
            {
                "group": "Recipes",
                "pages": [
                    "examples/zeta",
                    "examples/alpha",
                    "examples/unlisted-a",
                    "examples/unlisted-b",
                ],
            },
        ]

    def test_section_directory_becomes_rootless_group_with_landing_first(self, sync):
        pages = {"": None, "alpha": None, "infra_features": None, "infra_features/beta": None}
        entries = sync.build_navigation(pages, ["alpha"])
        assert entries[-1] == {
            "group": "Infra Features",
            "pages": ["examples/infra-features", "examples/infra-features/beta"],
        }


class TestBuildPages:
    def _mini_tree(self, tmp_path, monkeypatch, sync, index_text):
        examples = tmp_path / "examples"
        (examples / "demo").mkdir(parents=True)
        (examples / "README.md").write_text(index_text)
        (examples / "demo" / "README.md").write_text("# Demo\n\nA demo example. More prose.\n")
        monkeypatch.setattr(sync, "EXAMPLES", examples)

    def test_unregistered_directory_is_an_error(self, sync, tmp_path, monkeypatch):
        self._mini_tree(tmp_path, monkeypatch, sync, "# Examples\n\nNo bullets here.\n")
        with pytest.raises(sync.SyncError, match="not listed in examples/README.md"):
            sync.build_pages()

    def test_registered_directory_renders_with_its_bullet_description(self, sync, tmp_path, monkeypatch):
        self._mini_tree(tmp_path, monkeypatch, sync, "# Examples\n\n- **[demo](./demo)**: One-liner.\n")
        _, rendered, order = sync.build_pages()
        page = next(text for path, text in rendered.items() if path.name == "demo.md")
        assert 'description: "One-liner."' in page
        assert order == ["demo"]


def test_docs_examples_matches_the_readmes():
    """The committed mirror must be exactly what the generator produces (drift gate)."""
    result = subprocess.run([sys.executable, str(SCRIPT), "--check"], capture_output=True, text=True, cwd=REPO_ROOT)
    assert result.returncode == 0, f"docs/examples is out of sync with examples/:\n{result.stderr}"
