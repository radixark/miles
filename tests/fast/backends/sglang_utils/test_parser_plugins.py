"""Installed parsers are available to CLI rendering and fresh rollout processes."""

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=120, suite="stage-b-cpu", labels=["sglang"])

_PLUGIN = """
from sglang.srt.function_call.function_call_parser import FunctionCallParser
from sglang.srt.function_call.parser_names import TOOL_CALL_PARSER_NAMES

def register():
    FunctionCallParser.ToolCallParserEnum["miles_test_parser"] = FunctionCallParser.ToolCallParserEnum["qwen25"]
    TOOL_CALL_PARSER_NAMES.append("miles_test_parser")
"""


@pytest.mark.parametrize("seams", [("cli", "engine"), ("worker",)], ids=["cli", "worker"])
def test_installed_parser_plugin(tmp_path, seams):
    # Real distribution metadata exercises entry-point discovery without installing a package.
    (tmp_path / "miles_test_parser.py").write_text(textwrap.dedent(_PLUGIN))
    distribution = tmp_path / "miles_test_parser-0.0.0.dist-info"
    distribution.mkdir()
    (distribution / "METADATA").write_text("Metadata-Version: 2.1\nName: miles-test-parser\nVersion: 0.0.0\n")
    (distribution / "entry_points.txt").write_text(
        "[sglang.srt.plugins]\nmiles_test_parser = miles_test_parser:register\n"
    )
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(tmp_path), str(Path(__file__).resolve().parents[4]), env.get("PYTHONPATH", "")]
    )
    env["SGLANG_PLUGINS"] = "miles_test_parser"
    for seam in seams:
        result = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), seam],
            env=env,
            text=True,
            capture_output=True,
            timeout=120,
        )
        assert result.returncode == 0, result.stdout + result.stderr


def _exercise(seam):
    if seam == "cli":
        import argparse

        from miles.backends.sglang_utils.arguments import add_sglang_arguments

        parser = add_sglang_arguments(argparse.ArgumentParser())
        for prefix in ("sglang", "eval-sglang"):
            args = parser.parse_args([f"--{prefix}-tool-call-parser", "miles_test_parser"])
            assert getattr(args, prefix.replace("-", "_") + "_tool_call_parser") == "miles_test_parser"
    elif seam == "engine":
        from miles.backends.sglang_utils.server_args_utils import parse_server_args_argv, server_args_to_argv

        argv = server_args_to_argv({"model_path": "dummy", "tool_call_parser": "miles_test_parser", "device": "cpu"})
        assert parse_server_args_argv(argv).tool_call_parser == "miles_test_parser"
    else:
        from miles.rollout.generate_utils.tool_call_utils import create_tool_call_parser

        tools = [{"type": "function", "function": {"name": "lookup", "parameters": {"type": "object"}}}]
        parser = create_tool_call_parser(tools, "miles_test_parser")
        _, calls = parser.parse_non_stream('<tool_call>\n{"name":"lookup","arguments":{}}\n</tool_call>')
        assert len(calls) == 1 and calls[0].name == "lookup"


if __name__ == "__main__":
    _exercise(sys.argv[1])
