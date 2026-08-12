import os
import shlex
import subprocess
import sys
from pathlib import Path

import pytest

import miles.utils.external_utils.model_args_utils as model_args_utils
from miles.utils.external_utils.model_args_utils import (
    load_model_args,
    load_sibling_model_args,
    moe_layer_freq,
    shell_safe_model_args,
)

_MODEL_ARGS_CLI = Path(model_args_utils.__file__).resolve()

_SCRIPT_BODY = """
import os

from model_args_utils import moe_layer_freq


def model_args(nlayers: int | None = None) -> str:
    nlayers = nlayers if nlayers is not None else int(os.environ.get("MODEL_ARGS_NUM_LAYERS") or 61)
    return (
        "--swiglu "
        f"--num-layers {nlayers} "
        f"--moe-layer-freq {moe_layer_freq(nlayers=nlayers, first_k_dense_replace=3)} "
    )
"""

_WRAPPER_BODY = """
from model_args_utils import load_sibling_model_args


def model_args() -> str:
    return load_sibling_model_args(__file__, "fake-model.4layer", nlayers=4)
"""


@pytest.fixture
def model_script(monkeypatch, tmp_path):
    path = tmp_path / "fake-model.4layer.py"
    path.write_text(_SCRIPT_BODY)
    (tmp_path / "fake-wrapper.py").write_text(_WRAPPER_BODY)
    monkeypatch.setattr("miles.utils.external_utils.model_args_utils._MODEL_SCRIPT_DIR", tmp_path)
    monkeypatch.delenv("MODEL_ARGS_NUM_LAYERS", raising=False)
    return path


class TestLoadModelArgsSplitting:
    def test_splits_each_line_the_way_read_ra_would(self, model_script):
        """One source line per flag must expand to the same argv the shell array held."""
        assert load_model_args("fake-model.4layer").split()[:3] == ["--swiglu", "--num-layers", "61"]

    def test_collapses_a_declaration_that_spans_several_lines(self, model_script):
        """A newline makes the launcher's `read -ra ... <<< "$(...)"` stop after the first line, silently."""
        model_script.write_text("def model_args() -> str:\n    return '--a 1\\n--b 2'\n")

        assert load_model_args("fake-model.4layer") == "--a 1 --b 2"

    def test_rejects_a_model_script_that_declares_nothing(self, model_script):
        """An all-whitespace declaration is a generator bug, not an empty argument."""
        model_script.write_text("def model_args() -> str:\n    return '   '\n")

        with pytest.raises(AssertionError):
            load_model_args("fake-model.4layer")

    def test_keeps_the_bracket_patterns_megatron_expects(self, model_script):
        """--moe-layer-freq values contain brackets and stars, which must survive as one token."""
        model_script.write_text("def model_args() -> str:\n    return '--moe-layer-freq [0]*3+[1]*75'\n")

        assert load_model_args("fake-model.4layer") == "--moe-layer-freq [0]*3+[1]*75"


class TestMoeLayerFreq:
    def test_renders_the_dense_prefix_then_moe_layers(self):
        """The mask must match `arr+=(0)` for the first K layers and `arr+=(1)` after."""
        assert moe_layer_freq(nlayers=5, first_k_dense_replace=2) == "[0,0,1,1,1]"

    def test_renders_only_dense_layers_when_the_model_is_shorter_than_the_dense_prefix(self):
        """The shell loop ran over the layer count, so a 2-layer deepseek-v3 got [0,0], not [0,0,0]."""
        assert moe_layer_freq(nlayers=2, first_k_dense_replace=3) == "[0,0]"

    def test_renders_an_all_moe_mask_when_no_dense_layers(self):
        """DeepSeek V4 has no dense prefix, so every entry is a MoE layer."""
        assert moe_layer_freq(nlayers=3, first_k_dense_replace=0) == "[1,1,1]"


class TestLoadModelArgs:
    def test_returns_the_declared_argv(self, model_script):
        """A python consumer gets argv tokens directly instead of sourcing a shell script."""
        assert load_model_args("fake-model.4layer").split() == [
            "--swiglu",
            "--num-layers",
            "61",
            "--moe-layer-freq",
            moe_layer_freq(nlayers=61, first_k_dense_replace=3),
        ]

    def test_forwards_keyword_overrides(self, model_script):
        """Layer-count variants are the same script called with a different argument."""
        assert load_model_args("fake-model.4layer", nlayers=4).split()[2] == "4"

    def test_rejects_an_unknown_model_type(self, model_script):
        """A typo must fail loudly rather than silently produce an argument-less run."""
        with pytest.raises(AssertionError):
            load_model_args("no-such-model")

    def test_a_dotted_filename_is_importable(self, model_script):
        """Model names like glm4.5-106B-A12B cannot be imported by module path."""
        assert "." in model_script.stem
        assert load_model_args(model_script.stem)

    def test_reads_the_model_scripts_of_the_requested_checkout(self, model_script, tmp_path_factory):
        """A launcher must get the model definition of its own checkout, not of the installed package."""
        other = tmp_path_factory.mktemp("other-checkout")
        (other / model_script.name).write_text(
            _SCRIPT_BODY.replace('"MODEL_ARGS_NUM_LAYERS") or 61', '"UNUSED") or 7')
        )

        assert load_model_args(model_script.stem, model_script_dir=other).split()[2] == "7"

    def test_a_wrapper_stays_inside_the_checkout_it_was_loaded_from(self, model_script, tmp_path_factory):
        """A variant script must reach the base script next to it, not the one of the installed package."""
        other = tmp_path_factory.mktemp("other-checkout")
        (other / "fake-wrapper.py").write_text(_WRAPPER_BODY)
        (other / model_script.name).write_text(
            _SCRIPT_BODY.replace('"MODEL_ARGS_NUM_LAYERS") or 61', '"UNUSED") or 7')
        )

        assert load_model_args("fake-wrapper", model_script_dir=other).split() == ["--swiglu", "--num-layers", "4"] + [
            "--moe-layer-freq",
            moe_layer_freq(nlayers=4, first_k_dense_replace=3),
        ]

    def test_still_honours_the_environment_override_the_shell_scripts_read(self, model_script, monkeypatch):
        """MODEL_ARGS_NUM_LAYERS used to reach the sourced .sh, so it must reach the .py too."""
        monkeypatch.setenv("MODEL_ARGS_NUM_LAYERS", "9")

        assert load_model_args("fake-model.4layer").split()[2] == "9"

    def test_ignores_an_empty_environment_override(self, model_script, monkeypatch):
        """`${VAR:-default}` falls back to the default when the variable is set but empty."""
        monkeypatch.setenv("MODEL_ARGS_NUM_LAYERS", "")

        assert load_model_args("fake-model.4layer").split()[2] == "61"

    def test_an_explicit_override_beats_the_environment(self, model_script, monkeypatch):
        """`MODEL_ARGS_NUM_LAYERS=5 source x.sh` let the caller's assignment win; keyword arguments must too."""
        monkeypatch.setenv("MODEL_ARGS_NUM_LAYERS", "9")

        assert load_model_args("fake-model.4layer", nlayers=4).split()[2] == "4"

    def test_honours_a_zero_override(self, model_script):
        """A dense-layer count of zero is a real value; `x or default` would silently restore the default."""
        model_script.write_text(_SCRIPT_BODY.replace("first_k_dense_replace=3", "first_k_dense_replace=nlayers"))

        assert load_model_args("fake-model.4layer", nlayers=0) == "--swiglu --num-layers 0 --moe-layer-freq []"

    def test_ignores_an_environment_override_the_model_does_not_declare(self, model_script, monkeypatch):
        """A model without a rotary base must not fail because some other model's variable is exported."""
        monkeypatch.setenv("MODEL_ARGS_ROTARY_BASE", "5000000")

        assert load_model_args("fake-model.4layer").split()[2] == "61"

    def test_rejects_an_override_the_model_does_not_declare(self, model_script):
        """The keyword reaches model_args() directly, so a misspelling is a TypeError rather than a silent no-op."""
        with pytest.raises(TypeError):
            load_model_args("fake-model.4layer", n_layers=4)


class TestLoadSiblingModelArgs:
    def test_resolves_the_base_next_to_the_variant_script(self, model_script, tmp_path_factory):
        """The variant knows where it lives; nothing else in the process does."""
        other = tmp_path_factory.mktemp("sibling-checkout")
        (other / model_script.name).write_text(
            _SCRIPT_BODY.replace('"MODEL_ARGS_NUM_LAYERS") or 61', '"UNUSED") or 7')
        )

        assert load_sibling_model_args(str(other / "anything.py"), model_script.stem).split()[2] == "7"


class TestModelArgsScript:
    def test_shell_consumers_recover_the_original_tokens(self):
        """Bracket patterns must survive read -ra without being glob-expanded."""
        script = (
            f'set -e; MODEL_ARGS_LINE="$({sys.executable} {_MODEL_ARGS_CLI} qwen3-4B)" || exit 1; '
            'read -ra MODEL_ARGS <<< "${MODEL_ARGS_LINE}"; printf "%s\\n" "${MODEL_ARGS[@]}"'
        )
        result = subprocess.run(["bash", "-c", script], capture_output=True, text=True, check=True)

        assert result.stdout.splitlines() == load_model_args("qwen3-4B").split()

    def test_an_unknown_model_type_stops_the_launcher(self):
        """A bare `read -ra ... <<< "$(...)"` swallows the failure and trains with no architecture flags."""
        script = (
            f'MODEL_ARGS_LINE="$({sys.executable} {_MODEL_ARGS_CLI} no-such-model 2>/dev/null)" || exit 1; '
            'echo "the launcher kept going"'
        )
        result = subprocess.run(["bash", "-c", script], capture_output=True, text=True)

        assert result.returncode == 1
        assert result.stdout == ""

    def test_a_here_string_read_stops_at_the_first_line(self):
        """Why load_model_args() collapses its result: read -ra drops the rest of a multi-line value silently."""
        script = 'read -ra MODEL_ARGS <<< "$1"; printf "%s\\n" "${MODEL_ARGS[@]}"'
        result = subprocess.run(
            ["bash", "-c", script, "_", "--a 1\n--b 2"], capture_output=True, text=True, check=True
        )

        assert result.stdout.split() == ["--a", "1"]

    def test_runs_from_a_checkout_whose_package_is_not_installed(self):
        """Executed by path with no site-packages, a model script must still reach the loader's helpers."""
        result = subprocess.run(
            [sys.executable, "-S", "-E", str(_MODEL_ARGS_CLI), "qwen3-30B-A3B"],
            capture_output=True,
            text=True,
            check=True,
            cwd="/",
        )

        assert "--num-layers" in result.stdout

    def test_forwards_the_rotary_base_override(self):
        """The geo3k launcher prefixes the command with MODEL_ARGS_ROTARY_BASE, as the shell scripts did."""
        result = subprocess.run(
            [sys.executable, str(_MODEL_ARGS_CLI), "qwen3-4B"],
            capture_output=True,
            text=True,
            check=True,
            env={**os.environ, "MODEL_ARGS_ROTARY_BASE": "5000000"},
        )

        assert "--rotary-base 5000000" in result.stdout


class TestShellSafeModelArgs:
    def test_quotes_the_tokens_a_shell_would_reinterpret(self):
        """--moe-layer-freq [0,0,0,1,1] is a glob; unquoted it expands against the launch directory."""
        assert "--moe-layer-freq '[0,0,0,1,1]'" in shell_safe_model_args("deepseek-v3-5layer")

    def test_leaves_ordinary_tokens_alone(self):
        """Quoting everything would churn every snapshot for no gain."""
        assert "--num-layers 36" in shell_safe_model_args("qwen3-4B")

    def test_survives_a_shell_round_trip_unchanged(self):
        """Escaping is only correct if the training process receives exactly the declared argv."""
        assert (
            shlex.split(shell_safe_model_args("deepseek-v3-5layer")) == load_model_args("deepseek-v3-5layer").split()
        )

    def test_is_empty_without_a_model_type(self):
        """FSDP launchers pass None and must contribute no argv at all."""
        assert shell_safe_model_args(None) == ""
