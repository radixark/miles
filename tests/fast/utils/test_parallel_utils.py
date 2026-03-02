import dataclasses
import importlib
import sys
import types

import pytest

# parallel_utils lives under cli/ but doesn't depend on the cli package.
# Importing via the normal path triggers cli/__init__.py which eagerly loads
# the full command tree (and currently fails on Python 3.12).  We pre-populate
# sys.modules with a placeholder so the parent package __init__ is never
# executed, then import the module directly.
_CLI_PKG = "miles.utils.debug_utils.run_megatron.cli"
if _CLI_PKG not in sys.modules:
    sys.modules[_CLI_PKG] = types.ModuleType(_CLI_PKG)

from miles.utils.debug_utils.run_megatron.cli.parallel_utils import (  # noqa: E402
    ParallelConfig,
    parse_parallel_args,
)


# ---------------------------------------------------------------------------
# Construction & defaults
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_defaults(self) -> None:
        config = ParallelConfig()
        assert config.tp == 1
        assert config.pp == 1
        assert config.cp == 1
        assert config.ep is None
        assert config.etp == 1

    def test_explicit_values(self) -> None:
        config = ParallelConfig(tp=2, pp=4, cp=2, ep=8, etp=2)
        assert config.tp == 2
        assert config.pp == 4
        assert config.cp == 2
        assert config.ep == 8
        assert config.etp == 2

    def test_frozen(self) -> None:
        config = ParallelConfig()
        with pytest.raises(dataclasses.FrozenInstanceError):
            config.tp = 2  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Properties
# ---------------------------------------------------------------------------


class TestProperties:
    def test_nproc(self) -> None:
        assert ParallelConfig(tp=2, pp=4, cp=2).nproc == 16

    def test_nproc_defaults(self) -> None:
        assert ParallelConfig().nproc == 1

    def test_effective_ep_defaults_to_tp(self) -> None:
        config = ParallelConfig(tp=4)
        assert config.effective_ep == 4

    def test_effective_ep_explicit(self) -> None:
        config = ParallelConfig(tp=4, ep=2)
        assert config.effective_ep == 2


# ---------------------------------------------------------------------------
# __post_init__ validation
# ---------------------------------------------------------------------------


class TestValidation:
    def test_valid_config_no_error(self) -> None:
        ParallelConfig(tp=2, pp=2, cp=2, ep=4)

    def test_nproc_not_divisible_by_ep_raises(self) -> None:
        with pytest.raises(ValueError, match="not divisible by effective EP"):
            ParallelConfig(tp=2, pp=1, cp=1, ep=3)

    def test_nproc_not_divisible_by_default_ep_raises(self) -> None:
        with pytest.raises(ValueError, match="not divisible by effective EP"):
            ParallelConfig(tp=3, pp=2, cp=1)


# ---------------------------------------------------------------------------
# from_parsed_args
# ---------------------------------------------------------------------------


class TestFromParsedArgs:
    def test_all_provided(self) -> None:
        config = ParallelConfig.from_parsed_args({"tp": 2, "pp": 4, "cp": 2, "ep": 8, "etp": 2})
        assert config == ParallelConfig(tp=2, pp=4, cp=2, ep=8, etp=2)

    def test_partial_uses_defaults(self) -> None:
        config = ParallelConfig.from_parsed_args({"tp": 4})
        assert config == ParallelConfig(tp=4, pp=1, cp=1, ep=None, etp=1)

    def test_empty_dict_gives_defaults(self) -> None:
        config = ParallelConfig.from_parsed_args({})
        assert config == ParallelConfig()


# ---------------------------------------------------------------------------
# from_run_args
# ---------------------------------------------------------------------------


class TestFromRunArgs:
    def test_from_run_args(self) -> None:
        from miles.utils.debug_utils.run_megatron.cli.commands.args import RunArgs

        args = RunArgs(
            model_type="test",
            hf_checkpoint="/tmp/hf",
            tp=2,
            pp=4,
            cp=1,
            ep=2,
            etp=1,
        )
        config = ParallelConfig.from_run_args(args)
        assert config == ParallelConfig(tp=2, pp=4, cp=1, ep=2, etp=1)


# ---------------------------------------------------------------------------
# __str__
# ---------------------------------------------------------------------------


class TestStr:
    def test_str_repr(self) -> None:
        config = ParallelConfig(tp=2, pp=4, cp=2, ep=8, etp=2)
        text = str(config)
        assert "tp=2" in text
        assert "pp=4" in text
        assert "cp=2" in text
        assert "ep=8" in text
        assert "etp=2" in text
        assert "nproc=16" in text

    def test_str_defaults(self) -> None:
        text = str(ParallelConfig())
        assert "nproc=1" in text


# ---------------------------------------------------------------------------
# dir_name
# ---------------------------------------------------------------------------


class TestDirName:
    def test_tp_only(self) -> None:
        assert ParallelConfig(tp=2).dir_name() == "tp2"

    def test_all_non_default(self) -> None:
        assert ParallelConfig(tp=2, pp=2, cp=2, ep=4, etp=2).dir_name() == "tp2_pp2_cp2_ep4_etp2"

    def test_ep_equal_tp_omitted(self) -> None:
        assert ParallelConfig(tp=4, ep=4).dir_name() == "tp4"

    def test_ep_different_from_tp_included(self) -> None:
        assert ParallelConfig(tp=4, ep=2).dir_name() == "tp4_ep2"

    def test_pp1_cp1_omitted(self) -> None:
        assert ParallelConfig(tp=2, pp=1, cp=1).dir_name() == "tp2"

    def test_defaults_dir_name(self) -> None:
        assert ParallelConfig().dir_name() == "tp1"


# ---------------------------------------------------------------------------
# parse_parallel_args
# ---------------------------------------------------------------------------


class TestParseParallelArgs:
    def test_full_string(self) -> None:
        result = parse_parallel_args("--tp 2 --pp 4 --cp 2 --ep 8 --etp 2")
        assert result == {"tp": 2, "pp": 4, "cp": 2, "ep": 8, "etp": 2}

    def test_partial_string(self) -> None:
        result = parse_parallel_args("--tp 4 --cp 2")
        assert result == {"tp": 4, "cp": 2}

    def test_empty_string(self) -> None:
        result = parse_parallel_args("")
        assert result == {}

    def test_roundtrip_with_from_parsed_args(self) -> None:
        parsed = parse_parallel_args("--tp 2 --pp 4 --cp 2")
        config = ParallelConfig.from_parsed_args(parsed)
        assert config == ParallelConfig(tp=2, pp=4, cp=2)
