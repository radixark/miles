from __future__ import annotations

import dataclasses

import pytest

from miles.utils.debug_utils.run_megatron.cli.parallel_utils import (
    ParallelConfig,
    parse_parallel_args,
)


class TestParallelConfigDefaults:
    def test_frozen(self) -> None:
        config = ParallelConfig()
        with pytest.raises(dataclasses.FrozenInstanceError):
            config.tp = 2  # type: ignore[misc]

    def test_defaults(self) -> None:
        config = ParallelConfig()
        assert config.tp == 1
        assert config.pp == 1
        assert config.cp == 1
        assert config.ep is None
        assert config.etp == 1


class TestEffectiveEp:
    def test_ep_none_falls_back_to_tp(self) -> None:
        config = ParallelConfig(tp=4)
        assert config.effective_ep == 4

    def test_ep_explicit(self) -> None:
        config = ParallelConfig(tp=4, ep=2)
        assert config.effective_ep == 2

    def test_ep_1_is_not_none(self) -> None:
        config = ParallelConfig(tp=4, ep=1)
        assert config.effective_ep == 1


class TestNproc:
    def test_all_ones(self) -> None:
        assert ParallelConfig().nproc == 1

    def test_tp_pp_cp(self) -> None:
        config = ParallelConfig(tp=2, pp=3, cp=4)
        assert config.nproc == 24

    def test_ep_etp_dont_affect_nproc(self) -> None:
        config = ParallelConfig(tp=2, ep=2, etp=2)
        assert config.nproc == 2


class TestPostInit:
    def test_valid_nproc_divisible_by_ep(self) -> None:
        ParallelConfig(tp=4, pp=1, cp=1, ep=2)

    def test_invalid_nproc_not_divisible_by_ep(self) -> None:
        with pytest.raises(ValueError, match="not divisible by effective EP"):
            ParallelConfig(tp=3, pp=1, cp=1, ep=2)

    def test_ep_none_uses_tp_for_validation(self) -> None:
        # nproc=tp*pp*cp=2, effective_ep=tp=2, 2%2=0 → valid
        ParallelConfig(tp=2)

    def test_ep_none_invalid(self) -> None:
        # nproc=tp*pp*cp=2*3*1=6, effective_ep=tp=2, 6%2=0 → valid
        ParallelConfig(tp=2, pp=3)
        # nproc=3*1*1=3, effective_ep=3, 3%3=0 → valid
        ParallelConfig(tp=3)


class TestDirName:
    def test_tp_only(self) -> None:
        # tp is never skipped (skip value is None), pp=1 skipped, cp=1 skipped,
        # ep=None skipped, etp=1 skipped
        name = ParallelConfig(tp=2).dir_name()
        assert name == "tp2"

    def test_tp_cp(self) -> None:
        name = ParallelConfig(tp=2, cp=4).dir_name()
        assert name == "tp2_cp4"

    def test_ep_equals_tp_is_skipped(self) -> None:
        # ep == tp → skipped
        name = ParallelConfig(tp=4, ep=4).dir_name()
        assert "ep" not in name

    def test_ep_differs_from_tp(self) -> None:
        name = ParallelConfig(tp=4, ep=2).dir_name()
        assert "ep2" in name

    def test_pp_1_skipped(self) -> None:
        name = ParallelConfig(tp=2, pp=1).dir_name()
        assert "pp" not in name

    def test_pp_gt1_included(self) -> None:
        name = ParallelConfig(tp=2, pp=2).dir_name()
        assert "pp2" in name

    def test_etp_gt1_included(self) -> None:
        name = ParallelConfig(tp=2, etp=2).dir_name()
        assert "etp2" in name

    def test_all_fields(self) -> None:
        name = ParallelConfig(tp=2, pp=3, cp=4, ep=8, etp=2).dir_name()
        assert name == "tp2_pp3_cp4_ep8_etp2"

    def test_default_config(self) -> None:
        # tp=1, pp=1, cp=1, ep=None, etp=1
        # tp=1 is not None so included; pp=1 skipped; cp=1 skipped; ep=None skipped; etp=1 skipped
        name = ParallelConfig().dir_name()
        assert name == "tp1"


class TestFromParsedArgs:
    def test_full_args(self) -> None:
        parsed = {"tp": 2, "pp": 3, "cp": 4, "ep": 8, "etp": 2}
        config = ParallelConfig.from_parsed_args(parsed)
        assert config.tp == 2
        assert config.pp == 3
        assert config.cp == 4
        assert config.ep == 8
        assert config.etp == 2

    def test_partial_args_use_defaults(self) -> None:
        parsed = {"tp": 4}
        config = ParallelConfig.from_parsed_args(parsed)
        assert config.tp == 4
        assert config.pp == 1
        assert config.cp == 1
        assert config.ep is None
        assert config.etp == 1

    def test_empty_args(self) -> None:
        config = ParallelConfig.from_parsed_args({})
        assert config == ParallelConfig()


class TestFromRunArgs:
    def test_extracts_fields(self) -> None:
        from miles.utils.debug_utils.run_megatron.cli.commands.args import RunArgs

        args = RunArgs(
            model_type="deepseek_v3",
            hf_checkpoint="/fake",  # type: ignore[arg-type]
            tp=4, pp=2, cp=2, ep=8, etp=2,
        )
        config = ParallelConfig.from_run_args(args)
        assert config.tp == 4
        assert config.pp == 2
        assert config.cp == 2
        assert config.ep == 8
        assert config.etp == 2


class TestStr:
    def test_contains_all_fields(self) -> None:
        config = ParallelConfig(tp=2, pp=3, cp=4, ep=8, etp=2)
        s = str(config)
        assert "tp=2" in s
        assert "pp=3" in s
        assert "cp=4" in s
        assert "ep=8" in s
        assert "etp=2" in s
        assert "nproc=24" in s


class TestParseParallelArgs:
    def test_basic(self) -> None:
        result = parse_parallel_args("--tp 2 --cp 4")
        assert result == {"tp": 2, "cp": 4}

    def test_all_flags(self) -> None:
        result = parse_parallel_args("--tp 2 --pp 3 --cp 4 --ep 8 --etp 2")
        assert result == {"tp": 2, "pp": 3, "cp": 4, "ep": 8, "etp": 2}

    def test_empty_string(self) -> None:
        result = parse_parallel_args("")
        assert result == {}

    def test_omitted_flags_excluded(self) -> None:
        result = parse_parallel_args("--tp 2")
        assert "pp" not in result
        assert "cp" not in result
