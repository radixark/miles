import asyncio
from collections.abc import ContextManager
from contextlib import nullcontext
from types import SimpleNamespace

from tests.fast.ray.rollout.conftest import make_args, make_sample

from miles.rollout import sglang_rollout


class _LegacyGenerateState:
    def __init__(self) -> None:
        self.aborted = False
        self.semaphore = asyncio.Semaphore(1)

    def dp_rank_context(self) -> ContextManager[None]:
        return nullcontext()


class TestLegacyRolloutSampleOwnership:
    async def test_generate_stamps_compact_rows_with_the_issued_source(self, monkeypatch) -> None:
        """The legacy generate boundary preserves one issued source across compact rows."""
        source = make_sample(index=7, reward=0.0)
        rows = [make_sample(index=7, reward=0.0), make_sample(index=7, reward=0.0)]
        args = make_args(
            partial_rollout=False,
            mask_offpolicy_in_partial_rollout=False,
            group_rm=True,
            custom_generate_function_path="compact",
        )
        monkeypatch.setattr(sglang_rollout, "GenerateState", lambda _args: _LegacyGenerateState())

        async def generate(_input):
            return SimpleNamespace(samples=rows)

        monkeypatch.setattr(sglang_rollout, "load_generate_function", lambda _path: generate)

        output = await sglang_rollout.generate_and_rm(args, source, {}, evaluation=False)

        assert [
            (
                row.lineage.source_sample_index,
                row.lineage.output_index,
                row.lineage.output_count,
            )
            for row in output
        ] == [
            (7, 0, 2),
            (7, 1, 2),
        ]
