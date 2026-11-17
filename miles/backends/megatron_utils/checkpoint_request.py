from collections.abc import Iterator
from contextlib import contextmanager

from typing_extensions import Self

from miles.backends.megatron_utils.megatron_config import MegatronArgsNamespace
from miles.utils.pydantic_utils import FrozenStrictBaseModel

CHECKPOINT_LOAD_FIELDS = frozenset({"load", "no_load_optim", "no_load_rng", "finetune", "ckpt_step"})


class MegatronCkptLoadConfig(FrozenStrictBaseModel):
    load: str | None
    resume_from_ckpt: bool
    no_load_optim: bool | None
    no_load_rng: bool | None
    finetune: bool | None
    ckpt_step: int | None

    @classmethod
    def for_peer_recovery(cls) -> Self:
        return cls(
            load=None,
            resume_from_ckpt=True,
            no_load_optim=False,
            no_load_rng=False,
            finetune=False,
            ckpt_step=None,
        )

    @contextmanager
    def apply(self, args: MegatronArgsNamespace) -> Iterator[None]:
        assert CHECKPOINT_LOAD_FIELDS.isdisjoint(vars(args)), "Checkpoint load fields must be runtime-only"
        assert "requested_load" not in vars(args), "requested_load belongs to the orchestrator"
        with args.mutable():
            for name in CHECKPOINT_LOAD_FIELDS:
                setattr(args, name, vars(self)[name])
            try:
                yield
            finally:
                for name in CHECKPOINT_LOAD_FIELDS:
                    delattr(args, name)
                assert CHECKPOINT_LOAD_FIELDS.isdisjoint(vars(args)), "Checkpoint load fields must be runtime-only"
                assert "requested_load" not in vars(args), "requested_load belongs to the orchestrator"
