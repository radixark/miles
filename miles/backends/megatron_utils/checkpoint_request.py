from argparse import Namespace
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Self

from miles.backends.megatron_utils.megatron_config import MegatronArgsNamespace, resolve_args_checkpoint_load
from miles.utils.pydantic_utils import FrozenStrictBaseModel

CHECKPOINT_LOAD_FIELDS = frozenset({"load", "no_load_optim", "no_load_rng", "finetune", "ckpt_step"})


class MegatronCheckpointLoad(FrozenStrictBaseModel):
    load: str | None
    resume_from_ckpt: bool
    no_load_optim: bool | None
    no_load_rng: bool | None
    finetune: bool | None
    ckpt_step: int | None

    @classmethod
    def from_args(cls, args: Namespace) -> Self:
        resolved = Namespace(**vars(args))
        resume_from_ckpt = resolve_args_checkpoint_load(resolved)
        return cls(
            resume_from_ckpt=resume_from_ckpt,
            **{name: vars(resolved)[name] for name in CHECKPOINT_LOAD_FIELDS},
        )

    @contextmanager
    def apply(self, args: MegatronArgsNamespace) -> Iterator[None]:
        previous = {name: vars(args)[name] for name in CHECKPOINT_LOAD_FIELDS if name in vars(args)}
        with args.mutable():
            for name in CHECKPOINT_LOAD_FIELDS:
                setattr(args, name, vars(self)[name])
            try:
                yield
            finally:
                for name in CHECKPOINT_LOAD_FIELDS:
                    if name in previous:
                        setattr(args, name, previous[name])
                    else:
                        delattr(args, name)
