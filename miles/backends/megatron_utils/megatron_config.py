import argparse
import logging
import re
from typing import Any, Literal

import pydantic
import yaml

from miles.utils.file_arg_utils import resolve_file_arg
from miles.utils.pydantic_utils import FrozenStrictBaseModel
from miles.utils.workers.argv_utils import coerce_dict_to_args

logger = logging.getLogger(__name__)


# ---------------------------- constants -----------------------------

ACTOR_ROLE = "actor"
CRITIC_ROLE = "critic"
DEFAULT_MODEL_ROLE = ACTOR_ROLE
TrainerRole = Literal["actor", "critic"]
MODEL_ID_PATTERN = re.compile(r"\A[a-z0-9]([a-z0-9-]*[a-z0-9])?\Z")

PER_POLICY_ARGS: frozenset[str] = frozenset(
    {
        "hf_checkpoint",
        "ref_load",
        "megatron_to_hf_mode",
        "num_layers",
        "hidden_size",
        "ffn_hidden_size",
        "num_attention_heads",
        "group_query_attention",
        "num_query_groups",
        "kv_channels",
        "add_qkv_bias",
        "qk_layernorm",
        "swiglu",
        "normalization",
        "layernorm_epsilon",
        "add_bias_linear",
        "use_rotary_position_embeddings",
        "rotary_base",
        "vocab_size",
        "optimizer",
        "lr",
        "min_lr",
        "lr_decay_style",
        "lr_warmup_iters",
        "lr_warmup_fraction",
        "weight_decay",
        "adam_beta1",
        "adam_beta2",
        "clip_grad",
        "tensor_model_parallel_size",
        "pipeline_model_parallel_size",
        "context_parallel_size",
        "expert_model_parallel_size",
        "expert_tensor_parallel_size",
        "sequence_parallel",
        "global_batch_size",
        "micro_batch_size",
        "max_tokens_per_gpu",
        "use_dynamic_batch_size",
        "advantage_estimator",
        "use_kl_loss",
        "kl_loss_coef",
        "kl_loss_type",
        "entropy_coef",
        "eps_clip",
        "eps_clip_high",
    }
)


# ---------------------------- raw config -----------------------------


class _RawMegatronTrainerConfig(FrozenStrictBaseModel):
    model_id: str
    role: TrainerRole = DEFAULT_MODEL_ROLE
    trainer_id: str | None = None
    overrides: dict[str, Any] = {}


class _RawMegatronConfig(FrozenStrictBaseModel):
    trainers: list[_RawMegatronTrainerConfig] = pydantic.Field(
        validation_alias=pydantic.AliasChoices("trainers", "megatron")
    )

    @classmethod
    def from_file_arg(cls, value: str) -> "_RawMegatronConfig":
        return cls.model_validate(yaml.safe_load(resolve_file_arg(value)))


# ---------------------------- resolved config -----------------------------


class MegatronTrainerConfig(FrozenStrictBaseModel):
    trainer_id: str
    model_id: str | None
    role: TrainerRole
    overrides: dict[str, Any]

    @classmethod
    def resolve(cls, raw: _RawMegatronTrainerConfig) -> "MegatronTrainerConfig":
        return cls(
            trainer_id=raw.trainer_id if raw.trainer_id is not None else f"{raw.model_id}-{raw.role}",
            model_id=raw.model_id,
            role=raw.role,
            overrides=_resolve_overrides(raw.overrides, model_id=raw.model_id),
        )


class MegatronConfig(FrozenStrictBaseModel):
    trainers: list[MegatronTrainerConfig]

    @pydantic.model_validator(mode="after")
    def _validate_ids(self) -> "MegatronConfig":
        _assert_valid_ids(self.model_ids, kind="model")
        _assert_valid_trainer_ids([t.trainer_id for t in self.trainers])
        return self

    @pydantic.model_validator(mode="after")
    def _validate_one_actor_per_model(self) -> "MegatronConfig":
        actor_model_ids = [t.model_id for t in self.trainers if t.role == ACTOR_ROLE]
        assert len(set(actor_model_ids)) == len(actor_model_ids), (
            f"--megatron-config declares several actors for the same model id ({actor_model_ids}); the run "
            f"keys its trainers by model id, so all but the last one would be launched and then ignored"
        )
        return self

    @property
    def model_ids(self) -> list[str]:
        return list(dict.fromkeys(t.model_id for t in self.trainers if t.model_id is not None))

    @property
    def leader_model_id(self) -> str | None:
        return self.trainers[0].model_id

    @property
    def is_multi_policy(self) -> bool:
        return len(set(self.model_ids)) > 1

    def get(self, model_id: str) -> MegatronTrainerConfig:
        for trainer in self.trainers:
            if trainer.model_id == model_id:
                return trainer
        raise KeyError(f"Unknown trainer model id {model_id!r}, known ids: {self.model_ids}")


def resolve_megatron_config(args) -> MegatronConfig:
    return MegatronConfig(trainers=_compute_trainers(args))


def _compute_trainers(args) -> list[MegatronTrainerConfig]:
    if (raw := _resolve_raw_megatron_config(args.megatron_config)) is None:
        trainers = [MegatronTrainerConfig(trainer_id=ACTOR_ROLE, model_id=None, role=ACTOR_ROLE, overrides={})]
    else:
        _assert_no_declared_critic(raw)
        trainers = [MegatronTrainerConfig.resolve(raw=t) for t in raw.trainers]
        assert trainers, "--megatron-config must declare at least one trainer"

    return trainers


def _resolve_raw_megatron_config(value: str | None) -> "_RawMegatronConfig | None":
    if value is None:
        return None
    return _RawMegatronConfig.from_file_arg(value)


def _assert_no_declared_critic(raw: "_RawMegatronConfig") -> None:
    # TODO: accept a declared critic once the critic overrides are applied to it too, not only to a synthesized one
    declared = [t.model_id for t in raw.trainers if t.role == CRITIC_ROLE]
    assert not declared, (
        f"--megatron-config declares a critic for {declared}, which is not supported yet: the critic "
        f"checkpoint, learning rate and neutralized knobs are only applied to the critic the run "
        f"synthesizes itself from --use-critic"
    )


# ---------------------------- validation -----------------------------


def _assert_valid_trainer_ids(trainer_ids: list[str]) -> None:
    assert len(set(trainer_ids)) == len(trainer_ids), (
        f"--megatron-config trainer ids must be unique, got {trainer_ids}; a trainer id addresses one trainer "
        f"controller and its engine pool, so two entries sharing it would land in the same pool"
    )
    _assert_valid_ids(trainer_ids, kind="trainer")


def _assert_valid_ids(ids: list[str], *, kind: str) -> None:
    bad_ids = [identifier for identifier in ids if MODEL_ID_PATTERN.match(identifier) is None]
    assert not bad_ids, (
        f"--megatron-config {kind} ids {bad_ids} are not usable as Kubernetes pool names or path components: "
        f"each must match {MODEL_ID_PATTERN.pattern}"
    )


# ---------------------------- override coercion -----------------------------


def _resolve_overrides(overrides: dict[str, Any], *, model_id: str) -> dict[str, Any]:
    if not overrides:
        return overrides

    return coerce_dict_to_args(
        overrides,
        parser=get_megatron_arg_parser(),
        allowed_names=PER_POLICY_ARGS,
        context=f"--megatron-config model {model_id!r}",
    )


def get_megatron_arg_parser() -> argparse.ArgumentParser:
    # TODO: revisit once the args refactor lands; this throwaway construction may then be optimized
    from miles.backends.megatron_utils.arguments import parse_args
    from miles.utils.arguments import get_miles_extra_args_provider

    class ParserCaptured(Exception):
        def __init__(self, parser: argparse.ArgumentParser) -> None:
            super().__init__("the throwaway parser was captured before anything was parsed")
            self.parser = parser

    def capture(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        raise ParserCaptured(get_miles_extra_args_provider()(parser))

    try:
        parse_args(extra_args_provider=capture)
    except ParserCaptured as captured:
        return captured.parser
    raise AssertionError(
        "megatron's parse_args returned without calling the extra args provider, so the arguments this "
        "run declares could not be read; --megatron-config overrides cannot be typed"
    )
