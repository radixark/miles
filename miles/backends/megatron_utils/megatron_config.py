import argparse
import copy
import logging
import os
import re
from argparse import Namespace
from pathlib import Path
from typing import Any, Literal

import pydantic
import yaml

from miles.utils.file_arg_utils import resolve_file_arg
from miles.utils.pydantic_utils import FrozenStrictBaseModel
from miles.utils.workers.argv_utils import coerce_dict_to_args, declared_arg_dests
from miles.utils.workers.naming import DNS_LABEL_PATTERN, TRAINER_ID_MAX_LENGTH

logger = logging.getLogger(__name__)


# ---------------------------- constants -----------------------------

ACTOR_ROLE = "actor"
CRITIC_ROLE = "critic"
DEFAULT_MODEL_ROLE = ACTOR_ROLE
TrainerRole = Literal["actor", "critic"]
TRAINER_CHECKPOINT_DIRNAME = "trainers"
MODEL_ID_PATTERN = re.compile(rf"\A{DNS_LABEL_PATTERN}\Z")

PER_POLICY_ARGS: frozenset[str] = frozenset(
    {
        "hf_checkpoint",
        "ref_load",
        "megatron_to_hf_mode",
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

MODEL_DEFINITION_ARGS: frozenset[str] = frozenset(
    {
        "activation_func_clamp_value",
        "add_bias_linear",
        "add_qkv_bias",
        "apply_layernorm_1p",
        "attention_dropout",
        "attention_output_gate",
        "attention_softmax_in_fp32",
        "beta_fast",
        "beta_slow",
        "custom_model_provider_path",
        "dsa_indexer_head_dim",
        "dsa_indexer_n_heads",
        "dsa_indexer_topk",
        "dsv4_compress_ratios",
        "dsv4_compress_rope_theta",
        "dsv4_hc_mult",
        "dsv4_hc_sinkhorn_iters",
        "dsv4_n_hash_layers",
        "dsv4_o_groups",
        "dsv4_o_lora_rank",
        "dsv4_window_size",
        "enable_experimental",
        "experimental_attention_variant",
        "ffn_hidden_size",
        "group_query_attention",
        "hidden_dropout",
        "hidden_size",
        "kv_channels",
        "kv_lora_rank",
        "layernorm_epsilon",
        "make_vocab_size_divisible_by",
        "max_position_embeddings",
        "moe_aux_loss_coeff",
        "moe_ffn_hidden_size",
        "moe_grouped_gemm",
        "moe_latent_size",
        "moe_layer_freq",
        "moe_permute_fusion",
        "moe_router_bias_update_rate",
        "moe_router_dtype",
        "moe_router_enable_expert_bias",
        "moe_router_group_topk",
        "moe_router_load_balancing_type",
        "moe_router_num_groups",
        "moe_router_pre_softmax",
        "moe_router_score_function",
        "moe_router_topk",
        "moe_router_topk_scaling_factor",
        "moe_shared_expert_gate",
        "moe_shared_expert_intermediate_size",
        "moe_token_dispatcher_type",
        "moe_token_drop_policy",
        "mscale",
        "mscale_all_dim",
        "mtp_num_layers",
        "multi_latent_attention",
        "norm_epsilon",
        "normalization",
        "num_attention_heads",
        "num_experts",
        "num_layers",
        "num_query_groups",
        "original_max_position_embeddings",
        "position_embedding_type",
        "post_mlp_layernorm",
        "post_self_attn_layernorm",
        "q_lora_rank",
        "qk_head_dim",
        "qk_layernorm",
        "qk_pos_emb_head_dim",
        "rope_type",
        "rotary_base",
        "rotary_interleaved",
        "rotary_percent",
        "rotary_scaling_factor",
        "seq_length",
        "softmax_type",
        "spec",
        "swiglu",
        "untie_embeddings_and_output_weights",
        "use_rope_scaling",
        "use_rotary_position_embeddings",
        "v_head_dim",
        "vocab_size",
        "window_attn_skip_freq",
        "window_size",
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

    if getattr(args, "use_critic", False):
        assert (
            len({trainer.model_id for trainer in trainers}) == 1
        ), "training several policy models does not support --use-critic"
        trainers = [*trainers, _compute_critic_trainer(args, policy=trainers[0])]

    return trainers


def _compute_critic_trainer(args, *, policy: MegatronTrainerConfig) -> MegatronTrainerConfig:
    model_id = policy.model_id
    return MegatronTrainerConfig(
        trainer_id=CRITIC_ROLE if model_id is None else f"{model_id}-{CRITIC_ROLE}",
        model_id=model_id,
        role=CRITIC_ROLE,
        overrides={**policy.overrides, **_compute_critic_overrides(args)},
    )


def _compute_critic_overrides(args) -> dict[str, Any]:
    return {
        "kl_coef": 0,
        "use_opd": False,
        "disable_param_buffers_cpu_backup": False,
        "load": args.critic_load,
        "save": args.critic_save,
        "lr": args.critic_lr,
        "lr_warmup_iters": args.critic_lr_warmup_iters,
    }


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


# ---------------------------- per policy args -----------------------------


_ROLLOUT_SHARED_ARGS: frozenset[str] = frozenset(
    {"advantage_estimator", "max_tokens_per_gpu", "micro_batch_size", "use_dynamic_batch_size"}
)


def compute_trainer_args(args: Namespace, trainer: MegatronTrainerConfig) -> Namespace:
    # TODO: support policies with different global batch sizes.
    assert "global_batch_size" not in trainer.overrides, (
        f"--megatron-config trainer {trainer.trainer_id!r} overrides global_batch_size; every policy has to "
        f"share the run's"
    )

    # TODO: let policies differ in these once the rollout side reads per-trainer arguments.
    forbidden = _ROLLOUT_SHARED_ARGS & set(trainer.overrides)
    assert not forbidden, (
        f"--megatron-config trainer {trainer.trainer_id!r} overrides {sorted(forbidden)}; every policy has to "
        f"share the run's rollout arguments"
    )

    ans = copy.deepcopy(args)
    ans.trainer_id = trainer.trainer_id
    ans.trainer_model_id = trainer.model_id

    for key, value in trainer.overrides.items():
        assert hasattr(ans, key), (
            f"--megatron-config trainer {trainer.trainer_id!r} overrides {key!r}, which this run's argument "
            f"parser does not know"
        )
        setattr(ans, key, value)

    _apply_critical_derived_overrides(ans, base=args, trainer=trainer)

    if trainer.model_id is not None:
        ans.save = compute_trainer_checkpoint_dir(base_dir=ans.save, trainer_id=trainer.trainer_id)
        ans.load = compute_trainer_checkpoint_dir(base_dir=ans.load, trainer_id=trainer.trainer_id)
        ans.save_hf = compute_trainer_checkpoint_dir(base_dir=ans.save_hf, trainer_id=trainer.trainer_id)

    # TODO: a --use-critic critic keeps the actor's requested_load, so a hot restart reads the actor's checkpoint.
    if args.megatron_config is not None:
        resolve_args_checkpoint_load(ans)

    return ans


def _apply_critical_derived_overrides(ans: Namespace, *, base: Namespace, trainer: MegatronTrainerConfig) -> None:
    # TODO: most derived defaults are still computed from the base args; revisit after the arguments refactor
    if "hf_checkpoint" in trainer.overrides and base.tokenizer_model == base.hf_checkpoint:
        ans.tokenizer_model = ans.hf_checkpoint


# ---------------------------- checkpoint dirs -----------------------------


def compute_trainer_checkpoint_dir(*, base_dir: str | None, trainer_id: str) -> str | None:
    if base_dir is None:
        return None
    return str(Path(base_dir) / TRAINER_CHECKPOINT_DIRNAME / trainer_id)


def resolve_args_checkpoint_load(args: Namespace) -> None:
    # TODO: refactor
    args.requested_load = args.load

    # TODO: During loading, we need to set the start_rollout_id here.
    if args.megatron_to_hf_mode == "bridge":
        # Fresh runs pass a not-yet-created `--load` dir; fall back to the reference
        # weights (loaded via the HF bridge) instead of asserting in load_checkpoint.
        # Mirrors the non-bridge branch below.
        if not _has_megatron_checkpoint(args.load):
            args.load = args.ref_load or args.hf_checkpoint
            args.start_rollout_id = 0
    else:
        if not _has_megatron_checkpoint(args.load):
            args.no_load_optim = True
            args.no_load_rng = True
            args.finetune = True
            args.load = args.ref_load
            if args.ref_ckpt_step is not None:
                args.ckpt_step = args.ref_ckpt_step
            args.start_rollout_id = 0


def _has_megatron_checkpoint(load_dir: str | None) -> bool:
    return (
        load_dir is not None
        and os.path.exists(load_dir)
        and os.path.exists(os.path.join(load_dir, "latest_checkpointed_iteration.txt"))
    )


# ---------------------------- validation -----------------------------


def _assert_valid_trainer_ids(trainer_ids: list[str]) -> None:
    assert len(set(trainer_ids)) == len(trainer_ids), (
        f"--megatron-config trainer ids must be unique, got {trainer_ids}; a trainer id addresses one trainer "
        f"controller and its engine pool, so two entries sharing it would land in the same pool"
    )
    _assert_valid_ids(trainer_ids, kind="trainer")

    too_long_trainer_ids = [trainer_id for trainer_id in trainer_ids if len(trainer_id) > TRAINER_ID_MAX_LENGTH]
    assert not too_long_trainer_ids, (
        f"--megatron-config trainer ids {too_long_trainer_ids} are longer than {TRAINER_ID_MAX_LENGTH} "
        f"characters; shorten them"
    )


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

    parser = get_megatron_arg_parser()
    return coerce_dict_to_args(
        overrides,
        parser=parser,
        allowed_names=PER_POLICY_ARGS | (MODEL_DEFINITION_ARGS & declared_arg_dests(parser)),
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
