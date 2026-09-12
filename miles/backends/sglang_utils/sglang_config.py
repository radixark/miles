"""Configuration models for SGLang engine deployment."""

import logging
from dataclasses import dataclass
from typing import Literal

import pydantic
import yaml

from miles.backends.sglang_utils.arguments import collect_eval_sglang_overrides
from miles.utils.file_arg_utils import resolve_file_arg
from miles.utils.pydantic_utils import FrozenStrictBaseModel

logger = logging.getLogger(__name__)


# ---------------------------- raw config -----------------------------


class _RawServerGroupConfig(FrozenStrictBaseModel):
    """Configuration for a single server group.

    Attributes:
        worker_type: One of "regular", "prefill", "decode", or "placeholder".
                     "placeholder" reserves GPU slots without creating engines.
        num_gpus: Total number of GPUs for this group.
        num_gpus_per_engine: GPUs per engine for this group.  Overrides the
                             model-level or global ``--rollout-num-gpus-per-engine``.
        overrides: Optional dict of SGLang ``ServerArgs`` field overrides.
                   These are applied on top of the base CLI ``--sglang-*``
                   arguments in ``_compute_server_args``.
    """

    worker_type: Literal["regular", "prefill", "decode", "placeholder"]
    num_gpus: int
    num_gpus_per_engine: int | None = None
    overrides: dict = {}


class _RawModelConfig(FrozenStrictBaseModel):
    """Configuration for a single model deployment.

    Attributes:
        name: Unique name for this model (e.g. "actor", "reward").
        model_path: HF checkpoint path.  Falls back to ``args.hf_checkpoint``.
        num_gpus_per_engine: Default GPUs per engine for all groups in this
                             model.  Individual groups can override.
        server_groups: Server group configurations for this model.
        update_weights: Whether this model receives weight updates from
                        training.  Set to ``False`` for frozen models
                        (reference, reward, etc.).  When ``None`` (default),
                        automatically inferred in ``resolve()``: ``True`` if
                        model_path matches ``args.hf_checkpoint``, ``False``
                        otherwise.
    """

    name: str
    model_path: str | None = None
    num_gpus_per_engine: int | None = None
    server_groups: list[_RawServerGroupConfig] = pydantic.Field(
        default_factory=list,
        validation_alias=pydantic.AliasChoices("server_groups", "engine_groups"),
    )
    update_weights: bool | None = None

    @property
    def total_num_gpus(self) -> int:
        return sum(g.num_gpus for g in self.server_groups)


class _RawSglangConfig(FrozenStrictBaseModel):
    """Configuration for SGLang engine deployment.

    Loaded from ``--sglang-config``: either a YAML file path or an inline ``base64:`` payload.

    **Config format**::

        sglang:
          - name: actor
            model_path: /path/to/actor
            update_weights: true          # receives training weight updates (default)
            num_gpus_per_engine: 2
            server_groups:
              - worker_type: prefill
                num_gpus: 4
                num_gpus_per_engine: 2
              - worker_type: decode
                num_gpus: 8
                num_gpus_per_engine: 4
          - name: ref
            model_path: /path/to/ref
            update_weights: false          # frozen, no weight updates
            server_groups:
              - worker_type: regular
                num_gpus: 4

    Each model gets its own router.  ``placeholder`` groups reserve GPU
    slots without creating engines.  ``overrides`` are ``ServerArgs``
    field names applied on top of the base ``--sglang-*`` CLI args.

    Set ``update_weights: false`` for frozen models (reference, reward,
    etc.) that should not receive weight updates from training.

    .. note::

       ``engine_groups`` is accepted as a backward-compatible alias for
       ``server_groups`` in the YAML config.
    """

    models: list[_RawModelConfig] = pydantic.Field(validation_alias=pydantic.AliasChoices("models", "sglang"))

    @classmethod
    def from_file_arg(cls, value: str) -> "_RawSglangConfig":
        return cls.model_validate(yaml.safe_load(resolve_file_arg(value)))

    @staticmethod
    def from_prefill_num_servers(args) -> "_RawSglangConfig":
        """Build a config equivalent to the legacy --prefill-num-servers flag."""
        total_gpus = args.rollout_num_gpus
        prefill_gpus = args.prefill_num_servers * args.rollout_num_gpus_per_engine
        decode_gpus = total_gpus - prefill_gpus
        assert decode_gpus > 0, f"No decode GPUs: total {total_gpus}, prefill {prefill_gpus}"
        return _RawSglangConfig(
            models=[
                _RawModelConfig(
                    name="default",
                    server_groups=[
                        _RawServerGroupConfig(worker_type="prefill", num_gpus=prefill_gpus),
                        _RawServerGroupConfig(worker_type="decode", num_gpus=decode_gpus),
                    ],
                )
            ]
        )

    @property
    def total_num_gpus(self) -> int:
        return sum(m.total_num_gpus for m in self.models)


# ---------------------------- resolved config -----------------------------


class ServerGroupConfig(FrozenStrictBaseModel):
    worker_type: Literal["regular", "prefill", "decode", "placeholder"]
    num_gpus: int = pydantic.Field(gt=0)
    num_gpus_per_engine: int = pydantic.Field(gt=0)
    gpu_offset: int = pydantic.Field(ge=0)
    engine_offset: int = pydantic.Field(ge=0)
    overrides: dict = pydantic.Field(default_factory=dict)
    needs_offload: bool

    @property
    def model_path(self) -> str:
        return self.overrides["model_path"]

    @classmethod
    def resolve(
        cls,
        raw: _RawServerGroupConfig,
        args,
        default_gpus_per_engine: int,
        default_model_path: str,
        offset_cursor: "_OffsetCursor",
    ) -> "ServerGroupConfig":
        assert not ({"host", "port", "gated_launch_port"} & set(raw.overrides)), (
            f"sglang_overrides must not override host/port ({raw.overrides=}): the rollout process derives "
            f"each engine's url from the addr allocator, so an override would make it talk to the wrong endpoint"
        )

        rollout_pg_offset = _compute_rollout_offset(args)
        megatron_num_gpus = _compute_megatron_num_gpus(args)

        gpu_offset = offset_cursor.gpu
        group_abs_start = rollout_pg_offset + gpu_offset
        needs_offload = args.offload_rollout and group_abs_start < megatron_num_gpus

        ans = cls(
            worker_type=raw.worker_type,
            num_gpus=raw.num_gpus,
            num_gpus_per_engine=x if (x := raw.num_gpus_per_engine) is not None else default_gpus_per_engine,
            gpu_offset=gpu_offset,
            engine_offset=offset_cursor.engine,
            overrides={
                "model_path": default_model_path,
                **({"enable_memory_saver": False} if args.offload_rollout and not needs_offload else {}),
                **raw.overrides,
            },
            needs_offload=needs_offload,
        )

        offset_cursor.gpu += raw.num_gpus
        offset_cursor.engine += raw.num_gpus // min(ans.num_gpus_per_engine, args.num_gpus_per_node)
        return ans


class ModelConfig(FrozenStrictBaseModel):
    name: str
    model_path: str | None
    server_groups: list[ServerGroupConfig]
    update_weights: bool

    @classmethod
    def resolve(cls, raw: _RawModelConfig, args, offset_cursor: "_OffsetCursor") -> "ModelConfig":
        """Resolve per-group defaults from model-level then args-level values."""
        default_model_path = p if (p := raw.model_path) is not None else args.hf_checkpoint
        default_gpus_per_engine = n if (n := raw.num_gpus_per_engine) is not None else args.rollout_num_gpus_per_engine
        server_groups = [
            ServerGroupConfig.resolve(
                g,
                args,
                default_gpus_per_engine=default_gpus_per_engine,
                default_model_path=default_model_path,
                offset_cursor=offset_cursor,
            )
            for g in raw.server_groups
        ]

        if server_groups:
            model_paths = {g.overrides["model_path"] for g in server_groups}
            assert len(model_paths) == 1, (
                f"Model '{raw.name}' has server groups with different model_path values: "
                f"{model_paths}. All server groups within a model must use the same model_path."
            )
            effective_model_path = model_paths.pop()
        else:
            effective_model_path = default_model_path

        update_weights = raw.update_weights
        if update_weights is None:
            if effective_model_path != args.hf_checkpoint:
                logger.warning(
                    f"Model '{raw.name}' uses model_path='{effective_model_path}' which differs "
                    f"from hf_checkpoint='{args.hf_checkpoint}'. Defaulting update_weights to False. "
                    f"Set update_weights explicitly in the config to suppress this warning."
                )
                update_weights = False
            else:
                update_weights = True

        return cls(
            name=raw.name,
            model_path=raw.model_path,
            server_groups=server_groups,
            update_weights=update_weights,
        )

    @property
    def has_pd_disaggregation(self) -> bool:
        return any(g.worker_type in ("prefill", "decode") for g in self.server_groups)

    @property
    def num_server_cells(self) -> int:
        return sum(
            group.num_gpus // group.num_gpus_per_engine
            for group in self.server_groups
            if group.worker_type != "placeholder"
        )


class SglangConfig(FrozenStrictBaseModel):
    models: list[ModelConfig]

    @classmethod
    def resolve(cls, raw: _RawSglangConfig, args) -> "SglangConfig":
        offset_cursor = _OffsetCursor(gpu=0, engine=0)
        model_configs = [ModelConfig.resolve(m, args, offset_cursor) for m in raw.models]

        assert offset_cursor.gpu == raw.total_num_gpus
        return cls(models=model_configs)

    @property
    def has_pd_disaggregation(self) -> bool:
        return any(m.has_pd_disaggregation for m in self.models)


@dataclass
class _OffsetCursor:
    gpu: int
    engine: int


def resolve_sglang_config(args) -> SglangConfig:
    """Build a SglangConfig from args, choosing the right source."""
    raw = _compute_raw_sglang_config(args)
    return SglangConfig.resolve(raw, args)


def _compute_raw_sglang_config(args) -> _RawSglangConfig:
    if args.debug_train_only:
        return _RawSglangConfig(models=[])

    eval_num_gpus = args.eval_num_gpus

    if getattr(args, "sglang_config", None) is not None:
        config = _RawSglangConfig.from_file_arg(args.sglang_config)
        expected = args.rollout_num_gpus + eval_num_gpus
        actual = config.total_num_gpus
        assert (
            actual == expected
        ), f"sglang_config total GPUs ({actual}) != rollout_num_gpus + eval_num_gpus ({expected})"
        if eval_num_gpus == 0:
            return config
        eval_models = [m for m in config.models if m.name == "eval"]
        assert len(eval_models) == 1 and eval_models[0].total_num_gpus == eval_num_gpus, (
            f"--eval-num-gpus {eval_num_gpus} requires the sglang_config YAML to contain "
            f"exactly one model named 'eval' with that many GPUs."
        )
        return _RawSglangConfig(
            models=[_compute_eval_raw_model(m, args) if m.name == "eval" else m for m in config.models]
        )

    if args.prefill_num_servers is not None:
        config = _RawSglangConfig.from_prefill_num_servers(args)
    else:
        config = _RawSglangConfig(
            models=[
                _RawModelConfig(
                    name="default",
                    server_groups=[_RawServerGroupConfig(worker_type="regular", num_gpus=args.rollout_num_gpus)],
                )
            ]
        )

    if eval_num_gpus == 0:
        return config

    eval_model = _compute_eval_raw_model(
        _RawModelConfig(
            name="eval",
            server_groups=[_RawServerGroupConfig(worker_type="regular", num_gpus=eval_num_gpus)],
        ),
        args,
    )
    return _RawSglangConfig(models=[*config.models, eval_model])


def _eval_sglang_overrides(args) -> dict:
    """Eval-fleet engine settings; anything absent is inherited from the rollout engines."""
    overrides = {
        # Eval samples never feed training, so the replay side-channels are pure overhead.
        "enable_return_routed_experts": False,
        "enable_return_indexer_topk": False,
    }
    if args.eval_num_gpus_per_engine != args.rollout_num_gpus_per_engine:
        # Inheriting these across a different tp gives an engine SGLang refuses to boot.
        tp_coupled = ("dp_size", "pp_size", "ep_size", "attn_cp_size")
        overrides |= dict.fromkeys(tp_coupled, 1)
        logger.info(
            f"Eval tp={args.eval_num_gpus_per_engine} != rollout tp={args.rollout_num_gpus_per_engine}; "
            f"{', '.join(tp_coupled)} default to 1. Override with --eval-sglang-*."
        )
    return overrides | collect_eval_sglang_overrides(args)


def _compute_eval_raw_model(raw: _RawModelConfig, args) -> _RawModelConfig:
    """Fill the eval model from the ``--eval-*`` args: YAML > ``--eval-sglang-*`` > ``--sglang-*``."""
    overrides = _eval_sglang_overrides(args)
    return raw.model_copy(
        update=dict(
            # Never joins the training broadcast group; the fleet is synced by snapshot only.
            update_weights=False if raw.update_weights is None else raw.update_weights,
            server_groups=[
                group.model_copy(
                    update=dict(
                        num_gpus_per_engine=group.num_gpus_per_engine or args.eval_num_gpus_per_engine,
                        overrides=overrides | group.overrides,
                    )
                )
                for group in raw.server_groups
            ],
        )
    )


def _compute_rollout_offset(args) -> int:
    """Offset (in PG bundle slots) where rollout GPUs start."""
    if args.debug_train_only or args.debug_rollout_only or args.colocate:
        return 0
    if getattr(args, "critic_train_only", False):
        return args.critic_num_nodes * args.critic_num_gpus_per_node
    offset = args.actor_num_nodes * args.actor_num_gpus_per_node
    return offset


def _compute_megatron_num_gpus(args) -> int:
    """Total number of megatron (actor + critic) GPU slots in the placement group."""
    if getattr(args, "debug_rollout_only", False):
        return 0
    if getattr(args, "critic_train_only", False):
        return args.critic_num_nodes * args.critic_num_gpus_per_node
    num = args.actor_num_nodes * args.actor_num_gpus_per_node
    return num
