import argparse

from miles.utils.chat_template_utils import TITOTokenizerType, resolve_fixed_chat_template
from miles.utils.hf_config import load_hf_config


def add_tinker_arguments(parser):
    group = parser.add_argument_group("Tinker")

    def add_argument(name, **kwargs):
        return group.add_argument(f"--tinker-{name}", **kwargs)

    add_argument("server-host", default="0.0.0.0")
    add_argument("server-port", type=int, default=10613)
    add_argument(
        "base-model",
        help="Model name advertised by the gateway (default: --hf-checkpoint)",
    )
    add_argument(
        "checkpoint-root",
        help="Directory for tinker:// checkpoints (default: <save>/tinker)",
    )
    add_argument(
        "session-ttl-s",
        type=float,
        default=3600.0,
        help="Idle seconds before a recorded /oai/sessions/{sid} is swept, the safety net for agent trials that die before DELETE (default: 3600)",
    )
    add_argument(
        "session-server",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Mount the recorded-session routes /oai/sessions/* for agent harnesses; off = the plain Tinker gateway (default: off)",
    )
    add_argument(
        "session-max-body-bytes",
        type=int,
        default=16 * 1024 * 1024,
        help="Largest JSON body a /oai/sessions route accepts; bodies are parsed on the shared loop (default: 16 MiB)",
    )
    add_argument(
        "tito-model",
        choices=[t.value for t in TITOTokenizerType],
        default=None,
        help="TITO for recorded sessions: each turn's prompt inherits the previous turn's input + output tokens through this miles TITOTokenizer family, whose fixed chat template replaces --chat-template-path (default: off, full re-render every turn)",
    )
    add_argument("train-attn", action=argparse.BooleanOptionalAction, default=True)
    add_argument("train-mlp", action=argparse.BooleanOptionalAction, default=True)
    add_argument("train-unembed", action=argparse.BooleanOptionalAction, default=True)
    return parser


def configure_tinker_args(args):
    assert args.train_backend == "megatron", "Tinker requires the Megatron backend"
    assert (
        args.target_modules is None and args.exclude_modules is None
    ), "Tinker uses --tinker-train-attn/mlp/unembed; --target-modules and --exclude-modules are not supported"
    modules = _resolve_target_modules(
        load_hf_config(args.hf_checkpoint),
        train_attn=args.tinker_train_attn,
        train_mlp=args.tinker_train_mlp,
        train_unembed=args.tinker_train_unembed,
    )
    # The common LoRA validator parses and validates this before trainer/engine initialization.
    args.target_modules = ",".join(modules)
    _configure_tito(args)


def _configure_tito(args):
    """--tinker-tito-model: install the family's fixed chat template and merge its kwargs so both renders agree."""
    if args.tinker_tito_model is None:
        return
    if not args.tinker_session_server:
        raise ValueError(
            "--tinker-tito-model requires --tinker-session-server; TITO only applies to recorded sessions"
        )
    if args.chat_template_path is not None:
        raise ValueError(
            f"--chat-template-path cannot override the template registered for --tinker-tito-model={args.tinker_tito_model}"
        )
    template_path, fixed_kwargs = resolve_fixed_chat_template(args.tinker_tito_model)
    if template_path is not None:
        args.chat_template_path = template_path
    kwargs = dict(args.apply_chat_template_kwargs or {})
    for key, value in fixed_kwargs.items():
        if key in kwargs and kwargs[key] != value:
            raise ValueError(
                f"--apply-chat-template-kwargs {key}={kwargs[key]!r} conflicts with --tinker-tito-model={args.tinker_tito_model}: {value!r}"
            )
        kwargs[key] = value
    args.apply_chat_template_kwargs = kwargs


def _resolve_target_modules(hf_config, *, train_attn, train_mlp, train_unembed):
    # Other architectures need their own complete attention/MLP mapping.
    assert hf_config.model_type in (
        "qwen3",
        "qwen3_moe",
    ), f"Tinker target layout is not defined for model_type={hf_config.model_type!r}"
    modules = []
    if train_attn:
        modules.extend(("q_proj", "k_proj", "v_proj", "o_proj"))
    if train_mlp:
        modules.extend(("gate_proj", "up_proj", "down_proj"))
    if train_unembed:
        modules.append("lm_head")
    assert modules, "Tinker requires at least one trainable LoRA module group"
    return modules
