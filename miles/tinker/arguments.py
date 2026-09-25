import argparse

from miles.utils.chat_template_utils import TITOTokenizerType, configure_fixed_chat_template
from miles.utils.lora.hf_lora_targets import LORA_TARGET_GROUPS, parse_lora_targets


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
    add_argument(
        "session-strict-truncation",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Refuse (409) a recorded chat turn that continues a reply cut at max_tokens, as the miles session server v2 does (default: off)",
    )
    return parser


def configure_tinker_args(args):
    assert args.train_backend == "megatron", "Tinker requires the Megatron backend"
    assert args.exclude_modules is None, "Tinker selects complete training groups; --exclude-modules is not supported"
    groups = parse_lora_targets(args.target_modules)
    if groups is None:
        groups = list(LORA_TARGET_GROUPS)
    assert set(groups) <= set(LORA_TARGET_GROUPS), "Tinker --target-modules accepts only attn,mlp,unembed groups"
    args.tinker_lora_groups = groups
    args.target_modules = groups
    _configure_tito(args)


def _configure_tito(args):
    """--tinker-tito-model: install the family's fixed chat template and merge its kwargs so both renders agree."""
    if args.tinker_tito_model is None:
        return
    if not args.tinker_session_server:
        raise ValueError(
            "--tinker-tito-model requires --tinker-session-server; TITO only applies to recorded sessions"
        )
    configure_fixed_chat_template(args, args.tinker_tito_model, option="--tinker-tito-model")
