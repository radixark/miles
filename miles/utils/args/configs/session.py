from typing import ClassVar

from miles.utils.args.schema import A, Arg, BaseConfig
from miles.utils.chat_template_utils.tito_tokenizer import TITOTokenizerType


class SessionConfig(BaseConfig):
    _mutable_fields: ClassVar[frozenset[str]] = frozenset({"session_server_addrs", "session_server_instance_ids"})

    session_server_addrs: list[str] | None = None
    session_server_instance_ids: dict[str, str] | None = None
    use_session_server: A[
        str | bool,
        Arg(
            type_parser=None,
            nargs="?",
            const=True,
            help="Start a standalone session server for TITO/session support. "
            "Requires --hf-checkpoint. A named --tito-model resolves its registered template; "
            "--tito-model=default uses the checkpoint-native or explicit --chat-template-path template. "
            "Bare flag (or 'v1') selects the append-only linear v1 server; "
            "'--use-session-server v2' selects the tree-serving v2 "
            "(multi-lineage trajectories, always-branch).",
        ),
    ] = False
    session_server_workers: A[int, Arg(help="Number of session server instances.")] = 32
    session_server_ip: A[
        str | None,
        Arg(
            help=(
                "Address the session servers bind to, e.g. 0.0.0.0 to accept traffic from outside "
                "the cluster. Peers still reach them on the address their worker was placed on. "
                "Defaults to that placed address."
            )
        ),
    ] = None
    session_server_port: A[
        int | None,
        Arg(
            help=(
                "Base port for the session servers, so a network policy can whitelist a known range. "
                "Instance i listens on this port plus i. Defaults to a dynamically allocated port."
            )
        ),
    ] = None
    tito_model: A[
        str,
        Arg(
            choices=[t.value for t in TITOTokenizerType],
            help="TITO tokenizer type for pretokenized prefix reuse. "
            "Controls how token IDs are computed for messages appended after "
            "the pretokenized prefix in multi-turn agentic sessions.",
        ),
    ] = "default"
    session_message_matcher: A[
        str,
        Arg(
            help=(
                "Process-wide session history matcher: strict (default), "
                "loose_tool_call, role_content_only, or a trusted dotted import "
                "path. role_content_only is a high-risk opt-in that can collapse "
                "different tool-call lineages and does not reconcile call IDs."
            )
        ),
    ] = "strict"
    session_sample_picker_path: A[
        str,
        Arg(
            help=(
                "v2 only. Import path of the sample-pick hook for the "
                "session samples op: fn(leaf_samples, session_metadata) -> "
                "list[Sample], a pure selection over the per-leaf raw samples. "
                "Runs synchronously inside the session server process; long CPU "
                "work stalls every session on the instance. Default: the "
                "temporal-supersession retry trim."
            )
        ),
    ] = "miles.rollout.session.v2.picker_hub.drop_retries"
    session_sample_postprocessor_path: A[
        str,
        Arg(
            help=(
                "v2 only. Import path of the post-process hook for the "
                "session samples op: fn(leaf_samples, session_metadata) -> "
                "list[Sample], finalizing loss masks / rewards over the picked "
                "samples. Runs synchronously inside the session server process. "
                "Default: exactly-once completion masking + rewards keyed by "
                "response id."
            )
        ),
    ] = "miles.rollout.session.v2.postprocessor_hub.default_postprocess"
