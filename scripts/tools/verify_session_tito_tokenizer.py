#!/usr/bin/env python3
"""CLI: run the multi-role TITO session-server verifier against a real model.

Boots the miles rollout pipeline (sglang + miles-router) under
``--debug-rollout-only`` and runs two rollout steps (see
``miles.utils.test_utils.session_verify_agent``): the spec rollout probes every
continuation the selected family's ``FixedTemplate`` continuation roles accept
or refuse, and the normal rollout answers tool calls with tool messages and
answers with user questions.  PASS iff every sample completes without HTTP
error from the server-side prefix check, every spec probe gets the status the
spec decides, and the per-sample mismatch gates hold.

This script is a thin entrypoint over miles' canonical ``parse_args``: all
flags are miles' canonical flags (``--rollout-num-gpus-per-engine`` instead of
the old ``--tp-size``, ``--actor-num-gpus-per-node`` instead of ``--num-gpus``,
``--n-samples-per-prompt`` instead of ``--n-samples``, ``--sglang-reasoning-parser``
instead of ``--reasoning-parser``, etc.).  The wrapper-only knobs are
``--assistant-text-threshold`` (post-process gate on per-sample metrics) and
``--no-session-verify-spec-trajectory`` (run the normal rollout only).

Usage examples::

    # GLM-4.7-Flash with tool + user + system surface, single-node, TP=4
    python scripts/tools/verify_session_tito_tokenizer.py \\
        --hf-checkpoint zai-org/GLM-4.7-Flash \\
        --tito-model glm47 \\
        --sglang-reasoning-parser glm45 \\
        --sglang-tool-call-parser glm47 \\
        --rollout-num-gpus-per-engine 4

    # Qwen3-4B with tool + user surface, single-node, TP=1
    python scripts/tools/verify_session_tito_tokenizer.py \\
        --hf-checkpoint Qwen/Qwen3-4B \\
        --tito-model qwen3 \\
        --sglang-reasoning-parser qwen3 \\
        --sglang-tool-call-parser qwen25 \\
        --rollout-num-gpus-per-engine 1

    # Multi-node example: ray cluster must already be up across N nodes
    # (e.g. via rcli / slurm) and MILES_SCRIPT_EXTERNAL_RAY=1 set so
    # execute_train skips its head-only ray start.
    MILES_SCRIPT_EXTERNAL_RAY=1 \\
    python scripts/tools/verify_session_tito_tokenizer.py \\
        --hf-checkpoint zai-org/GLM-4.7-Flash \\
        --tito-model glm47 \\
        --sglang-reasoning-parser glm45 \\
        --sglang-tool-call-parser glm47 \\
        --rollout-num-gpus-per-engine 4 \\
        --actor-num-nodes 2 --actor-num-gpus-per-node 8
"""

from __future__ import annotations

import argparse
import logging
import sys

from miles.utils.args.runtime import AllConfig
from miles.utils.arguments import parse_args
from miles.utils.test_utils.session_verify_agent import fixed_template, spec_probe_steps
from miles.utils.test_utils.session_verify_runner import (
    SESSION_VERIFY_INVARIANT_ARGS,
    run_session_verify,
    session_verify_extras,
    session_verify_kinds,
)


def _with_session_verify_defaults(argv: list[str]) -> list[str]:
    defaults = [
        "--prompt-data",
        SESSION_VERIFY_INVARIANT_ARGS["prompt_data"],
        "--input-key",
        SESSION_VERIFY_INVARIANT_ARGS["input_key"],
        # parse_args needs a rollout count; the runner derives the run's own from its rollouts.
        "--num-rollout",
        "1",
        "--rollout-batch-size",
        str(SESSION_VERIFY_INVARIANT_ARGS["rollout_batch_size"]),
        "--rollout-max-response-len",
        str(SESSION_VERIFY_INVARIANT_ARGS["rollout_max_response_len"]),
        "--rollout-temperature",
        str(SESSION_VERIFY_INVARIANT_ARGS["rollout_temperature"]),
        "--global-batch-size",
        str(SESSION_VERIFY_INVARIANT_ARGS["global_batch_size"]),
        "--rm-type",
        SESSION_VERIFY_INVARIANT_ARGS["rm_type"],
        "--custom-generate-function-path",
        SESSION_VERIFY_INVARIANT_ARGS["custom_generate_function_path"],
        "--custom-agent-function-path",
        SESSION_VERIFY_INVARIANT_ARGS["custom_agent_function_path"],
        "--use-session-server",
        SESSION_VERIFY_INVARIANT_ARGS["use_session_server"],
        "--train-backend",
        SESSION_VERIFY_INVARIANT_ARGS["train_backend"],
        "--sglang-ep-size",
        str(SESSION_VERIFY_INVARIANT_ARGS["sglang_ep_size"]),
    ]
    for key in ("debug_rollout_only", "ci_test", "colocate"):
        if SESSION_VERIFY_INVARIANT_ARGS[key]:
            defaults.append("--" + key.replace("_", "-"))
    return [*defaults, *argv]


def _parse_session_verify_args() -> argparse.Namespace:
    wrapper_values = {}

    def extract_wrapper_args(args: argparse.Namespace) -> None:
        for name in ("assistant_text_threshold", "enable_spec", "session_verify_spec_trajectory"):
            wrapper_values[name] = vars(args).pop(name)

    config = parse_args(add_custom_arguments=session_verify_extras, preprocess_args=extract_wrapper_args)
    return _runner_namespace(config=config, **wrapper_values)


# TODO: Migrate the session verifier runner to typed configuration and remove this adapter.
def _runner_namespace(
    *, config: AllConfig, assistant_text_threshold: float, enable_spec: bool, session_verify_spec_trajectory: bool
) -> argparse.Namespace:
    generate = config.custom_generate_function_path
    agent = config.custom_agent_function_path
    assert generate is not None and generate.config is not None
    assert agent is not None
    generate_values = dict(generate.config)
    return argparse.Namespace(
        hf_checkpoint=config.hf_checkpoint,
        prompt_data=config.prompt_data,
        input_key=config.input_key,
        rollout_batch_size=config.rollout_batch_size,
        n_samples_per_prompt=config.n_samples_per_prompt,
        rollout_max_response_len=config.rollout_max_response_len,
        rollout_temperature=config.rollout_temperature,
        global_batch_size=config.global_batch_size,
        custom_generate_function_path=generate.path,
        custom_agent_function_path=agent.path,
        session_verify_cycles=generate_values["session_verify_cycles"],
        session_verify_spec_trajectory=session_verify_spec_trajectory,
        session_verify_gate=True,
        tito_model=config.tito_model,
        session_message_matcher=config.session_message_matcher,
        rollout_num_gpus_per_engine=config.rollout_num_gpus_per_engine,
        rollout_num_gpus=config.rollout_num_gpus,
        rm_type=config.rm_type,
        actor_num_nodes=config.actor_num_nodes,
        actor_num_gpus_per_node=config.actor_num_gpus_per_node,
        train_backend=config.train_backend,
        use_session_server=config.use_session_server,
        debug_rollout_only=config.debug_rollout_only,
        ci_test=config.ci_test,
        ci_tito_special_token_count_threshold=config.ci_tito_special_token_count_threshold,
        enable_sample_ownership_checker=config.enable_sample_ownership_checker,
        colocate=config.colocate,
        assistant_text_threshold=assistant_text_threshold,
        enable_spec=enable_spec,
        **{
            f"sglang_{name}": config.sglang.common_value(name)
            for name in (
                "reasoning_parser",
                "tool_call_parser",
                "context_length",
                "kv_cache_dtype",
                "mamba_full_memory_ratio",
                "cuda_graph_max_bs_decode",
                "cuda_graph_backend_prefill",
                "ep_size",
            )
        },
    )


def _print_rollout_table(tito_model: str, *, spec_trajectory: bool) -> None:
    fixed = fixed_template(tito_model)
    print("Rollout steps:")
    for step, kind in enumerate(session_verify_kinds(spec_trajectory)):
        print(f"  {step}. {kind}")
    print()
    print(f"Roles after a tool call:     {sorted(fixed.roles_after_tool_call)}")
    print(f"Roles after no tool call:    {sorted(fixed.roles_after_no_tool_call)}")
    if spec_trajectory:
        print("Spec probes (200 if the spec accepts the role there, else 400):")
        for step in spec_probe_steps(fixed):
            print(f"  - {step}")
    print()


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    sys.argv[1:] = _with_session_verify_defaults(sys.argv[1:])
    args = _parse_session_verify_args()

    # Resolve the family-owned contract before any GPU work starts so an
    # unknown family fails immediately.
    allowed_roles = sorted(fixed_template(args.tito_model).allowed_append_roles)

    print(f"Model:                  {args.hf_checkpoint}")
    print(f"TITO model family:      {args.tito_model}")
    print(f"Template append roles:  {allowed_roles}")
    print(f"sglang reasoning parser:{args.sglang_reasoning_parser}")
    print(f"sglang tool-call parser:{args.sglang_tool_call_parser or '(none)'}")
    print(f"Rollout GPUs per engine:{args.rollout_num_gpus_per_engine}")
    print(f"sglang expert-parallel: {args.sglang_ep_size}")
    print(f"Actor nodes:            {args.actor_num_nodes}")
    print(f"Actor GPUs per node:    {args.actor_num_gpus_per_node}")
    print(f"Samples per prompt:     {args.n_samples_per_prompt}")
    print(f"Normal rollout cycles:  {args.session_verify_cycles}")
    print(f"Spec rollout:           {args.session_verify_spec_trajectory}")
    print()

    _print_rollout_table(args.tito_model, spec_trajectory=args.session_verify_spec_trajectory)

    try:
        run_session_verify(args=args)
    except Exception as e:
        print()
        print(f"Verdict: FAIL -- {type(e).__name__}: {e}", file=sys.stderr)
        return 1

    print()
    print(
        "Verdict: PASS -- TITO incremental tokenization matched standard re-tokenize "
        "and the session server accepted and refused continuations as the family's spec decides."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
