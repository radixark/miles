#!/usr/bin/env python3
import argparse
import shlex
import textwrap
from pathlib import Path


ARRAY_TO_VAR = {
    "CKPT_ARGS": "ckpt_args",
    "ROLLOUT_ARGS": "rollout_args",
    "EVAL_ARGS": "eval_args",
    "PERF_ARGS": "perf_args",
    "GRPO_ARGS": "grpo_args",
    "OPTIMIZER_ARGS": "optimizer_args",
    "WANDB_ARGS": "wandb_args",
    "TB_ARGS": "tb_args",
    "SGLANG_ARGS": "sglang_args",
    "MISC_ARGS": "misc_args",
}

TRAIN_ORDER = [
    "ckpt_args",
    "rollout_args",
    "optimizer_args",
    "grpo_args",
    "wandb_args",
    "tb_args",
    "perf_args",
    "eval_args",
    "sglang_args",
    "misc_args",
]


def _is_array_start(line: str) -> str | None:
    stripped = line.strip()
    for key in ARRAY_TO_VAR:
        if stripped == f"{key}=(":
            return key
    return None


def _normalize_item(line: str) -> str | None:
    body = line.split("#", 1)[0].strip()
    if not body:
        return None
    if body in {")", "("}:
        return None
    return body


def _split_tokens(item: str) -> list[str]:
    try:
        return shlex.split(item, posix=True)
    except ValueError:
        return [item]


def parse_bash_arrays(script_text: str) -> dict[str, list[str]]:
    lines = script_text.splitlines()
    arrays: dict[str, list[str]] = {k: [] for k in ARRAY_TO_VAR}
    i = 0
    while i < len(lines):
        key = _is_array_start(lines[i])
        if key is None:
            i += 1
            continue

        i += 1
        while i < len(lines):
            raw = lines[i]
            if raw.strip() == ")":
                break
            norm = _normalize_item(raw)
            if norm is not None:
                arrays[key].append(norm)
            i += 1
        i += 1
    return arrays


def tokens_to_python_lines(items: list[str]) -> list[str]:
    out: list[str] = []
    for item in items:
        tokens = _split_tokens(item)
        if not tokens:
            continue
        rendered = " ".join(tokens)
        out.append(f'"--{rendered[2:]} "' if rendered.startswith("--") else f'"{rendered} "')
    return out


def render_section(var_name: str, items: list[str]) -> str:
    py_lines = tokens_to_python_lines(items)
    if not py_lines:
        return f'{var_name} = ""'
    body = "\n".join(f"        {line}" for line in py_lines)
    return f"{var_name} = (\\n{body}\\n    )"


def render_template(arrays: dict[str, list[str]]) -> str:
    rendered_sections = []
    for bash_name, py_name in ARRAY_TO_VAR.items():
        rendered_sections.append(render_section(py_name, arrays.get(bash_name, [])))
    sections = "\\n\\n    ".join(rendered_sections)

    train_join = "\\n".join(f'        f"{{{name}}} "' for name in TRAIN_ORDER)
    return textwrap.dedent(
        f"""\
        import os

        import miles.utils.external_utils.command_utils as U

        NUM_GPUS = 8
        MODEL_TYPE = "qwen3.5"


        def execute():
            {sections}

            train_args = (
        {train_join}
            )

            U.execute_train(
                train_args=train_args,
                num_gpus_per_node=NUM_GPUS,
                megatron_model_type=MODEL_TYPE,
            )


        if __name__ == "__main__":
            for proxy_var in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
                os.environ.pop(proxy_var, None)
            execute()
        """
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert run-*.sh args blocks into test-style Python args blocks."
    )
    parser.add_argument("input_sh", type=Path, help="Path to bash run script")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output python path; default prints to stdout",
    )
    args = parser.parse_args()

    script_text = args.input_sh.read_text(encoding="utf-8")
    arrays = parse_bash_arrays(script_text)
    py_code = render_template(arrays)

    if args.output:
        args.output.write_text(py_code, encoding="utf-8")
        print(f"Wrote: {args.output}")
    else:
        print(py_code)


if __name__ == "__main__":
    main()
