"""Render the archived workload for administrator-owned paths; never submit it."""

import json
import shlex
from pathlib import Path
from string import Template

from tap import Tap


class Args(Tap):
    output: str
    model: str
    reference: str
    tasks: str
    harbor: str
    megatron: str
    dependencies: str
    templates: str
    key_file: str
    api_url: str
    sandbox_url: str
    python: str = "/opt/sglang/bin/python3"
    submission_id: str = "e2b-rollout-repro"
    resume_roots: str = ""


def render(value: object, values: dict[str, str]) -> object:
    if isinstance(value, str):
        return Template(value).substitute(values)
    if isinstance(value, list):
        return [render(item, values) for item in value]
    if isinstance(value, dict):
        return {key: render(item, values) for key, item in value.items()}
    return value


def main() -> None:
    args = Args().parse_args()
    here = Path(__file__).resolve().parent
    repo = here.parents[2]
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    values = {key: str(value) for key, value in vars(args).items() if not key.startswith("_")}
    values.update(repo=str(repo), code=str(here), output=str(output))
    for key in ("model", "reference", "tasks", "harbor", "megatron", "dependencies", "templates", "key_file"):
        values[key] = str(Path(values[key]).resolve())
        if not Path(values[key]).exists():
            raise FileNotFoundError(f"{key}: {values[key]}")
    recipe = render(json.loads((here / "recipe.json").read_text()), values)
    template_rows = json.loads(Path(args.templates).read_text())["results"]
    template_names = {row["task"] for row in template_rows}
    rows = []
    task_names = json.loads((here / "train-task-names.json").read_text())
    for name in task_names:
        if name not in template_names:
            raise ValueError(f"Missing prebuilt E2B template: {name}")
        task = Path(args.tasks) / name
        rows.append({
            "prompt": [{"role": "user", "content": (task / "instruction.md").read_text()}],
            "label": name,
            "metadata": {"instance_id": name, "agent_name": "terminus-2",
                         "max_seq_len": 65536, "split": "train"},
        })
    (output / "train.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    env = recipe["environment"]
    env["FROZEN_RESUME_ROOTS"] = args.resume_roots
    env["HARBOR_ENV_KWARGS"] = json.dumps({"template_manifest": values["templates"]})
    (output / "runtime-env.json").write_text(json.dumps({"env_vars": env}, indent=2))
    command = [args.python, str(repo / "train.py"), *recipe["arguments"]]
    (output / "command.txt").write_text(shlex.join(command) + "\n")
    request = {"submission_id": args.submission_id, "entrypoint": shlex.join(command),
               "runtime_env": {"env_vars": env}}
    (output / "ray-job-request.json").write_text(json.dumps(request, indent=2))
    print(f"Prepared {len(rows)} task rows; no job submitted.")
    print(f"Review {output / 'command.txt'} and {output / 'runtime-env.json'} before submitting.")


if __name__ == "__main__":
    main()
