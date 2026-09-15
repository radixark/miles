from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Annotated, Any

import typer

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from miles.utils.external_utils.command_utils.helm_backend.launcher.values.helm_values_types import (  # noqa: E402
    InfraValues,
    MilesWorkbenchChartValues,
    ValuesModel,
)

DRAFT_07 = "http://json-schema.org/draft-07/schema#"

_SHARED_INFRA_PATH = Path("charts/shared-infra.schema.json")
_SHARED_INFRA_DESCRIPTION = (
    "Cluster-shaped values that every Miles chart accepts under the same paths, so one user-maintained "
    "values.yaml drives miles-workbench and miles-run alike. Helm does not resolve cross-file $ref, so each "
    "chart inlines these definitions verbatim into its own values.schema.json; "
    "tools/kubernetes/generate_chart_schema.py writes them from the same Python types."
)

# Helm hands every chart these two sections whether or not the chart declares them, so a root that
# forbids unknown keys has to name them itself.
_HELM_SECTIONS: dict[str, Any] = {
    "global": {
        "type": "object",
        "description": (
            "Helm injects the umbrella chart's global values into every subchart, so a strict root has to "
            "accept them."
        ),
    },
    "miles-common": {
        "type": "object",
        "additionalProperties": False,
        "properties": {"global": {"type": "object"}},
        "description": "Helm reserves a values section per dependency; the library chart itself consumes none.",
    },
}

app = typer.Typer(add_completion=False)


@app.command()
def main(
    check: Annotated[bool, typer.Option(help="Fail instead of writing when a file is out of date")] = False,
) -> None:
    stale = [path for path, content in generated_schemas().items() if not _write(path, content, check=check)]
    if stale:
        raise typer.Exit(code=1)


def generated_schemas() -> dict[Path, str]:
    return {
        _SHARED_INFRA_PATH: _rendered(_shared_infra_schema()),
        Path("charts/miles-workbench/values.schema.json"): _rendered(
            _chart_schema(MilesWorkbenchChartValues, title="miles-workbench values", required=["infra"])
        ),
    }


def _write(path: Path, content: str, *, check: bool) -> bool:
    absolute = REPO_ROOT / path
    if absolute.exists() and absolute.read_text() == content:
        return True
    if check:
        print(f"{path} is out of date; rerun tools/kubernetes/generate_chart_schema.py", flush=True)
        return False
    absolute.write_text(content)
    print(f"Wrote {path}", flush=True)
    return True


def _shared_infra_schema() -> dict[str, Any]:
    schema = _draft_07(InfraValues)
    definitions = schema.pop("definitions", None)
    return {
        "$schema": DRAFT_07,
        "title": "Miles shared infra values",
        "description": _SHARED_INFRA_DESCRIPTION,
        "type": "object",
        "properties": {"infra": schema},
        **({"definitions": definitions} if definitions else {}),
    }


def _chart_schema(values: type[ValuesModel], *, title: str, required: list[str]) -> dict[str, Any]:
    schema = _draft_07(values)
    return {
        "$schema": DRAFT_07,
        "title": title,
        "type": "object",
        "additionalProperties": False,
        "required": required,
        "properties": {**_HELM_SECTIONS, **schema["properties"]},
        **({"definitions": schema["definitions"]} if "definitions" in schema else {}),
    }


def _draft_07(model: type[ValuesModel]) -> dict[str, Any]:
    schema = model.model_json_schema(by_alias=True, ref_template="#/definitions/{model}")
    definitions = schema.pop("$defs", None)
    simplified = _simplify(schema)
    if definitions:
        simplified["definitions"] = {name: _simplify(entry) for name, entry in definitions.items()}
    return simplified


_DROPPED_KEYS = frozenset({"title", "default"})


def _simplify(schema: Any) -> Any:
    if isinstance(schema, list):
        return [_simplify(entry) for entry in schema]
    if not isinstance(schema, dict):
        return schema

    schema = _without_null_branch(schema)
    return {key: _simplify(value) for key, value in schema.items() if key not in _DROPPED_KEYS}


def _without_null_branch(schema: dict[str, Any]) -> dict[str, Any]:
    branches = schema.get("anyOf")
    if not isinstance(branches, list):
        return schema

    kept = [branch for branch in branches if branch != {"type": "null"}]
    if len(kept) != 1 or len(kept) == len(branches):
        return schema
    return {**{key: value for key, value in schema.items() if key != "anyOf"}, **kept[0]}


def _rendered(schema: dict[str, Any]) -> str:
    return json.dumps(schema, indent=2) + "\n"


if __name__ == "__main__":
    app()
