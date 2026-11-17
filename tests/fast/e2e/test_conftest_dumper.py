import ast
from pathlib import Path
from textwrap import dedent
from types import SimpleNamespace

import pytest
import sglang
import yaml
from sglang.srt.debug_utils.source_patcher.source_editor import apply_edits
from sglang.srt.debug_utils.source_patcher.types import PatchConfig, PatchSpec
from sglang.srt.layers.layer_boundary.construction import StagePlan
from tests.e2e.conftest_dumper import SGLANG_SOURCE_PATCHER_CONFIG_YAML


class TestDumperSourceInjection:
    @pytest.mark.parametrize(
        "patch", PatchConfig.model_validate(yaml.safe_load(SGLANG_SOURCE_PATCHER_CONFIG_YAML)).patches
    )
    def test_injections_match_installed_model_and_produce_valid_python(self, patch: PatchSpec) -> None:
        """Dumper injections remain applicable to the installed inference model."""
        module, class_name, method_name = patch.target.rsplit(".", 2)
        path = Path(sglang.__file__).parent.joinpath(*module.split(".")[1:]).with_suffix(".py")
        source = path.read_text()
        model = next(
            node for node in ast.parse(source).body if isinstance(node, ast.ClassDef) and node.name == class_name
        )
        method = next(node for node in model.body if isinstance(node, ast.FunctionDef) and node.name == method_name)
        method_source = ast.get_source_segment(source, method)
        assert method_source is not None

        patched = apply_edits(source=method_source, edits=patch.edits)

        compile(dedent(patched), str(path), "exec")

    @pytest.mark.parametrize(("group", "expected_dims"), [(None, "t h # tp:replicated"), ("tp", "t h[tp:partial]")])
    def test_mlp_injection_preserves_the_produced_reduction_contract(
        self, monkeypatch: pytest.MonkeyPatch, group: str | None, expected_dims: str
    ) -> None:
        """The injected dump reads the installed plan API and retains pending reductions."""
        plan = StagePlan.__new__(StagePlan)
        monkeypatch.setattr(plan, "path_for", lambda batch: SimpleNamespace(output=SimpleNamespace(group=group)))
        config = PatchConfig.model_validate(yaml.safe_load(SGLANG_SOURCE_PATCHER_CONFIG_YAML))
        patch = next(p for p in config.patches if p.target.endswith("Qwen3MoeDecoderLayer.forward"))
        edit = next(e for e in patch.edits if e.match == "hidden_states = self.mlp(hidden_states, forward_batch)")
        assert edit.append is not None
        observed = {}
        namespace = dict(
            self=SimpleNamespace(ffn_boundary=SimpleNamespace(plan=plan)),
            forward_batch=object(),
            hidden_states=object(),
            dumper=SimpleNamespace(dump=lambda name, tensor, **kwargs: observed.update(kwargs)),
        )

        exec(compile(edit.append, "<dumper mlp injection>", "exec"), namespace)

        assert observed == {"dims": expected_dims}
