import importlib.util
from pathlib import Path

import pytest
from tests.ci.ci_register import CIRegistry, HWBackend, register_cpu_ci

register_cpu_ci(est_time=1, suite="stage-a-cpu", labels=[])

ROOT = Path(__file__).parents[3]
SCRIPT = ".claude/skills/pr-ci-targeted/affected_tests.py"
spec = importlib.util.spec_from_file_location("affected_tests", ROOT / SCRIPT)
TRACER = importlib.util.module_from_spec(spec)
spec.loader.exec_module(TRACER)


def _gpu(path, labels=("precision",), disabled=None):
    return CIRegistry(
        backend=HWBackend.CUDA,
        filename=path,
        est_time=60,
        suite="stage-b-2-gpu-h200",
        labels=list(labels),
        hardware=["hopper"],
        disabled=disabled,
    )


def _cpu(path, labels=()):
    return CIRegistry(backend=HWBackend.CPU, filename=path, est_time=1, suite="stage-b-cpu", labels=list(labels))


def _nightly(path, labels):
    return CIRegistry(
        backend=HWBackend.CUDA,
        filename=path,
        est_time=60,
        suite="stage-c-2-gpu-h200",
        labels=list(labels),
        hardware=["hopper"],
        nightly=True,
    )


@pytest.fixture
def repo(tmp_path, monkeypatch):
    """Write files into a scratch repo root and trace a change there."""
    monkeypatch.chdir(tmp_path)

    def write(files):
        for path, text in files.items():
            (tmp_path / path).parent.mkdir(parents=True, exist_ok=True)
            (tmp_path / path).write_text(text)

    return write


def _trace(changed, registrations, *, depth=3, hub_limit=40):
    registrations = {r.filename: [r] for r in registrations}
    return TRACER.trace(changed, registrations, depth=depth, hub_limit=hub_limit)


def test_from_package_import_module_reaches_the_test(repo):
    repo(
        {
            "miles/__init__.py": "",
            "miles/kernels/__init__.py": "",
            "miles/kernels/embedding/__init__.py": "",
            "miles/kernels/embedding/ple_gate_conv.py": "",
            "tests/fast-gpu/test_ple.py": "from miles.kernels.embedding import ple_gate_conv as ple\n",
        }
    )

    result = _trace(["miles/kernels/embedding/ple_gate_conv.py"], [_gpu("tests/fast-gpu/test_ple.py")])

    assert result.reached == {"tests/fast-gpu/test_ple.py": "miles/kernels/embedding/ple_gate_conv.py (hop 0)"}


@pytest.mark.parametrize(
    "statement", ["from .model import build", "from . import model", "from ..megatron_utils.model import build"]
)
def test_relative_import_is_resolved_against_the_package(repo, statement):
    repo(
        {
            "miles/__init__.py": "",
            "miles/backends/__init__.py": "",
            "miles/backends/megatron_utils/__init__.py": "",
            "miles/backends/megatron_utils/model.py": "",
            "miles/backends/megatron_utils/actor.py": f"{statement}\n",
            "tests/fast/test_actor.py": "import miles.backends.megatron_utils.actor\n",
        }
    )

    result = _trace(["miles/backends/megatron_utils/model.py"], [_cpu("tests/fast/test_actor.py", ["rollout"])])

    assert result.reached == {"tests/fast/test_actor.py": "miles/backends/megatron_utils/actor.py (hop 1)"}


def test_bare_import_of_a_sibling_on_sys_path_reaches_the_test(repo):
    repo(
        {
            "tests/fast-gpu/linear_attn_reference.py": "",
            "tests/fast-gpu/test_linear_attn_layer.py": "import sys\nimport linear_attn_reference\n",
            "tests/fast-gpu/test_other.py": "import torch\n",
        }
    )

    result = _trace(
        ["tests/fast-gpu/linear_attn_reference.py"],
        [_gpu("tests/fast-gpu/test_linear_attn_layer.py"), _gpu("tests/fast-gpu/test_other.py")],
    )

    assert set(result.reached) == {"tests/fast-gpu/test_linear_attn_layer.py"}


def test_dotted_module_and_file_paths_in_string_literals_are_edges(repo):
    repo(
        {
            "miles/__init__.py": "",
            "miles/custom_loss.py": "def loss(): ...\n",
            "tools/convert.py": "",
            "tests/e2e/test_loss.py": 'ARGS = "--custom-loss-function-path miles.custom_loss.loss"\n',
            "tests/e2e/test_convert.py": 'CMD = f"python tools/convert.py --out {1}"\n',
        }
    )
    registrations = [_gpu("tests/e2e/test_loss.py"), _gpu("tests/e2e/test_convert.py")]

    assert set(_trace(["miles/custom_loss.py"], registrations).reached) == {"tests/e2e/test_loss.py"}
    assert set(_trace(["tools/convert.py"], registrations).reached) == {"tests/e2e/test_convert.py"}


def test_helper_loaded_by_file_name_or_path_suffix_reaches_its_tests(repo):
    load = 'spec_from_file_location("ref", pathlib.Path(__file__).with_name("dsa_reference.py"))\n'
    repo(
        {
            "tests/fast-gpu/kernels/dsa/dsa_reference.py": "",
            "tests/fast-gpu/kernels/dsa/test_dsa.py": load,
            "tests/fast-gpu/kernels/other/test_other.py": load,
            "tests/e2e/test_bench.py": 'REF = ROOT / "fast-gpu/kernels/dsa/dsa_reference.py"\n',
        }
    )
    registrations = [
        _gpu("tests/fast-gpu/kernels/dsa/test_dsa.py"),
        _gpu("tests/fast-gpu/kernels/other/test_other.py"),
        _gpu("tests/e2e/test_bench.py"),
    ]

    result = _trace(["tests/fast-gpu/kernels/dsa/dsa_reference.py"], registrations)

    assert set(result.reached) == {"tests/fast-gpu/kernels/dsa/test_dsa.py", "tests/e2e/test_bench.py"}


def test_model_name_matches_whole_names_and_variants_but_not_longer_names(repo):
    repo(
        {
            "scripts/models/deepseek-v3.py": "",
            "tests/e2e/test_v3.py": 'MODEL_TYPE = "deepseek-v3"\n',
            "tests/e2e/test_v3_5layer.py": 'MODEL_TYPE = "deepseek-v3-5layer"\n',
            "tests/e2e/test_v32.py": 'MODEL_TYPE = "deepseek-v32-5layer"\n',
        }
    )
    registrations = [_gpu(f"tests/e2e/test_{name}.py") for name in ("v3", "v3_5layer", "v32")]

    result = _trace(["scripts/models/deepseek-v3.py"], registrations)

    assert set(result.reached) == {"tests/e2e/test_v3.py", "tests/e2e/test_v3_5layer.py"}


def test_conftest_reaches_every_test_in_its_directory_subtree(repo):
    repo(
        {
            "tests/fast/rollout/conftest.py": "import pytest\n",
            "tests/fast/rollout/test_a.py": "",
            "tests/fast/rollout/sub/test_b.py": "",
            "tests/fast/router/test_c.py": "",
        }
    )
    registrations = [
        _cpu("tests/fast/rollout/test_a.py", ["rollout"]),
        _cpu("tests/fast/rollout/sub/test_b.py"),
        _cpu("tests/fast/router/test_c.py"),
    ]

    result = _trace(["tests/fast/rollout/conftest.py"], registrations)

    assert set(result.reached) == {"tests/fast/rollout/test_a.py", "tests/fast/rollout/sub/test_b.py"}


def test_a_file_reaching_too_many_gpu_tests_is_a_hub_and_is_not_followed(repo):
    tests = {f"tests/fast-gpu/test_{i}.py": "import miles.arguments\n" for i in range(3)}
    repo({"miles/__init__.py": "", "miles/arguments.py": "", **tests})

    result = _trace(["miles/arguments.py"], [_gpu(path) for path in tests], hub_limit=2)

    assert result.hubs == {"miles/arguments.py": TRACER.Hub(gpu_tests=3, reached_from="changed")}
    assert result.reached == {}


def test_a_hub_met_downstream_records_where_the_trace_reached_it(repo):
    tests = {f"tests/fast-gpu/test_{i}.py": "import miles.entry\n" for i in range(3)}
    repo(
        {
            "miles/__init__.py": "",
            "miles/builder.py": "",
            "miles/launcher.py": "import miles.builder\n",
            "miles/entry.py": "import miles.launcher\n",
            **tests,
        }
    )

    result = _trace(["miles/builder.py"], [_gpu(path) for path in tests], depth=1, hub_limit=2)

    assert result.hubs == {"miles/launcher.py": TRACER.Hub(gpu_tests=3, reached_from="hop 1 from miles/builder.py")}
    assert result.reached == {}


def test_the_depth_bound_reports_what_it_left_unfollowed(repo):
    chain = ["miles/a.py", "miles/b.py", "miles/c.py", "miles/d.py", "miles/e.py"]
    files = {"miles/__init__.py": "", chain[0]: ""}
    for importer, imported in zip(chain[1:], chain, strict=False):
        files[importer] = f"import {imported[:-3].replace('/', '.')}\n"
    files["tests/fast-gpu/test_e.py"] = "import miles.e\n"
    repo(files)
    registrations = [_gpu("tests/fast-gpu/test_e.py")]

    shallow = _trace(["miles/a.py"], registrations, depth=2)
    deep = _trace(["miles/a.py"], registrations, depth=5)

    assert shallow.reached == {}
    assert shallow.unfollowed == ["miles/d.py"]
    assert deep.reached == {"tests/fast-gpu/test_e.py": "miles/e.py (hop 4)"}
    assert deep.unfollowed == []


def test_disabled_and_cpu_tests_do_not_make_a_hub(repo):
    tests = {f"tests/fast/test_{i}.py": "import miles.arguments\n" for i in range(3)}
    repo({"miles/__init__.py": "", "miles/arguments.py": "", **tests})
    registrations = [_cpu("tests/fast/test_0.py"), _cpu("tests/fast/test_1.py", ["rollout"])]
    registrations.append(_gpu("tests/fast/test_2.py", disabled="flaky"))

    result = _trace(["miles/arguments.py"], registrations, hub_limit=0)

    assert result.hubs == {}
    assert len(result.reached) == 3


def test_files_the_training_entry_point_loads_are_listed(repo):
    repo(
        {
            "train.py": "from miles.trainer import run\n",
            "miles/__init__.py": "",
            "miles/trainer.py": "from miles.loss import compute\n",
            "miles/loss.py": "",
            "miles/unused.py": "",
        }
    )

    result = _trace(["miles/loss.py", "miles/unused.py"], [])

    assert result.on_training_path == ["miles/loss.py"]
    assert result.unreferenced == ["miles/unused.py"]


def test_non_python_changes_outside_the_no_gpu_allowlist_need_label_ci_even_when_mentioned(repo):
    repo({"tests/fast/test_image.py": 'DOCKERFILE = "docker/Dockerfile"\n'})

    result = _trace(
        ["docker/Dockerfile", "docs/index.md", ".claude/skills/x/SKILL.md", "README.md"],
        [_cpu("tests/fast/test_image.py")],
    )

    assert result.outside_trace == ["docker/Dockerfile"]
    assert set(result.reached) == {"tests/fast/test_image.py"}


def test_python_outside_the_import_graph_needs_label_ci_but_is_still_traced(repo):
    repo(
        {
            "miles/__init__.py": "",
            "miles/core.py": "",
            "tests/fast/test_build.py": 'BUILD = "docker/build.py"\n',
            "tests/ci/test/test_skill.py": 'SCRIPT = ".claude/skills/s/tool.py"\n',
        }
    )
    registrations = [_cpu("tests/fast/test_build.py"), _cpu("tests/ci/test/test_skill.py")]

    result = _trace(
        ["docker/build.py", "tools/convert.py", ".claude/skills/s/tool.py", "miles/core.py"], registrations
    )

    assert result.outside_trace == ["docker/build.py", "tools/convert.py"]
    assert set(result.reached) == {"tests/fast/test_build.py", "tests/ci/test/test_skill.py"}


def test_a_deleted_module_still_reaches_its_importers(repo):
    repo({"miles/__init__.py": "", "tests/fast/test_uses_gone.py": "from miles.gone import thing\n"})

    result = _trace(["miles/gone.py"], [_cpu("tests/fast/test_uses_gone.py")])

    assert set(result.reached) == {"tests/fast/test_uses_gone.py"}


def test_rerun_targets_are_enabled_cuda_and_labeled_cpu_tests():
    registrations = {
        r.filename: [r]
        for r in (
            _gpu("tests/fast-gpu/test_gpu.py"),
            _gpu("tests/fast-gpu/test_off.py", disabled="flaky"),
            _cpu("tests/fast/test_labeled.py", ["rollout"]),
            _cpu("tests/fast/test_plain.py"),
        )
    }
    result = TRACER.Trace(
        reached=dict.fromkeys(registrations, "changed"),
        unfollowed=[],
        hubs={},
        outside_trace=[],
        on_training_path=[],
        unreferenced=[],
    )

    assert TRACER.rerun_targets(result, registrations) == ["tests/fast-gpu/test_gpu.py", "tests/fast/test_labeled.py"]


def test_label_coverage_counts_only_what_a_pr_label_selects():
    registrations = {
        r.filename: [r]
        for r in (
            _gpu("tests/e2e/test_regular.py", labels=["short"]),
            _gpu("tests/e2e/test_other.py", labels=["short"]),
            _nightly("tests/e2e/test_nightly.py", labels=["short"]),
            _gpu("tests/e2e/test_off.py", labels=["short"], disabled="flaky"),
        )
    }
    registrations["tests/e2e/test_mixed.py"] = [
        _gpu("tests/e2e/test_mixed.py", labels=["long"]),
        _nightly("tests/e2e/test_mixed.py", labels=["long"]),
    ]
    targets = ["tests/e2e/test_regular.py", "tests/e2e/test_nightly.py", "tests/e2e/test_mixed.py"]

    assert TRACER.label_coverage(targets, registrations) == [("short", 1, 2), ("long", 1, 1)]
    assert TRACER.nightly_only(targets, registrations) == ["tests/e2e/test_nightly.py"]


def test_unreadable_registrations_are_reported_instead_of_aborting(repo):
    repo(
        {
            "tests/e2e/test_ok.py": 'from tests.ci.ci_register import register_cpu_ci\nregister_cpu_ci(est_time=1, suite="stage-b-cpu")\n',
            "tests/e2e/test_unregistered.py": "def test_x(): ...\n",
            "tests/e2e/test_broken.py": "def test_x(:\n",
            "tests/fast/test_gpu_in_fast.py": (
                "from tests.ci.ci_register import register_cuda_ci\n"
                'register_cuda_ci(est_time=1, suite="stage-b-2-gpu-h200", labels=["precision"], hardware=["hopper"])\n'
            ),
        }
    )
    files = ["tests/e2e/test_ok.py", "tests/e2e/test_unregistered.py", "tests/e2e/test_broken.py"]

    registrations, unreadable = TRACER.load_registrations([*files, "tests/fast/test_gpu_in_fast.py"])

    assert set(registrations) == {"tests/e2e/test_ok.py"}
    assert set(unreadable) == {*files[1:], "tests/fast/test_gpu_in_fast.py"}
    assert unreadable["tests/e2e/test_unregistered.py"] == "no CI registration"
    assert unreadable["tests/fast/test_gpu_in_fast.py"].startswith("register_cuda_ci is forbidden in tests/fast/")
