"""Checkpoint checks that run before Megatron's load must accept an msc:// (e.g. S3-backed) checkpoint.

An msc:// profile backed by a local directory stands in for object storage."""

import json
import uuid
from types import SimpleNamespace

import pytest
import torch
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint import FileSystemWriter

multistorageclient = pytest.importorskip("multistorageclient")

from megatron.core.msc_utils import MultiStorageClientFeature  # noqa: E402

from tests.fast.fixtures.args_fixtures import (  # noqa: E402
    make_trainer_args,
    parse_fsdp_test_config,
    parse_megatron_test_config,
)
from tests.fast.fixtures.megatron_config_fixtures import write_megatron_config_trainers  # noqa: E402

from miles.backends.megatron_utils import checkpoint  # noqa: E402
from miles.backends.megatron_utils.checkpoint import (  # noqa: E402
    LINEAR_ATTN_LAYOUT_KEY,
    _is_megatron_checkpoint,
    assert_linear_attn_checkpoint_is_current,
)
from miles.backends.megatron_utils.megatron_config import (  # noqa: E402
    compute_trainer_checkpoint_dir,
    get_megatron_arg_parser,
    has_megatron_checkpoint,
)
from miles.backends.megatron_utils.model import _has_loadable_ckpt  # noqa: E402
from miles.utils.msc_utils import enable_megatron_msc, reject_msc_checkpoint_args  # noqa: E402
from miles_plugins.models.deepseek_v4.arguments import assert_checkpoint_is_current  # noqa: E402

LAYER = "decoder.layers.0.self_attention.linear_attn"


@pytest.fixture
def bucket(tmp_path, monkeypatch):
    """A local directory and the msc:// URL that reaches it.

    MSC caches a client per profile for the life of the process, so every test gets its own profile."""
    root = tmp_path / "bucket"
    root.mkdir()
    profile = f"ckpt-{uuid.uuid4().hex}"
    config = tmp_path / "msc_config.json"
    config.write_text(
        json.dumps(
            {"profiles": {profile: {"storage_provider": {"type": "file", "options": {"base_path": str(root)}}}}}
        )
    )
    monkeypatch.setenv("MSC_CONFIG", str(config))
    return root, f"msc://{profile}"


def _write_torch_dist(root, step, keys):
    iteration = root / "run" / ("release" if step == "release" else f"iter_{int(step):07d}")
    iteration.mkdir(parents=True)
    dcp.save({key: torch.zeros(2) for key in keys}, storage_writer=FileSystemWriter(iteration), no_dist=True)
    (root / "run" / "latest_checkpointed_iteration.txt").write_text(str(step))


class TestMscCheckpointDiscovery:
    def test_a_tracked_checkpoint_is_found(self, bucket):
        root, url = bucket
        _write_torch_dist(root, 3, ["w"])

        assert has_megatron_checkpoint(f"{url}/run")
        assert _is_megatron_checkpoint(f"{url}/run")
        assert _has_loadable_ckpt(f"{url}/run")

    def test_a_selected_step_without_a_tracker_is_found(self, bucket):
        root, url = bucket
        _write_torch_dist(root, 3, ["w"])
        (root / "run" / "latest_checkpointed_iteration.txt").unlink()

        assert has_megatron_checkpoint(f"{url}/run", ckpt_step=3)
        assert not has_megatron_checkpoint(f"{url}/run")

    def test_a_missing_or_empty_location_is_not_a_checkpoint(self, bucket):
        root, url = bucket
        (root / "empty").mkdir()

        assert not has_megatron_checkpoint(f"{url}/missing")
        assert not _is_megatron_checkpoint(f"{url}/empty")
        assert not _has_loadable_ckpt(f"{url}/empty")
        assert not _has_loadable_ckpt(f"{url}/missing")

    def test_a_storage_error_is_raised_instead_of_reported_as_an_empty_checkpoint(self, bucket, monkeypatch):
        """msc Path.is_dir() returns False on any error (403, missing credentials, timeouts); loading would then
        be skipped and the trainer would start from random weights."""
        root, url = bucket
        _write_torch_dist(root, 3, ["w"])

        def denied(self):
            raise PermissionError("403 Forbidden")

        monkeypatch.setattr(multistorageclient.Path, "exists", denied)

        with pytest.raises(PermissionError, match="403"):
            _has_loadable_ckpt(f"{url}/run")

    def test_a_per_trainer_checkpoint_dir_keeps_the_msc_scheme(self):
        assert compute_trainer_checkpoint_dir(base_dir="msc://p/run", trainer_id="a") == "msc://p/run/trainers/a"
        assert compute_trainer_checkpoint_dir(base_dir="/ckpt/run", trainer_id="a") == "/ckpt/run/trainers/a"


class TestMscCheckpointLayoutGuards:
    def test_a_linear_attention_checkpoint_without_the_layout_marker_is_rejected(self, bucket):
        root, url = bucket
        _write_torch_dist(root, 12, [f"{LAYER}.in_proj_qkv.weight", f"{LAYER}.conv1d.weight"])

        with pytest.raises(ValueError, match="Re-convert it from the HuggingFace checkpoint"):
            assert_linear_attn_checkpoint_is_current(f"{url}/run")

    def test_a_linear_attention_checkpoint_with_the_layout_marker_passes(self, bucket):
        root, url = bucket
        _write_torch_dist(root, 12, [f"{LAYER}.in_proj_qkv.weight", f"{LAYER}.{LINEAR_ATTN_LAYOUT_KEY}"])

        assert_linear_attn_checkpoint_is_current(f"{url}/run")

    def test_a_deepseek_v4_checkpoint_with_the_old_names_is_rejected(self, bucket):
        root, url = bucket
        _write_torch_dist(root, "release", ["decoder.layers.0.self_attention.wq_a.weight"])

        with pytest.raises(ValueError, match="Re-convert it from the HuggingFace checkpoint"):
            assert_checkpoint_is_current(f"{url}/run")


def _validated(extra):
    """The full launch parse, which runs miles_validate_args after resolving every trainer."""
    return parse_megatron_test_config(*extra)


class TestMscLaunchChecks:
    def test_an_msc_ref_load_without_enable_msc_is_rejected(self, bucket):
        """Without --enable-msc Megatron cannot see the checkpoint and would start from random weights."""
        root, url = bucket
        _write_torch_dist(root, 3, ["w"])

        with pytest.raises(ValueError, match="--enable-msc"):
            _validated(["--ref-load", f"{url}/run"])

    def test_an_msc_ref_load_with_enable_msc_passes_the_kl_reference_check(self, bucket):
        root, url = bucket
        _write_torch_dist(root, 3, ["w"])

        _validated(["--enable-msc", "--ref-load", f"{url}/run", "--use-kl-loss"])

    def test_a_missing_msc_ref_load_still_fails_the_kl_reference_check(self, bucket):
        _, url = bucket

        with pytest.raises(FileNotFoundError, match="does not exist"):
            _validated(["--enable-msc", "--ref-load", f"{url}/missing", "--use-kl-loss"])

    def test_an_msc_load_is_rejected_because_resume_restores_rollout_state_locally(self, bucket):
        root, url = bucket
        _write_torch_dist(root, 3, ["w"])

        with pytest.raises(ValueError, match="use --ref-load"):
            _validated(["--enable-msc", "--load", f"{url}/run"])


class TestEnableMegatronMsc:
    @pytest.fixture(autouse=True)
    def _restore_feature(self):
        was_enabled = MultiStorageClientFeature.is_enabled()
        yield
        if was_enabled:
            MultiStorageClientFeature.enable()
        else:
            MultiStorageClientFeature.disable()

    def test_a_trainer_worker_enables_msc_when_the_driver_parsed_enable_msc(self):
        """Megatron enables MSC only inside parse_args, which trainer workers never run."""
        MultiStorageClientFeature.disable()

        enable_megatron_msc(SimpleNamespace(enable_msc=True))

        assert MultiStorageClientFeature.is_enabled()

    def test_a_trainer_worker_leaves_msc_off_without_enable_msc(self):
        MultiStorageClientFeature.disable()

        enable_megatron_msc(SimpleNamespace(enable_msc=False))

        assert not MultiStorageClientFeature.is_enabled()

    def test_a_worker_without_the_package_fails_instead_of_reading_locally(self, monkeypatch):
        MultiStorageClientFeature.disable()
        monkeypatch.setattr(MultiStorageClientFeature, "is_enabled", lambda *a: False)

        with pytest.raises(RuntimeError, match="multi-storage-client is not available"):
            enable_megatron_msc(SimpleNamespace(enable_msc=True))


class TestMscTrainerChecks:
    """The launch checks cover every resolved trainer, not only the top-level --load / --ref-load."""

    def test_a_policy_ref_load_from_megatron_config_needs_enable_msc(self, bucket, tmp_path):
        _, url = bucket
        config = write_megatron_config_trainers(
            tmp_path, [{"model_id": "a"}, {"model_id": "b", "overrides": {"ref_load": f"{url}/b"}}]
        )

        with pytest.raises(ValueError, match="trainer 'b-actor'.*--enable-msc"):
            _validated(["--megatron-config", config])
        _validated(["--megatron-config", config, "--enable-msc"])

    def test_an_msc_critic_load_is_rejected(self, bucket):
        """--critic-load becomes the critic's load and ref_load."""
        _, url = bucket

        with pytest.raises(ValueError, match="trainer 'critic'.*not supported"):
            _validated(["--enable-msc", "--advantage-estimator", "ppo", "--critic-load", f"{url}/critic"])

    def test_an_empty_ref_load_fails_the_kl_reference_check(self):
        """storage_path("") is the working directory, which exists."""
        with pytest.raises(FileNotFoundError, match="does not exist"):
            _validated(["--ref-load", "", "--use-kl-loss"])

    def test_an_fsdp_launch_rejects_an_msc_ref_load(self, bucket):
        """FSDP reads --ref-load from the local filesystem, so it stops at startup instead of in the worker."""
        _, url = bucket

        with pytest.raises(ValueError, match="only by --train-backend megatron"):
            parse_fsdp_test_config("--ref-load", f"{url}/ref", "--use-kl-loss")

    @pytest.mark.parametrize("backend", ["fsdp", "torchtitan"])
    def test_a_non_megatron_backend_rejects_an_msc_ref_load(self, bucket, backend):
        _, url = bucket
        args = get_megatron_arg_parser().parse_args(["--num-rollout", "1", "--rollout-batch-size", "64"])
        args.train_backend, args.ref_load = backend, f"{url}/ref"

        with pytest.raises(ValueError, match="only by --train-backend megatron"):
            reject_msc_checkpoint_args(args)


class TestMscLoadCheckpoint:
    @pytest.fixture(autouse=True)
    def _restore_feature(self):
        was_enabled = MultiStorageClientFeature.is_enabled()
        yield
        if was_enabled:
            MultiStorageClientFeature.enable()
        else:
            MultiStorageClientFeature.disable()

    @pytest.fixture
    def megatron_loads(self, monkeypatch):
        loads = []
        monkeypatch.setattr(checkpoint, "_load_checkpoint_megatron", lambda **kwargs: loads.append(kwargs) or (3, 0))
        return loads

    def _load(self, load):
        args = make_trainer_args(load=load, finetune=False, ckpt_step=None, lora_rank=0, spec=None)
        return checkpoint.load_checkpoint([], None, None, None, False, args=args)

    def test_an_msc_checkpoint_is_handed_to_megatron_when_msc_is_enabled(self, bucket, megatron_loads):
        root, url = bucket
        _write_torch_dist(root, 3, ["w"])
        MultiStorageClientFeature.enable()

        assert self._load(f"{url}/run").iteration == 3
        assert len(megatron_loads) == 1

    def test_an_msc_checkpoint_is_refused_when_megatron_cannot_read_it(self, bucket, megatron_loads):
        """With MSC off Megatron finds no tracker and silently starts from random weights."""
        root, url = bucket
        _write_torch_dist(root, 3, ["w"])
        MultiStorageClientFeature.disable()

        with pytest.raises(RuntimeError, match="pass --enable-msc"):
            self._load(f"{url}/run")
        assert not megatron_loads

    def test_a_missing_msc_checkpoint_fails_the_load(self, bucket, megatron_loads):
        _, url = bucket
        MultiStorageClientFeature.enable()

        with pytest.raises(AssertionError, match="does not exist or is an empty directory"):
            self._load(f"{url}/missing")
        assert not megatron_loads
