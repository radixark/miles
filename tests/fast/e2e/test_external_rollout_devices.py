import importlib.util
from pathlib import Path

import pytest

_E2E_PATH = Path(__file__).parents[3] / "tests/e2e/short/test_qwen2.5_0.5B_external_rollout.py"
_spec = importlib.util.spec_from_file_location("external_rollout_e2e", _E2E_PATH)
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)
compute_train_and_engine_devices = _module.compute_train_and_engine_devices


class TestComputeTrainAndEngineDevices:
    @pytest.mark.parametrize("visible_devices", ["0,1,2,3", "4,5,6,7"])
    def test_both_roles_stay_inside_the_partition_the_runner_offered(self, visible_devices):
        """A job container sees every physical gpu, so hardcoded device numbers would step onto the
        partition of whatever other runner shares the host."""
        train_devices, engine_devices = compute_train_and_engine_devices(visible_devices)

        assert not set(train_devices) & set(engine_devices)
        assert train_devices + engine_devices == visible_devices.split(",")

    def test_the_engines_take_the_devices_ray_will_not_hand_the_trainer(self):
        """Ray labels its gpus from the front of the visible set, so the trainer gets the first ones."""
        train_devices, engine_devices = compute_train_and_engine_devices("4,5,6,7")

        assert (train_devices, engine_devices) == (["4", "5"], ["6", "7"])

    def test_gpu_uuids_are_passed_through_unchanged(self):
        """CUDA_VISIBLE_DEVICES may name uuids or mig instances, which are not indices."""
        uuids = "GPU-aaaa,GPU-bbbb,GPU-cccc,GPU-dddd"

        train_devices, engine_devices = compute_train_and_engine_devices(uuids)

        assert (train_devices, engine_devices) == (["GPU-aaaa", "GPU-bbbb"], ["GPU-cccc", "GPU-dddd"])

    @pytest.mark.parametrize("visible_devices", [None, ""])
    def test_an_unpartitioned_host_falls_back_to_the_first_devices(self, visible_devices):
        """Running the script by hand on a whole node must keep working."""
        assert compute_train_and_engine_devices(visible_devices) == (["0", "1"], ["2", "3"])

    @pytest.mark.parametrize("visible_devices", ["0,1", "0,1,2,3,4,5,6,7"])
    def test_a_partition_of_the_wrong_size_fails_loudly(self, visible_devices):
        """Silently reusing a device or leaving one idle would look like a flake later."""
        with pytest.raises(AssertionError, match="the runner offered"):
            compute_train_and_engine_devices(visible_devices)
