import pytest
from pydantic import ValidationError

from miles.utils.external_utils.command_utils.helm_backend.launcher.values.helm_values_types import InfraValues


class TestDevShm:
    @pytest.mark.parametrize(
        "dev_shm",
        [
            {"mountPath": "/dev/shm"},
            {
                "mountPath": "/dev/shm",
                "hostPath": {"path": "/dev/shm"},
                "emptyDir": {"medium": "Memory"},
            },
        ],
        ids=["without-a-source", "with-two-sources"],
    )
    def test_dev_shm_requires_exactly_one_source(self, dev_shm: dict[str, object]) -> None:
        """Shared memory must declare one source rather than none or two competing sources."""
        values = {
            "image": {"repository": "radixark/miles", "tag": "dev"},
            "volumes": [],
            "devShm": dev_shm,
        }

        with pytest.raises(ValidationError, match="exactly one"):
            InfraValues.model_validate(values)
