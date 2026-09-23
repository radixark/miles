import pytest
from tests.utils.soak.recipes import gsm8k


class _FakeBackend:
    def __init__(self) -> None:
        self.calls: list[tuple[str, object]] = []

    def exec_command_cpu(self, command: str) -> None:
        self.calls.append(("exec_command_cpu", command))

    def convert_checkpoint(self, **kwargs: object) -> None:
        self.calls.append(("convert_checkpoint", kwargs))

    def hf_download_dataset(self, name: str, *, data_dir: str) -> None:
        self.calls.append(("hf_download_dataset", (name, data_dir)))


class _FakeTrainingLauncher:
    def __init__(self, *, error: BaseException | None = None) -> None:
        self.error = error
        self.calls: list[dict] = []

    def install(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(gsm8k, "launch_training", self)

    def __call__(self, **kwargs: object) -> None:
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
