class _FakeBackend:
    def __init__(self) -> None:
        self.calls: list[tuple[str, object]] = []

    def exec_command_cpu(self, command: str) -> None:
        self.calls.append(("exec_command_cpu", command))

    def convert_checkpoint(self, **kwargs: object) -> None:
        self.calls.append(("convert_checkpoint", kwargs))

    def hf_download_dataset(self, name: str, *, data_dir: str) -> None:
        self.calls.append(("hf_download_dataset", (name, data_dir)))
