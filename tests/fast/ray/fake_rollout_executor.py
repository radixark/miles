class FakeRemoteMethod:
    def __init__(self) -> None:
        self.calls: list[tuple] = []

    def remote(self, *args) -> str:
        self.calls.append(args)
        return f"object-ref-{len(self.calls)}"


class FakeRolloutExecutor:
    def __init__(self) -> None:
        self.set_train_parallel_config = FakeRemoteMethod()
