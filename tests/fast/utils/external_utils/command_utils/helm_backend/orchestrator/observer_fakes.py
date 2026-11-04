from pathlib import Path

from miles.utils.external_utils.command_utils.helm_backend.orchestrator.observer import ObservedPod


class _ObserverProbe:
    def __init__(self, *, generations: list[Path | None | Exception], pods: list[ObservedPod | None]) -> None:
        self.generations = iter(generations)
        self.pods = iter(pods)
        self.generation_reads = 0
        self.pod_reads = 0
        self.sleeps: list[float] = []

    def read_generation(self) -> Path | None:
        self.generation_reads += 1
        value = next(self.generations)
        if isinstance(value, Exception):
            raise value
        return value

    def read_pod(self) -> ObservedPod | None:
        self.pod_reads += 1
        return next(self.pods)

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        assert len(self.sleeps) < 100, "Observer exceeded the scripted poll budget"
