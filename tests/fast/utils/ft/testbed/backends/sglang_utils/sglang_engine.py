from __future__ import annotations

import ray


@ray.remote(num_cpus=0, max_restarts=0)
class TestbedSGLangEngine:
    """Simulates an SGLang inference engine.

    Exposes health_generate() for the real RolloutHealthChecker to call.
    max_restarts=0 so ray.kill() permanently kills it, simulating a crash.
    """

    def health_generate(self) -> bool:
        return True
