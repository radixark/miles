"""Use previously validated immutable TB2.1 templates without rebuilding images."""

import json
from pathlib import Path
from typing import Any

from harbor.environments.e2b import E2BEnvironment


class PrebuiltTB21E2B(E2BEnvironment):
    def __init__(self, *, template_manifest: str, **kwargs: Any) -> None:
        manifest = json.loads(Path(template_manifest).read_text())
        entries = {entry["task"]: entry for entry in manifest["results"]}
        name = kwargs["environment_name"].rsplit("/", 1)[-1]
        if name not in entries:
            raise ValueError(f"No immutable TB2.1 template for {name!r}")
        entry = entries[name]
        task_config = kwargs["task_env_config"]
        expected_image = entry["source_docker_image"]
        # These two v5 templates were built from the canonical task Dockerfiles,
        # with tmux/asciinema added, so their manifest intentionally has no image.
        if expected_image is None and name in {"qemu-alpine-ssh", "qemu-startup"}:
            expected_image = f"alexgshaw/{name}:20251031"
        if task_config.docker_image != expected_image:
            raise ValueError(f"Task image differs from the template manifest for {name}")
        if (task_config.cpus, task_config.memory_mb) != (entry["cpus"], entry["memory_mb"]):
            raise ValueError(f"Task resources differ from the template manifest for {name}")
        super().__init__(prebuilt_template_id=entry["template_id"], **kwargs)
