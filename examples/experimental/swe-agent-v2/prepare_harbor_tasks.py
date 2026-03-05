"""
Convert SWE-bench training data to Harbor task directories.

Reads the training JSONL (same format used by Miles run.sh) and creates
one Harbor task directory per instance. These directories are consumed by
server.py via Harbor's Trial API.

Usage:
    python prepare_harbor_tasks.py \
        --input /root/swe_train.jsonl \
        --output /root/harbor_tasks/swebench/

Each instance produces:
    {output}/{instance_id}/
    ├── instruction.md        # problem_statement
    ├── task.toml             # timeout, difficulty metadata
    ├── environment/
    │   └── Dockerfile        # FROM swebench Docker image
    ├── tests/
    │   ├── test.sh           # pytest + swebench grading harness
    │   └── config.json       # test_patch, test_cmd, FAIL_TO_PASS, etc.
    └── solution/
        └── solve.sh          # oracle patch (optional)
"""

import argparse
import json
import logging
import os
import textwrap
from pathlib import Path

logger = logging.getLogger(__name__)


def _try_harbor_adapter(input_path: str, output_dir: str) -> bool:
    """Try to use Harbor's official SWEBenchToHarbor adapter.

    Returns True if successful, False if Harbor adapter is not available.
    """
    try:
        from harbor.benchmarks.swebench.adapter import SWEBenchToHarbor

        adapter = SWEBenchToHarbor(
            input_path=input_path,
            output_dir=output_dir,
        )
        adapter.convert()
        return True
    except ImportError:
        logger.info("Harbor SWEBenchToHarbor adapter not available, using manual conversion")
        return False
    except Exception as e:
        logger.warning(f"Harbor adapter failed ({e}), falling back to manual conversion")
        return False


def _get_docker_image(instance_id: str, metadata: dict) -> str:
    """Get the SWE-bench Docker image name for an instance.

    SWE-bench eval images follow the convention:
        swebench/sweb.eval.x86_64.{instance_id}:latest
    """
    # SWE-bench image naming: replace special chars for Docker tag compatibility
    sanitized = instance_id.replace("/", "__")
    return f"swebench/sweb.eval.x86_64.{sanitized}:latest"


def _create_task_dir(
    instance_id: str,
    metadata: dict,
    output_dir: Path,
    docker_network: str | None = None,
) -> Path:
    """Create a Harbor task directory for a single SWE-bench instance."""
    task_dir = output_dir / instance_id
    task_dir.mkdir(parents=True, exist_ok=True)

    # instruction.md — problem statement
    problem_statement = metadata.get("problem_statement", "")
    (task_dir / "instruction.md").write_text(problem_statement)

    # task.toml — metadata
    repo = metadata.get("repo", "")
    version = metadata.get("version", "")
    task_toml = textwrap.dedent(
        f"""\
        [task]
        name = "{instance_id}"
        repo = "{repo}"
        version = "{version}"

        [limits]
        timeout = 1800
        step_limit = 250
        cost_limit = 3
    """
    )
    (task_dir / "task.toml").write_text(task_toml)

    # environment/Dockerfile
    env_dir = task_dir / "environment"
    env_dir.mkdir(exist_ok=True)
    docker_image = _get_docker_image(instance_id, metadata)
    dockerfile = textwrap.dedent(
        f"""\
        FROM {docker_image}
        # SWE-bench evaluation environment for {instance_id}
        # Base image contains the repo at the correct commit with conda env set up.
    """
    )
    (env_dir / "Dockerfile").write_text(dockerfile)

    # environment/docker-compose.yaml — network override so the container
    # can reach Miles Router.  Harbor's DockerEnvironment merges this file
    # with its base compose files automatically.
    if docker_network:
        compose_yaml = textwrap.dedent(
            f"""\
            services:
              main:
                networks:
                  - {docker_network}
            networks:
              {docker_network}:
                external: true
        """
        )
        (env_dir / "docker-compose.yaml").write_text(compose_yaml)

    # tests/
    tests_dir = task_dir / "tests"
    tests_dir.mkdir(exist_ok=True)

    # tests/config.json — test configuration
    test_config = {
        "instance_id": instance_id,
        "repo": repo,
        "version": version,
        "base_commit": metadata.get("base_commit", ""),
        "test_patch": metadata.get("test_patch", ""),
        "test_cmd": metadata.get("test_cmd", ""),
        "FAIL_TO_PASS": metadata.get("FAIL_TO_PASS", []),
        "PASS_TO_PASS": metadata.get("PASS_TO_PASS", []),
    }
    (tests_dir / "config.json").write_text(json.dumps(test_config, indent=2))

    # tests/test.sh — grading script using swebench harness
    test_sh = textwrap.dedent(
        f"""\
        #!/bin/bash
        # Run tests and grade using swebench harness
        set -e

        INSTANCE_ID="{instance_id}"
        CONFIG_DIR="$(dirname "$0")"

        # Apply test patch if present
        if [ -f "$CONFIG_DIR/config.json" ]; then
            TEST_PATCH=$(python3 -c "import json; print(json.load(open('$CONFIG_DIR/config.json')).get('test_patch', ''))")
            if [ -n "$TEST_PATCH" ]; then
                echo "$TEST_PATCH" | git apply --allow-empty
            fi
        fi

        # Run the test command
        TEST_CMD=$(python3 -c "import json; print(json.load(open('$CONFIG_DIR/config.json')).get('test_cmd', 'pytest'))")
        eval "$TEST_CMD" || true

        # Grade using swebench harness
        python3 -c "
        from swebench.harness.grading import grade_instance
        import json

        config = json.load(open('$CONFIG_DIR/config.json'))
        result = grade_instance(config)
        print(json.dumps(result))
        "
    """
    )
    (tests_dir / "test.sh").write_text(test_sh)
    os.chmod(tests_dir / "test.sh", 0o755)

    # solution/ (optional — oracle patch)
    patch = metadata.get("patch", "")
    if patch:
        sol_dir = task_dir / "solution"
        sol_dir.mkdir(exist_ok=True)
        solve_sh = textwrap.dedent(
            f"""\
            #!/bin/bash
            # Oracle solution — apply gold patch
            cat << 'PATCH_EOF' | git apply
            {patch}
            PATCH_EOF
        """
        )
        (sol_dir / "solve.sh").write_text(solve_sh)
        os.chmod(sol_dir / "solve.sh", 0o755)

    return task_dir


def convert(input_path: str, output_dir: str, docker_network: str | None = None) -> int:
    """Convert all instances from JSONL to Harbor task directories.

    Args:
        docker_network: If set, each task's ``environment/docker-compose.yaml``
            will join the container to this external Docker network so it can
            reach Miles Router.

    Returns the number of tasks created.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Try Harbor's official adapter first
    if _try_harbor_adapter(input_path, output_dir):
        count = sum(1 for d in output_path.iterdir() if d.is_dir())
        logger.info(f"Harbor adapter created {count} task directories")
        return count

    # Manual conversion fallback
    count = 0
    with open(input_path) as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError as e:
                logger.warning(f"Skipping line {line_num}: invalid JSON ({e})")
                continue

            metadata = data.get("metadata", data)
            instance_id = metadata.get("instance_id", "")
            if not instance_id:
                logger.warning(f"Skipping line {line_num}: no instance_id")
                continue

            _create_task_dir(instance_id, metadata, output_path, docker_network=docker_network)
            count += 1
            if count % 100 == 0:
                logger.info(f"Created {count} task directories...")

    logger.info(f"Created {count} task directories in {output_dir}")
    return count


def main():
    parser = argparse.ArgumentParser(description="Convert SWE-bench JSONL to Harbor task directories")
    parser.add_argument(
        "--input",
        required=True,
        help="Path to SWE-bench training JSONL (e.g. /root/swe_train.jsonl)",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output directory for Harbor tasks (e.g. /root/harbor_tasks/swebench/)",
    )
    parser.add_argument(
        "--docker-network",
        default=None,
        help="External Docker network for containers to join (e.g. swe-net). "
        "Generates a docker-compose.yaml override in each task's environment/ dir.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    convert(args.input, args.output, docker_network=args.docker_network)


if __name__ == "__main__":
    main()
