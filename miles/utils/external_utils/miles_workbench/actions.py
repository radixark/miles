from __future__ import annotations

import logging
import shutil

from miles.utils.external_utils.command_utils.common import run_process
from miles.utils.external_utils.command_utils.helm_backend.launcher.command_wrapper import Helm, Kubectl
from miles.utils.external_utils.miles_workbench.naming import CHART_DIR, PACKAGE, object_name
from miles.utils.external_utils.miles_workbench.options import ExecArgs, InstallArgs, ReleaseArgs
from miles.utils.external_utils.miles_workbench.preflight.runner import run_preflight_checks
from miles.utils.external_utils.miles_workbench.render import helm_value_overrides

logger = logging.getLogger(__name__)


def install(args: InstallArgs) -> None:
    if args.dry_run:
        run_preflight_checks(args)
        logger.info("Dry run: nothing was created, installed or waited for")
        return

    _ensure_namespace(args.namespace)

    if args.skip_preflight:
        _run(["helm", "dependency", "build", str(CHART_DIR)])
    else:
        run_preflight_checks(args)
    _run(_helm_install_command(args))
    _wait_until_ready(namespace=args.namespace, release=args.release, timeout=args.timeout)

    logger.info(
        "The workbench is ready; get a shell with: python -m %s exec -n %s -r %s",
        PACKAGE,
        args.namespace,
        args.release,
    )


def exec_shell(args: ExecArgs) -> None:
    command = (args.command[1:] if args.command[:1] == ("--",) else args.command) or ("bash",)
    _run(["kubectl", "exec", "-it", f"statefulset/{object_name(args.release)}", "-n", args.namespace, "--", *command])


def uninstall(args: ReleaseArgs) -> None:
    _run(["helm", "uninstall", args.release, "--namespace", args.namespace])


def _helm_install_command(args: InstallArgs) -> list[str]:
    command = Helm.upgrade_command(args.release, args.namespace, CHART_DIR, [])
    command += ["--set-string", f"objectName={object_name(args.release)}"]
    return command + helm_value_overrides(args)


def _ensure_namespace(namespace: str) -> None:
    _require_binary("kubectl")

    existing = Kubectl.run_raw("get", "namespace", "--", namespace)
    if existing.returncode == 0:
        return

    output = (existing.stdout + existing.stderr).strip()
    if "(NotFound)" in output:
        _run(["kubectl", "create", "namespace", namespace])
        return
    if "(Forbidden)" in output:
        _probe_namespace_from_inside(namespace)
        return

    logger.error("FAIL  could not read namespace %s (%s)", namespace, output)
    raise SystemExit(1)


def _probe_namespace_from_inside(namespace: str) -> None:
    result = Kubectl.run_raw("get", "serviceaccounts", "-n", namespace, "-o", "name")
    output = (result.stdout + result.stderr).strip()
    if result.returncode == 0:
        return

    if "(NotFound)" in output:
        logger.error(
            "FAIL  namespace %s does not exist and this account may not create it; ask a cluster admin for it",
            namespace,
        )
        raise SystemExit(1)

    logger.warning(
        "WARN  this account may not read namespace %s itself, so nothing here says whether it exists; "
        "continuing on the namespaced rights it does hold (%s)",
        namespace,
        output,
    )


def _wait_until_ready(*, namespace: str, release: str, timeout: int) -> None:
    fullname = object_name(release)
    command = ["kubectl", "rollout", "status", f"statefulset/{fullname}", "-n", namespace, f"--timeout={timeout}s"]
    if (code := _exit_code_of(command)) == 0:
        return

    logger.error("The workbench was not ready within %ss", timeout)
    _exit_code_of(["kubectl", "get", "pods", "-n", namespace])
    _exit_code_of(["kubectl", "describe", "statefulset", fullname, "-n", namespace])
    raise SystemExit(code)


def _run(command: list[str]) -> None:
    if (code := _exit_code_of(command)) != 0:
        logger.error("FAIL  %s exited %s", " ".join(command), code)
        raise SystemExit(code)


def _exit_code_of(command: list[str]) -> int:
    _require_binary(command[0])
    return run_process(command, capture_output=False, check=False).returncode


def _require_binary(binary: str) -> None:
    if shutil.which(binary) is None:
        logger.error("FAIL  %s is installed", binary)
        raise SystemExit(1)
