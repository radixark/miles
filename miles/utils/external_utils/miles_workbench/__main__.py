from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Annotated

import typer

from miles.utils.external_utils.miles_workbench.actions import (
    collect_diagnosis_command,
    exec_shell,
    install,
    uninstall,
)
from miles.utils.external_utils.miles_workbench.naming import DEFAULT_RELEASE, PROGRAM_NAME, run_release_name
from miles.utils.external_utils.miles_workbench.options import DiagnosisArgs, ExecArgs, InstallArgs, ReleaseArgs
from miles.utils.workers.types import DeployComponent

app = typer.Typer(
    name=PROGRAM_NAME,
    help="Install a miles-workbench release and get a shell in it.",
    no_args_is_help=True,
    add_completion=False,
)


def _non_empty(value: str) -> str:
    if not value:
        raise typer.BadParameter("names a kubernetes object, so an empty value can only be a dangling flag")
    return value


Namespace = Annotated[
    str, typer.Option("-n", "--namespace", help="Namespace the workbench lives in", callback=_non_empty)
]
Release = Annotated[str, typer.Option("-r", "--release", help="helm release name", callback=_non_empty)]


@app.command(name="install", help="Check, then helm upgrade --install, then wait for the pod")
def install_command(
    namespace: Namespace,
    release: Release = DEFAULT_RELEASE,
    rbac: Annotated[bool, typer.Option(help="Create the identity, unless the admin pre-created it")] = True,
    lws: Annotated[bool, typer.Option(help="Grant LeaderWorkerSet rights, unless the admin grants them")] = True,
    dry_run: Annotated[bool, typer.Option(help="Run the checks only, changing nothing in the cluster")] = False,
    values: Annotated[list[Path] | None, typer.Option("-f", "--values", help="Values file, repeatable")] = None,
    overrides: Annotated[list[str] | None, typer.Option("--set", help="Raw helm --set, repeatable")] = None,
    skip_preflight: Annotated[bool, typer.Option(help="Install without checking permissions first")] = False,
    timeout: Annotated[int, typer.Option(help="Seconds to wait for the workbench pod to become ready")] = 600,
) -> None:
    args = InstallArgs(
        namespace=namespace,
        release=release,
        rbac=rbac,
        lws=lws,
        dry_run=dry_run,
        values=tuple(values or ()),
        overrides=tuple(overrides or ()),
        skip_preflight=skip_preflight,
        timeout=timeout,
    )
    install(args)


@app.command(
    name="exec",
    help="Shell into the pod",
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
)
def exec_command(
    namespace: Namespace,
    release: Release = DEFAULT_RELEASE,
    command: Annotated[list[str] | None, typer.Argument(help="Command to run, bash by default")] = None,
) -> None:
    args = ExecArgs(namespace=namespace, release=release, command=tuple(command or ()))
    exec_shell(args)


@app.command(name="stop", help="helm uninstall the release of one run, keeping the workbench")
def stop_command(
    namespace: Namespace,
    run_id: Annotated[str, typer.Argument(help="Run id the release is named after")],
    deploy_component: Annotated[
        DeployComponent, typer.Option(help="Which deployment of the run to stop, when it was deployed in parts")
    ] = DeployComponent.ALL,
    deploy_instance_id: Annotated[
        str | None, typer.Option(help="Which instance of that deployment to stop, when it was deployed in instances")
    ] = None,
) -> None:
    uninstall(ReleaseArgs(namespace=namespace, release=run_release_name(run_id, deploy_component, deploy_instance_id)))


@app.command(name="uninstall", help="helm uninstall the release, keeping the namespace")
def uninstall_command(namespace: Namespace, release: Release = DEFAULT_RELEASE) -> None:
    uninstall(ReleaseArgs(namespace=namespace, release=release))


@app.command(name="collect-diagnosis", help="Write pod logs, describes and events into one directory")
def diagnosis_command(
    namespace: Namespace,
    output_dir: Annotated[Path | None, typer.Option(help="Directory the diagnosis directory is created in")] = None,
    run_dir: Annotated[Path | None, typer.Option(help="Run directory whose latest verdict to copy")] = None,
) -> None:
    args = DiagnosisArgs(namespace=namespace, output_dir=output_dir or Path.cwd(), run_dir=run_dir)
    collect_diagnosis_command(args)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stderr, force=True)
    app()


if __name__ == "__main__":
    main()
