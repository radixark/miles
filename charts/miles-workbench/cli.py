#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

CHART_DIR = Path(__file__).resolve().parent
CHART_NAME = "miles-workbench"

# ================================= Rules the chart needs =================================

WRITE = ("create", "delete", "get", "list", "patch", "update", "watch")


def role_name(release: str) -> str:
    name = (release if CHART_NAME in release else f"{release}-{CHART_NAME}")[:52]
    return name[:-1] if name.endswith("-") else name


GRANTED_RULES: dict[str, tuple[str, ...]] = {
    "configmaps": WRITE,
    "secrets": WRITE,
    "serviceaccounts": WRITE,
    "services": WRITE,
    "pods": ("delete", "get", "list", "watch"),
    "pods/exec": ("create",),
    "pods/log": ("get",),
    "events": ("get", "list", "watch"),
    "persistentvolumeclaims": ("get", "list", "watch"),
    "statefulsets.apps": WRITE,
    "jobs.batch": WRITE,
}

LWS_RESOURCE = "leaderworkersets.leaderworkerset.x-k8s.io"

GRANTED_LWS_RULES: dict[str, tuple[str, ...]] = {
    LWS_RESOURCE: WRITE,
}

CHART_RULES: dict[str, tuple[str, ...]] = {
    "statefulsets.apps": ("create", "delete", "get", "patch"),
    "secrets": ("create", "delete", "get", "list", "update"),
}

CHART_SERVICE_ACCOUNT_RULES: dict[str, tuple[str, ...]] = {
    "serviceaccounts": ("create", "delete", "get", "patch"),
}

CHART_RBAC_RULES: dict[str, tuple[str, ...]] = {
    "roles.rbac.authorization.k8s.io": ("create", "delete", "get", "patch"),
    "rolebindings.rbac.authorization.k8s.io": ("create", "delete", "get", "patch"),
}

WORKFLOW_RULES: dict[str, tuple[str, ...]] = {
    "pods": ("get", "list"),
    "pods/exec": ("create",),
    "pods/log": ("get",),
    "statefulsets.apps": ("get", "list", "watch"),
}

# ====================================== Doctor ======================================


class Doctor:
    def __init__(self, namespace: str) -> None:
        self.namespace = namespace
        self.failed = False
        self.checks = 0
        self.failures = 0

    def report(self, ok: bool, message: str) -> bool:
        self.checks += 1
        if ok:
            print(f"PASS  {message}", flush=True)
        else:
            print(f"FAIL  {message}", file=sys.stderr, flush=True)
            self.failed = True
            self.failures += 1
        return ok

    def warn(self, message: str) -> None:
        print(f"WARN  {message}", file=sys.stderr, flush=True)

    def may_delegate_rules_it_does_not_hold(self, role: str) -> bool:
        return all(self.holds_on_roles(verb, role) for verb in ("escalate", "bind"))

    def holds_on_roles(self, verb: str, role: str) -> bool:
        if self.can_i(verb, "roles.rbac.authorization.k8s.io"):
            return True
        query = ["auth", "can-i", verb, f"roles.rbac.authorization.k8s.io/{role}", "-n", self.namespace]
        return self.kubectl(*query).returncode == 0

    def reset_counters(self) -> None:
        self.checks = 0
        self.failures = 0

    def kubectl(self, *args: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            ["kubectl", *args],
            capture_output=True,
            text=True,
        )

    def can_i(self, verb: str, resource: str) -> bool:
        target, _, subresource = resource.partition("/")
        args = ["auth", "can-i", verb, target]
        if subresource:
            args.append(f"--subresource={subresource}")
        args += ["-n", self.namespace]
        return self.kubectl(*args).returncode == 0

    def denied_rules(self, *rule_sets: dict[str, tuple[str, ...]]) -> list[str]:
        denied = []
        for rules in rule_sets:
            for resource, verbs in rules.items():
                missing = [verb for verb in verbs if not self.can_i(verb, resource)]
                if missing:
                    denied.append(f"{resource}({' '.join(missing)})")
        return denied

    def check_rules(self, what: str, *rule_sets: dict[str, tuple[str, ...]]) -> bool:
        denied = self.denied_rules(*rule_sets)
        message = f"may {what} in namespace {self.namespace}"
        if denied:
            return self.report(False, f"{message} (denied: {', '.join(denied)})")
        return self.report(True, message)

    def check_binary(self, binary: str) -> bool:
        return self.report(shutil.which(binary) is not None, f"{binary} is installed")

    def check_present(self, kind: str, name: str) -> bool:
        result = self.kubectl("get", kind, "--", name)
        if result.returncode == 0:
            return True
        output = (result.stdout + result.stderr).strip()
        if "(Forbidden)" in output:
            return True
        return self.report(False, f"{kind} {name} exists ({output})")

    def check_cluster_reachable(self) -> None:
        result = self.kubectl("get", "--raw", "/version")
        if result.returncode == 0:
            return
        output = (result.stdout + result.stderr).strip()
        self.report(False, f"cluster is reachable and your credentials are accepted ({output})")
        print(
            "fix your kubeconfig, credentials or network before reading anything below as a permission problem",
            file=sys.stderr,
            flush=True,
        )
        raise SystemExit(1)


def doctor(args: argparse.Namespace) -> int:
    checks = Doctor(namespace=args.namespace)

    checks.check_binary("kubectl")
    checks.check_binary("helm")
    if checks.failed:
        print("install the missing binaries before continuing", file=sys.stderr, flush=True)
        return 1

    checks.check_cluster_reachable()
    checks.check_present("namespace", args.namespace)
    checks.reset_counters()

    checks.check_rules("use the workbench", WORKFLOW_RULES)

    if args.rbac:
        checks.check_rules("install the chart", CHART_RULES, CHART_SERVICE_ACCOUNT_RULES, CHART_RBAC_RULES)

        granted = [GRANTED_RULES]
        if args.leader_worker_sets:
            checks.check_present("crd", LWS_RESOURCE)
            granted.append(GRANTED_LWS_RULES)

        denied = checks.denied_rules(*granted)
        if not denied:
            checks.report(True, f"may grant the workbench its Role in namespace {args.namespace}")
        elif checks.may_delegate_rules_it_does_not_hold(role_name(args.release)):
            checks.warn(
                f"you do not hold every rule the chart grants (denied: {', '.join(denied)}), but you have "
                "escalate and bind on roles, which is what Kubernetes checks when creating the Role and "
                "its RoleBinding"
            )
        else:
            checks.report(
                False,
                f"may grant the workbench its Role in namespace {args.namespace} "
                f"(denied: {', '.join(denied)}; escalate and bind on roles would also do)",
            )
            if any("leaderworkersets" in entry for entry in denied):
                print(
                    "a cluster admin must install the LWS CRDs and grant you the LeaderWorkerSet rights "
                    "before this chart can grant them to the workbench",
                    file=sys.stderr,
                    flush=True,
                )
    else:
        checks.check_rules("install the chart", CHART_RULES)

    if checks.failed:
        if checks.checks and checks.failures == checks.checks:
            print(
                "every check was denied: confirm the namespace name and your kubectl context before "
                "treating this as missing RBAC",
                file=sys.stderr,
                flush=True,
            )
        print("doctor failed", file=sys.stderr, flush=True)
        return 1

    print("doctor passed", flush=True)
    return 0


# ================================= Install and exec =================================


def run(command: list[str]) -> int:
    if shutil.which(command[0]) is None:
        print(f"FAIL  {command[0]} is installed", file=sys.stderr, flush=True)
        return 1
    print("+ " + " ".join(command), file=sys.stderr, flush=True)
    return subprocess.run(command).returncode


def helm_install_command(args: argparse.Namespace) -> list[str]:
    command = [
        "helm",
        "upgrade",
        "--install",
        args.release,
        str(CHART_DIR),
        "-n",
        args.namespace,
    ]
    if args.image_tag:
        command += ["--set-string", f"image.tag={args.image_tag}"]
    if not args.rbac:
        command += ["--set", "rbac.create=false"]
    if not args.leader_worker_sets:
        command += ["--set", "rbac.leaderWorkerSets=false"]
    for values_file in args.values:
        command += ["-f", str(values_file)]
    for override in args.set:
        command += ["--set", override]
    return command


def install(args: argparse.Namespace) -> int:
    overrides = [str(path) for path in args.values] + args.set
    if any("rbac" in override for override in overrides):
        print(
            "WARN  an rbac value is set through -f or --set; the doctor only knows about --no-rbac and --no-lws",
            file=sys.stderr,
            flush=True,
        )
    if not args.skip_doctor and doctor(args) != 0:
        return 1

    for command in [["helm", "dependency", "build", str(CHART_DIR)], helm_install_command(args)]:
        code = run(command)
        if code != 0:
            return code
    return 0


def exec_shell(args: argparse.Namespace) -> int:
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    return run(
        ["kubectl", "exec", "-it", f"statefulset/{role_name(args.release)}", "-n", args.namespace, "--", *command]
    )


# ========================================= CLI =========================================


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="cli.py",
        description="Install a miles-workbench release and get a shell in it.",
    )
    subcommands = parser.add_subparsers(dest="subcommand", required=True)

    for name, help_text in [
        ("install", "check, then helm upgrade --install"),
        ("exec", "shell into the pod"),
        ("doctor", "check that you may install the chart and use the workbench"),
    ]:
        subcommand = subcommands.add_parser(name, help=help_text)
        subcommand.add_argument("-n", "--namespace", required=True, help="namespace the workbench lives in")
        subcommand.add_argument("-r", "--release", required=True, help="helm release name")

    for name in ("install", "doctor"):
        subcommand = subcommands.choices[name]
        subcommand.add_argument(
            "--no-rbac", dest="rbac", action="store_false", help="the admin pre-created the identity"
        )
        subcommand.add_argument(
            "--no-lws",
            dest="leader_worker_sets",
            action="store_false",
            help="the admin grants LeaderWorkerSet rights separately",
        )

    installer = subcommands.choices["install"]
    installer.add_argument("--image-tag", help="training image tag to run")
    installer.add_argument("-f", "--values", action="append", default=[], type=Path, help="values file, repeatable")
    installer.add_argument("--set", action="append", default=[], help="raw helm --set override, repeatable")
    installer.add_argument("--skip-doctor", action="store_true", help="install without checking permissions first")

    shell = subcommands.choices["exec"]
    shell.add_argument("command", nargs=argparse.REMAINDER, help="command to run, bash by default")

    args = parser.parse_args(argv)
    if not args.namespace:
        parser.error("--namespace requires a value")
    if not args.release:
        parser.error("--release requires a value")
    if args.subcommand == "exec" and not args.command:
        args.command = ["bash"]
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    return {"install": install, "exec": exec_shell, "doctor": doctor}[args.subcommand](args)


if __name__ == "__main__":
    raise SystemExit(main())
