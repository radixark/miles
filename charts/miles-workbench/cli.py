#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import NamedTuple

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
    "pods": ("delete", "get", "list", "patch", "update", "watch"),
    "pods/exec": ("create",),
    "pods/log": ("get",),
    "events": ("get", "list", "watch"),
    "persistentvolumeclaims": ("get", "list", "watch"),
    "deployments.apps": WRITE,
    "statefulsets.apps": WRITE,
    "jobs.batch": WRITE,
    "roles.rbac.authorization.k8s.io": WRITE,
    "rolebindings.rbac.authorization.k8s.io": WRITE,
}

MANAGED_BY = "Helm"
CHART_FAMILY = ("miles-workbench", "miles-run")
LWS_API_GROUP = "leaderworkerset.x-k8s.io"
LWS_RESOURCE = "leaderworkersets.leaderworkerset.x-k8s.io"
LWS_CONTROLLER_NAMESPACE = "lws-system"
LWS_CONTROLLER_DEPLOYMENT = "lws-controller-manager"
AVAILABLE_CONDITION = "jsonpath={.status.conditions[?(@.type=='Available')].status}"

NAMESPACE_KINDS = (
    "all",
    "configmap",
    "secret",
    "persistentvolumeclaim",
    "serviceaccount",
    "role.rbac.authorization.k8s.io",
    "rolebinding.rbac.authorization.k8s.io",
    "leaderworkerset.leaderworkerset.x-k8s.io",
)
UNSERVED_RESOURCE_MARKERS = ("doesn't have a resource type", "could not find the requested resource")
CLUSTER_PROVIDED_RESOURCES = ("configmap/kube-root-ca.crt", "serviceaccount/default")
DEFAULT_TOKEN_PREFIX = "secret/default-token-"

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


class RbacPlan(NamedTuple):
    creates_role: bool
    grants_leader_worker_sets: bool


class Doctor:
    def __init__(self, namespace: str) -> None:
        self.namespace = namespace
        self.failed = False
        self.checks = 0
        self.failures = 0

    def report(self, ok: bool, message: str, counted: bool = True) -> bool:
        if counted:
            self.checks += 1
        if ok:
            print(f"PASS  {message}", flush=True)
        else:
            print(f"FAIL  {message}", file=sys.stderr, flush=True)
            self.failed = True
            if counted:
                self.failures += 1
        return ok

    def warn(self, message: str) -> None:
        print(f"WARN  {message}", file=sys.stderr, flush=True)

    def report_unverifiable(self, message: str, reason: str) -> bool:
        print(f"UNKNOWN  {message}: this account may not look, so nothing here confirms it ({reason})", flush=True)
        return False

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

    def check_present(self, kind: str, name: str, namespace: str | None = None) -> bool:
        scope = ["-n", namespace] if namespace else []
        result = self.kubectl("get", kind, *scope, "--", name)
        where = f" in namespace {namespace}" if namespace else ""
        message = f"{kind} {name} exists{where}"
        if result.returncode == 0:
            return True
        output = (result.stdout + result.stderr).strip()
        if "(Forbidden)" in output:
            return self.report_unverifiable(message, output)
        return self.report(False, f"{message} ({output})")

    def check_leader_worker_set_api(self) -> bool:
        message = f"the cluster serves {LWS_RESOURCE}"
        result = self.kubectl("api-resources", "--api-group", LWS_API_GROUP, "-o", "name")
        output = (result.stdout + result.stderr).strip()
        if result.returncode != 0:
            return self.report_unverifiable(message, output)
        if LWS_RESOURCE not in result.stdout.split():
            return self.report(
                False, f"{message} (api discovery served {output or 'nothing'} in {LWS_API_GROUP})", counted=False
            )
        return self.report(True, message, counted=False)

    def check_leader_worker_set_controller(self) -> bool:
        message = f"deployment {LWS_CONTROLLER_DEPLOYMENT} is available in namespace {LWS_CONTROLLER_NAMESPACE}"
        result = self.kubectl(
            "get",
            "deployment.apps",
            "-n",
            LWS_CONTROLLER_NAMESPACE,
            "-o",
            AVAILABLE_CONDITION,
            "--",
            LWS_CONTROLLER_DEPLOYMENT,
        )
        output = (result.stdout + result.stderr).strip()
        if result.returncode != 0:
            if "(Forbidden)" in output:
                return self.report_unverifiable(message, output)
            return self.report(False, f"{message} ({output})", counted=False)
        if output != "True":
            return self.report(
                False, f"{message} (the Available condition reads {output or 'nothing'})", counted=False
            )
        return self.report(True, message, counted=False)

    def check_rbac_plan(self, args: argparse.Namespace) -> RbacPlan:
        message = "your values render the chart"
        rendered = render_chart(args)
        if rendered.returncode != 0:
            self.report(False, f"{message} ({(rendered.stdout + rendered.stderr).strip()})", counted=False)
            return RbacPlan(creates_role=args.rbac, grants_leader_worker_sets=args.leader_worker_sets)

        self.report(True, message, counted=False)
        return rbac_plan_of(rendered.stdout)

    def check_namespace_holds_only(self) -> bool:
        message = f"namespace {self.namespace} holds nothing but Miles releases"
        family = ",".join(CHART_FAMILY)
        selectors = [
            f"app.kubernetes.io/managed-by!={MANAGED_BY}",
            f"app.kubernetes.io/managed-by={MANAGED_BY},app.kubernetes.io/name notin ({family})",
        ]

        foreign: list[str] = []
        for kind in NAMESPACE_KINDS:
            for selector in selectors:
                result = self.kubectl("get", kind, "-n", self.namespace, "-l", selector, "-o", "name")
                output = (result.stdout + result.stderr).strip()
                if result.returncode != 0:
                    if any(marker in output for marker in UNSERVED_RESOURCE_MARKERS):
                        break
                    return self.report(False, f"{message} (could not list {kind}: {output})", counted=False)
                foreign += [
                    name for name in result.stdout.split() if name not in foreign and not is_cluster_provided(name)
                ]

        if foreign:
            return self.report(False, f"{message} (found: {', '.join(foreign)})", counted=False)
        return self.report(True, message, counted=False)

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


def run_preflight_checks(args: argparse.Namespace) -> int:
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
    checks.check_namespace_holds_only()

    plan = checks.check_rbac_plan(args)

    if plan.grants_leader_worker_sets:
        checks.check_leader_worker_set_api()
        checks.check_leader_worker_set_controller()

    if plan.creates_role:
        checks.check_rules("install the chart", CHART_RULES, CHART_SERVICE_ACCOUNT_RULES, CHART_RBAC_RULES)

        granted = [GRANTED_RULES]
        if plan.grants_leader_worker_sets:
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
        print("preflight checks failed", file=sys.stderr, flush=True)
        return 1

    print("preflight checks passed", flush=True)
    return 0


def render_chart(args: argparse.Namespace) -> subprocess.CompletedProcess:
    if not args.dry_run:
        return render_chart_from(args, chart_dir=CHART_DIR)

    with tempfile.TemporaryDirectory() as scratch:
        charts_copy = Path(scratch) / CHART_DIR.parent.name
        shutil.copytree(CHART_DIR.parent, charts_copy)
        return render_chart_from(args, chart_dir=charts_copy / CHART_DIR.name)


def render_chart_from(args: argparse.Namespace, *, chart_dir: Path) -> subprocess.CompletedProcess:
    build = subprocess.run(
        ["helm", "dependency", "build", str(chart_dir)],
        capture_output=True,
        text=True,
    )
    if build.returncode != 0:
        return build
    return subprocess.run(
        ["helm", "template", args.release, str(chart_dir), "-n", args.namespace, *helm_value_overrides(args)],
        capture_output=True,
        text=True,
    )


def is_cluster_provided(name: str) -> bool:
    return name in CLUSTER_PROVIDED_RESOURCES or name.startswith(DEFAULT_TOKEN_PREFIX)


def rbac_plan_of(rendered: str) -> RbacPlan:
    roles = [document for document in rendered.split("\n---") if "\nkind: Role\n" in document]
    return RbacPlan(
        creates_role=bool(roles),
        grants_leader_worker_sets=any("leaderworkersets" in document for document in roles),
    )


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
        command += ["--set-string", f"infra.image.tag={args.image_tag}"]
    return command + helm_value_overrides(args)


def helm_value_overrides(args: argparse.Namespace) -> list[str]:
    overrides: list[str] = []
    if not args.rbac:
        overrides += ["--set", "rbac.create=false"]
    if not args.leader_worker_sets:
        overrides += ["--set", "rbac.leaderWorkerSets=false"]
    for values_file in args.values:
        overrides += ["-f", str(values_file)]
    for override in args.set:
        overrides += ["--set", override]
    return overrides


def install(args: argparse.Namespace) -> int:
    if args.dry_run:
        if run_preflight_checks(args) != 0:
            return 1
        print("dry run: nothing was created, installed or waited for", flush=True)
        return 0

    code = ensure_namespace(args.namespace)
    if code != 0:
        return code

    if not args.skip_doctor and run_preflight_checks(args) != 0:
        return 1

    for command in [["helm", "dependency", "build", str(CHART_DIR)], helm_install_command(args)]:
        code = run(command)
        if code != 0:
            return code

    code = wait_until_ready(namespace=args.namespace, release=args.release, timeout=args.timeout)
    if code != 0:
        return code

    print(
        f"the workbench is ready; get a shell with: {CHART_DIR / 'cli.py'} exec "
        f"-n {args.namespace} -r {args.release}",
        flush=True,
    )
    return 0


def ensure_namespace(namespace: str) -> int:
    if shutil.which("kubectl") is None:
        print("FAIL  kubectl is installed", file=sys.stderr, flush=True)
        return 1
    existing = subprocess.run(["kubectl", "get", "namespace", "--", namespace], capture_output=True, text=True)
    if existing.returncode == 0:
        return 0

    output = (existing.stdout + existing.stderr).strip()
    if "(NotFound)" in output:
        return run(["kubectl", "create", "namespace", namespace])
    if "(Forbidden)" in output:
        return probe_namespace_from_inside(namespace)

    print(f"FAIL  could not read namespace {namespace} ({output})", file=sys.stderr, flush=True)
    return 1


def probe_namespace_from_inside(namespace: str) -> int:
    result = subprocess.run(
        ["kubectl", "get", "serviceaccounts", "-n", namespace, "-o", "name"],
        capture_output=True,
        text=True,
    )
    output = (result.stdout + result.stderr).strip()
    if result.returncode == 0:
        return 0
    if "(NotFound)" in output:
        print(
            f"FAIL  namespace {namespace} does not exist and this account may not create it; "
            "ask a cluster admin for it",
            file=sys.stderr,
            flush=True,
        )
        return 1

    print(
        f"WARN  this account may not read namespace {namespace} itself, so nothing here says whether it "
        f"exists; continuing on the namespaced rights it does hold ({output})",
        file=sys.stderr,
        flush=True,
    )
    return 0


def wait_until_ready(namespace: str, release: str, timeout: int) -> int:
    fullname = role_name(release)
    code = run(["kubectl", "rollout", "status", f"statefulset/{fullname}", "-n", namespace, f"--timeout={timeout}s"])
    if code != 0:
        print(f"the workbench was not ready within {timeout}s", file=sys.stderr, flush=True)
        run(["kubectl", "get", "pods", "-n", namespace])
        run(["kubectl", "describe", "statefulset", fullname, "-n", namespace])
    return code


def exec_shell(args: argparse.Namespace) -> int:
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    return run(
        ["kubectl", "exec", "-it", f"statefulset/{role_name(args.release)}", "-n", args.namespace, "--", *command]
    )


def uninstall(args: argparse.Namespace) -> int:
    return run(["helm", "uninstall", args.release, "--namespace", args.namespace])


def collect_diagnosis(args: argparse.Namespace) -> int:
    if shutil.which("kubectl") is None:
        print("FAIL  kubectl is installed", file=sys.stderr, flush=True)
        return 1

    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir = args.output_dir / f"miles-diagnosis-{args.namespace}-{stamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    failed: list[str] = []
    if not capture(
        path=output_dir / "events.txt",
        command=["kubectl", "get", "events", "-n", args.namespace, "--sort-by=.lastTimestamp"],
    ):
        failed.append("events")

    listing = list_pods(namespace=args.namespace)
    if not listing.listed:
        failed.append(f"pod listing in namespace {args.namespace}")
    for pod in listing.names:
        if not capture(
            path=output_dir / f"{pod}.log",
            command=["kubectl", "logs", pod, "-n", args.namespace, "--all-containers"],
        ):
            failed.append(f"logs of {pod}")
        capture(
            path=output_dir / f"{pod}.previous.log",
            command=["kubectl", "logs", pod, "-n", args.namespace, "--all-containers", "--previous"],
            skip_when_it_fails=True,
        )
        if not capture(
            path=output_dir / f"{pod}.describe.txt",
            command=["kubectl", "describe", "pod", pod, "-n", args.namespace],
        ):
            failed.append(f"describe of {pod}")
    if args.run_dir is not None:
        verdict = args.run_dir / "orchestrator.exit"
        text = verdict.read_text() if verdict.is_file() else f"{verdict} does not exist\n"
        (output_dir / "orchestrator.exit").write_text(text)

    print(str(output_dir), flush=True)
    if failed:
        print(
            f"FAIL  the diagnosis is incomplete, these could not be collected: {', '.join(failed)}",
            file=sys.stderr,
            flush=True,
        )
        return 1
    return 0


class PodListing(NamedTuple):
    listed: bool
    names: list[str]


def list_pods(namespace: str) -> PodListing:
    result = subprocess.run(
        ["kubectl", "get", "pods", "-n", namespace, "-o", "name"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(f"WARN  could not list pods in namespace {namespace}", file=sys.stderr, flush=True)
        return PodListing(listed=False, names=[])
    return PodListing(listed=True, names=[line.partition("/")[2] for line in result.stdout.split() if line])


def capture(path: Path, command: list[str], skip_when_it_fails: bool = False) -> bool:
    print("+ " + " ".join(command), file=sys.stderr, flush=True)
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0 and skip_when_it_fails:
        return True
    path.write_text(result.stdout + result.stderr)
    return result.returncode == 0


# ========================================= CLI =========================================


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="cli.py",
        description="Install a miles-workbench release and get a shell in it.",
    )
    subcommands = parser.add_subparsers(dest="subcommand", required=True)

    for name, help_text in [
        ("install", "check, then helm upgrade --install, then wait for the pod"),
        ("exec", "shell into the pod"),
        ("uninstall", "helm uninstall the release, keeping the namespace"),
        ("collect-diagnosis", "write pod logs, describes and events into one directory"),
    ]:
        subcommand = subcommands.add_parser(name, help=help_text)
        subcommand.add_argument("-n", "--namespace", required=True, help="namespace the workbench lives in")
        subcommand.add_argument("-r", "--release", required=True, help="helm release name")

    installer = subcommands.choices["install"]
    installer.add_argument("--no-rbac", dest="rbac", action="store_false", help="the admin pre-created the identity")
    installer.add_argument(
        "--no-lws",
        dest="leader_worker_sets",
        action="store_false",
        help="the admin grants LeaderWorkerSet rights separately",
    )
    installer.add_argument(
        "--dry-run", action="store_true", help="run the checks only, changing nothing in the cluster"
    )
    installer.add_argument("--image-tag", help="training image tag to run")
    installer.add_argument("-f", "--values", action="append", default=[], type=Path, help="values file, repeatable")
    installer.add_argument("--set", action="append", default=[], help="raw helm --set override, repeatable")
    installer.add_argument("--skip-doctor", action="store_true", help="install without checking permissions first")
    installer.add_argument(
        "--timeout", type=int, default=600, help="seconds to wait for the workbench pod to become ready"
    )

    shell = subcommands.choices["exec"]
    shell.add_argument("command", nargs=argparse.REMAINDER, help="command to run, bash by default")

    diagnosis = subcommands.choices["collect-diagnosis"]
    diagnosis.add_argument(
        "--output-dir", type=Path, default=Path.cwd(), help="directory the diagnosis directory is created in"
    )
    diagnosis.add_argument("--run-dir", type=Path, help="run directory whose orchestrator.exit verdict to copy")

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
    return {
        "install": install,
        "exec": exec_shell,
        "uninstall": uninstall,
        "collect-diagnosis": collect_diagnosis,
    }[
        args.subcommand
    ](args)


if __name__ == "__main__":
    raise SystemExit(main())
