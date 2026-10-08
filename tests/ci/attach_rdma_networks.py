"""Attach host-provisioned RoCE networks to an isolated CI job container."""

# doc-dev: docs/developer/ci/00-stage.md

import argparse
import http.client
import ipaddress
import json
import os
import re
import shutil
import socket
import subprocess
import time
from contextlib import closing
from pathlib import Path
from urllib.parse import urlencode

LABEL = "miles.ci.rdma"


def command(*args):
    return subprocess.check_output(args, text=True, timeout=30).strip()


def ip_json(*args):
    return json.loads(command("ip", "-j", *args))


class Docker(http.client.HTTPConnection):
    def __init__(self):
        super().__init__("localhost", timeout=30)

    def connect(self):
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.settimeout(self.timeout)
        self.sock.connect("/var/run/docker.sock")

    def call(self, method, path, body=None):
        self.request(
            method,
            path,
            body=None if body is None else json.dumps(body),
            headers={"Content-Type": "application/json"},
        )
        response = self.getresponse()
        payload = response.read()
        if response.status >= 300:
            raise RuntimeError(f"Docker {method} {path}: {response.status}: {payload.decode()}")
        return json.loads(payload) if payload else None


def validate_network(network):
    options = network["Options"]
    if (
        network["Driver"] != "ipvlan"
        or not network["Internal"]
        or not network["EnableIPv6"]
        or network["EnableIPv4"]
        or options.get("ipvlan_mode") != "l2"
        or options.get("ipvlan_flag") != "private"
        or not options.get("parent")
    ):
        raise ValueError(f"Unsafe RDMA network profile: {network['Name']}")
    config = json.loads(network["Labels"][LABEL])
    if not re.fullmatch(r"[a-zA-Z0-9_-]+", config["device"]):
        raise ValueError("Invalid RDMA device")
    if not config["routes"]:
        raise ValueError("RDMA network needs explicit fabric routes")
    for route in config["routes"]:
        if ipaddress.IPv6Network(route["dst"]).prefixlen == 0:
            raise ValueError("RDMA networks must not install a default route")
        if not ipaddress.IPv6Address(route["gateway"]).is_link_local:
            raise ValueError("Expected a link-local fabric gateway")
        if not isinstance(route["metric"], int) or route["metric"] < 1:
            raise ValueError("Invalid route metric")
    return config


def defaults():
    return [ip_json(family, "route", "show", "default") for family in ("-4", "-6")]


def endpoint_ready(device, address):
    root = Path("/sys/class/infiniband") / device / "ports/1"
    deadline = time.monotonic() + 10
    while True:
        for gid in root.joinpath("gids").glob("*"):
            try:
                ndev = root.joinpath("gid_attrs/ndevs", gid.name).read_text().strip()
                kind = root.joinpath("gid_attrs/types", gid.name).read_text().strip()
                if kind != "RoCE v2" or ipaddress.IPv6Address(gid.read_text().strip()) != address:
                    continue
                for link in ip_json("-6", "addr", "show", "dev", ndev):
                    for addr in link["addr_info"]:
                        if ipaddress.IPv6Address(addr["local"]) != address:
                            continue
                        if addr.get("dadfailed"):
                            raise RuntimeError(f"Duplicate IPv6 address on {ndev}")
                        if not addr.get("tentative"):
                            return ndev, int(gid.name)
            except OSError:
                continue
        if time.monotonic() >= deadline:
            raise RuntimeError(f"No usable local RoCE v2 GID for {device}: {address}")
        time.sleep(0.2)


def add_route(route, iface):
    current = ip_json("-6", "route", "show", "dev", iface)
    if any(all(row.get(key) == value for key, value in route.items()) for row in current):
        return None
    args = [
        route["dst"],
        "via",
        route["gateway"],
        "dev",
        iface,
        "metric",
        str(route["metric"]),
    ]
    command("ip", "-6", "route", "add", *args)
    return args


def attach(container_id):
    if not re.fullmatch(r"[a-f0-9]{64}", container_id):
        raise ValueError("Pass the full job.container.id from GitHub Actions")
    with closing(Docker()) as docker:
        query = urlencode({"filters": json.dumps({"label": [LABEL]})})
        networks = docker.call("GET", "/networks?" + query)
        if not networks:
            print("No provisioned RDMA networks on this Docker host")
            return
        version = tuple(int(part) for part in docker.call("GET", "/version")["ApiVersion"].split("."))
        if version < (1, 48):
            raise RuntimeError("Provisioned RDMA networks require Docker API 1.48 or newer")
        configs = [validate_network(network) for network in networks]
        if len({c["device"] for c in configs}) != len(configs):
            raise ValueError("Multiple RDMA networks configured for one HCA")
        container = docker.call("GET", f"/containers/{container_id}/json")
        mode = container["HostConfig"]["NetworkMode"]
        if mode in ("host", "none") or mode.startswith("container:"):
            raise ValueError("RDMA setup requires an existing isolated bridge network")
        existing = {ep["NetworkID"] for ep in container["NetworkSettings"]["Networks"].values()}
        if shutil.which("ip") is None:
            print("Installing iproute2 for provisioned RDMA networks", flush=True)
            subprocess.run(["apt-get", "update"], check=True, timeout=120)
            subprocess.run(
                ["apt-get", "install", "-y", "--no-install-recommends", "iproute2"],
                check=True,
                timeout=120,
                env=dict(os.environ, DEBIAN_FRONTEND="noninteractive"),
            )
        before = defaults()
        added_networks, added_routes = [], []
        try:
            # Covering fabric routes would make Docker reject subsequent endpoint subnets.
            for network in networks:
                if network["Id"] not in existing:
                    docker.call(
                        "POST",
                        f"/networks/{network['Id']}/connect",
                        {"Container": container_id, "EndpointConfig": {"GwPriority": -1}},
                    )
                    added_networks.append(network["Id"])
            container = docker.call("GET", f"/containers/{container_id}/json")
            endpoints = {ep["NetworkID"]: ep for ep in container["NetworkSettings"]["Networks"].values()}
            for network, config in zip(networks, configs, strict=True):
                address = ipaddress.IPv6Address(endpoints[network["Id"]]["GlobalIPv6Address"])
                iface, gid = endpoint_ready(config["device"], address)
                for route in config["routes"]:
                    added = add_route(route, iface)
                    if added:
                        added_routes.append(added)
                print(f"RDMA ready: {config['device']} {iface} gid_index={gid}", flush=True)
            if defaults() != before:
                raise RuntimeError("RDMA attachment changed a default route")
        except BaseException:
            for route in reversed(added_routes):
                try:
                    command("ip", "-6", "route", "del", *route)
                except Exception as exc:
                    print(f"Route rollback failed: {exc}", flush=True)
            for network_id in reversed(added_networks):
                try:
                    docker.call(
                        "POST",
                        f"/networks/{network_id}/disconnect",
                        {"Container": container_id, "Force": False},
                    )
                except Exception as exc:
                    print(f"Endpoint rollback failed: {exc}", flush=True)
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--container-id", default=os.environ.get("JOB_CONTAINER_ID", ""))
    attach(parser.parse_args().container_id)


if __name__ == "__main__":
    main()
