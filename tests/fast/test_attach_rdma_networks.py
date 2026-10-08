import copy
import json
import unittest
from unittest.mock import patch

from tests.ci import attach_rdma_networks as rdma

CID = "a" * 64


def network(index):
    return {
        "Id": str(index),
        "Name": f"rdma-{index}",
        "Driver": "ipvlan",
        "Internal": True,
        "EnableIPv6": True,
        "EnableIPv4": False,
        "Options": {
            "parent": f"bond{index}",
            "ipvlan_mode": "l2",
            "ipvlan_flag": "private",
        },
        "Labels": {
            rdma.LABEL: json.dumps(
                {
                    "device": f"mlx5_bond_{index}",
                    "routes": [
                        {
                            "dst": "2001:db8::/32",
                            "gateway": "fe80::1",
                            "metric": 1010 + index,
                        }
                    ],
                }
            )
        },
    }


class AttachTests(unittest.TestCase):
    def setUp(self):
        self.networks = [network(0), network(1)]
        self.events = []
        self.connected = set()
        self.mode = "bridge"
        self.fail_connect = None
        self.version = "1.48"
        outer = self

        class Docker:
            def call(self, method, path, body=None):
                if path.startswith("/networks?"):
                    return outer.networks
                if path == "/version":
                    return {"ApiVersion": outer.version}
                if path.startswith("/containers/"):
                    return {
                        "HostConfig": {"NetworkMode": outer.mode},
                        "NetworkSettings": {
                            "Networks": {
                                nid: {
                                    "NetworkID": nid,
                                    "GlobalIPv6Address": f"2001:db8:{nid}::10",
                                }
                                for nid in outer.connected
                            }
                        },
                    }
                nid = path.split("/")[2]
                if path.endswith("/connect"):
                    self_check = body["EndpointConfig"]["GwPriority"]
                    assert self_check == -1
                    if nid == outer.fail_connect:
                        raise RuntimeError("connect failed")
                    outer.connected.add(nid)
                    outer.events.append(("connect", nid))
                elif path.endswith("/disconnect"):
                    outer.connected.remove(nid)
                    outer.events.append(("disconnect", nid))
                else:
                    raise AssertionError(path)

            def close(self):
                pass

        def add_route(route, iface):
            self.events.append(("route", iface))
            return [iface]

        patches = [
            patch.object(rdma, "Docker", Docker),
            patch.object(rdma.shutil, "which", return_value="/sbin/ip"),
            patch.object(
                rdma,
                "endpoint_ready",
                side_effect=lambda dev, addr: ("eth" + dev[-1], 7),
            ),
            patch.object(rdma, "add_route", side_effect=add_route),
            patch.object(rdma, "defaults", return_value=[[{"dev": "eth0"}], []]),
            patch.object(
                rdma,
                "command",
                side_effect=lambda *args: self.events.append(("command", args)),
            ),
        ]
        for item in patches:
            item.start()
            self.addCleanup(item.stop)

    def test_all_endpoints_connect_before_fabric_routes(self):
        rdma.attach(CID)
        self.assertEqual(
            self.events,
            [("connect", "0"), ("connect", "1"), ("route", "eth0"), ("route", "eth1")],
        )

    def test_connection_failure_rolls_back_only_new_endpoint(self):
        self.fail_connect = "1"
        with self.assertRaisesRegex(RuntimeError, "connect failed"):
            rdma.attach(CID)
        self.assertEqual(self.events, [("connect", "0"), ("disconnect", "0")])
        self.assertFalse(self.connected)

    def test_rerun_does_not_reconnect_endpoints_after_routes_exist(self):
        self.connected = {"0", "1"}
        rdma.attach(CID)
        self.assertFalse(any(event[0] == "connect" for event in self.events))

    def test_default_route_change_fails_and_preserves_existing_endpoint(self):
        self.connected = {"0"}
        with patch.object(rdma, "defaults", side_effect=[[["old"]], [["changed"]]]):
            with self.assertRaisesRegex(RuntimeError, "default route"):
                rdma.attach(CID)
        self.assertEqual(self.connected, {"0"})
        self.assertIn(("disconnect", "1"), self.events)
        self.assertEqual(len([event for event in self.events if event[0] == "command"]), 2)

    def test_host_namespace_is_rejected_before_mutation(self):
        for mode in ("host", "none", "container:another-job"):
            with self.subTest(mode=mode):
                self.mode = mode
                with self.assertRaisesRegex(ValueError, "isolated bridge"):
                    rdma.attach(CID)
                self.assertFalse(self.events)

    def test_unconfigured_host_does_not_require_new_docker_or_ip(self):
        self.networks = []
        self.version = "1.41"
        with patch.object(
            rdma.shutil,
            "which",
            side_effect=AssertionError("must not install dependencies"),
        ):
            rdma.attach(CID)
        self.assertFalse(self.events)

    def test_old_api_rejected_on_configured_host(self):
        self.version = "1.47"
        with self.assertRaisesRegex(RuntimeError, "1.48"):
            rdma.attach(CID)
        self.assertFalse(self.events)


class RouteTests(unittest.TestCase):
    def test_exact_existing_route_is_not_readded(self):
        route = {"dst": "2001:db8::/32", "gateway": "fe80::1", "metric": 1010}
        with (
            patch.object(rdma, "ip_json", return_value=[dict(route)]),
            patch.object(rdma, "command") as run,
        ):
            self.assertIsNone(rdma.add_route(route, "eth1"))
            run.assert_not_called()

    def test_default_route_configuration_is_rejected(self):
        config = network(0)
        values = json.loads(config["Labels"][rdma.LABEL])
        values["routes"][0]["dst"] = "::/0"
        config["Labels"][rdma.LABEL] = json.dumps(values)
        with self.assertRaisesRegex(ValueError, "default route"):
            rdma.validate_network(config)

    def test_non_private_ipvlan_configuration_is_rejected(self):
        config = copy.deepcopy(network(0))
        config["Options"]["ipvlan_flag"] = "bridge"
        with self.assertRaisesRegex(ValueError, "Unsafe"):
            rdma.validate_network(config)


if __name__ == "__main__":
    unittest.main()
