import dataclasses
import logging
from urllib.parse import quote

import requests
import sglang_router
from packaging.version import parse

logger = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True)
class SGLangRouterApiClient:
    router_url: str

    def add_worker(
        self,
        worker_url: str,
        worker_type: str,
        use_legacy_api: bool,
        bootstrap_port: int | None = None,
    ):
        if use_legacy_api:
            assert worker_type == "regular", "pd disaggregation is not supported in old router or miles router."
            response = requests.post(f"{self.router_url}/add_worker?url={worker_url}")
        else:
            payload = {
                "url": worker_url,
                "worker_type": worker_type,
            }
            if worker_type == "prefill":
                payload["bootstrap_port"] = bootstrap_port
            response = requests.post(
                f"{self.router_url}/workers",
                json=payload,
            )
        response.raise_for_status()

    def remove_worker(self, worker_url: str, use_legacy_api: bool):
        response = None
        if use_legacy_api:
            response = requests.post(f"{self.router_url}/remove_worker?url={worker_url}")
        elif parse(sglang_router.__version__) < parse("0.3.0"):
            quoted_worker_url = quote(worker_url, safe="")
            response = requests.delete(f"{self.router_url}/workers/{quoted_worker_url}")
        else:
            try:
                all_workers = requests.get(f"{self.router_url}/workers").json()["workers"]
                for worker in all_workers:
                    if worker["url"] == worker_url:
                        worker_id = worker["id"]
                        response = requests.delete(f"{self.router_url}/workers/{worker_id}")
                        break
                else:
                    logger.warning(f"Worker {worker_url} not found in router during shutdown.")
            except Exception as e:
                logger.warning(f"Failed to fetch workers list or remove worker: {e}")

        if response is not None:
            response.raise_for_status()
