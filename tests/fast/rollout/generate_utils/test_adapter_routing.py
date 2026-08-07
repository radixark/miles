"""Registration-scoped adapter routing: serving identity stamped onto engine
requests, plus the live-upsert serving_version override."""

from types import SimpleNamespace

from miles.rollout.generate_utils.generate_endpoint_utils import apply_adapter_routing
from miles.utils.lora import LORA_ADAPTER_NAME
from miles.utils.multi_lora import cache_extra_key, rid_prefix, serving_lora_name
from miles.utils.types import AdapterRef, Sample


def make_adapter_sample(version: int = 3) -> Sample:
    sample = Sample(prompt="p")
    sample.adapter = AdapterRef(name="a", registration_id="reg1", serving_version=version, slot=0)
    return sample


def base_args(**overrides) -> SimpleNamespace:
    defaults = dict(multi_lora=True, lora_rank=16, lora_adapter_path=None, lora_train_only=False)
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def test_routing_is_registration_scoped():
    payload: dict = {}
    apply_adapter_routing(base_args(), payload, make_adapter_sample())
    assert payload["lora_path"] == serving_lora_name("a", "reg1")
    # rid lives in the registration's abort namespace: DELETE aborts by this prefix.
    assert payload["rid"].startswith(rid_prefix("a", "reg1"))
    assert payload["extra_key"] == cache_extra_key("a", "reg1", 3)

    # Adapter-less samples fall back to the fixed single-LoRA name — and only
    # when the rollout side participates (--lora-train-only opts out).
    payload = {}
    apply_adapter_routing(base_args(lora_train_only=True), payload, Sample(prompt="p"))
    assert "lora_path" not in payload
    payload = {}
    apply_adapter_routing(base_args(), payload, Sample(prompt="p"))
    assert payload["lora_path"] == LORA_ADAPTER_NAME


def test_serving_version_override_renames_the_kv_namespace():
    # The live-upsert path routes under the adapter's CURRENT published
    # revision, not the one stamped at sample-submission time.
    payload: dict = {}
    apply_adapter_routing(base_args(), payload, make_adapter_sample(version=3), serving_version=5)
    assert payload["extra_key"] == cache_extra_key("a", "reg1", 5)
    assert payload["lora_path"] == serving_lora_name("a", "reg1")


def test_two_registrations_of_one_name_never_share_identity():
    first, second = {}, {}
    sample = make_adapter_sample()
    apply_adapter_routing(base_args(), first, sample)
    sample.adapter = AdapterRef(name="a", registration_id="reg2", serving_version=1, slot=0)
    apply_adapter_routing(base_args(), second, sample)
    assert first["lora_path"] != second["lora_path"]
    assert first["extra_key"] != second["extra_key"]
    assert first["rid"] != second["rid"]
