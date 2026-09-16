"""The two fakes the collector tests add on top of ``harness.py``: a character tokenizer and a sampler snapshot on disk.

Everything else is reused: ``make_service`` / ``FakeBackend.sample`` (a dict in ``fail_on["sample"]`` doubles as the canned
result), ``write_checkpoint_dir`` (the real checkpoint writer) and ``resolve_checkpoint_dir`` (the real layout).
"""

import hashlib
from pathlib import Path

from miles.backends.training_utils.checkpoint_io import write_checkpoint_dir
from miles.tinker.core.utils import resolve_checkpoint_dir

TENANT = "tml-owner"
OTHER_TENANT = "tml-other"
SAMPLER = "tinker://m1/sampler_weights/v0"

ROLE_IDS = {"system": 1, "user": 2, "assistant": 3, "tool": 4}
THINK_ON = 7
GEN = 9


class FakeTokenizer:
    """Deterministic ids: one role marker per message, one id per character, a marker when thinking is on, then the generation prompt."""

    def apply_chat_template(self, messages, add_generation_prompt=True, tokenize=True, **kwargs):
        ids = []
        for message in messages:
            ids.append(ROLE_IDS[message["role"]])
            ids.extend(ord(char) for char in message.get("content") or "")
        if kwargs.get("enable_thinking"):
            ids.append(THINK_ON)
        if add_generation_prompt:
            ids.append(GEN)
        return ids

    def decode(self, ids, skip_special_tokens=True):
        return "".join(chr(token) for token in ids if token >= 32)


def write_sampler(checkpoint_root: Path, base_model: str, tenant: str = TENANT) -> str:
    """Lay down m1/sampler_weights/v0/META.json owned by `tenant`, the way save_weights_for_sampler does (build_checkpoint_metadata + write_checkpoint_dir)."""
    path = resolve_checkpoint_dir(str(checkpoint_root), "m1", "sampler_weights", "v0")
    metadata = {
        "tenant_digest": hashlib.sha256(tenant.encode()).hexdigest(),
        "base_model": base_model,
        "lora_rank": 8,
        "lora_alpha": 16.0,
        "train_attn": True,
        "train_mlp": True,
        "train_unembed": False,
    }
    write_checkpoint_dir(path, lambda _: None, metadata=metadata)
    return path
