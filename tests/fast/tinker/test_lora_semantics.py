from argparse import Namespace

from miles.backends.megatron_utils.multi_lora_utils import _lora_a_init_method


def test_tinker_service_uses_peft_lora_initialization(monkeypatch) -> None:
    monkeypatch.setenv("MILES_TINKER_API", "1")
    args = Namespace(lora_A_init_method="xavier")

    assert _lora_a_init_method(args) == "kaiming"
