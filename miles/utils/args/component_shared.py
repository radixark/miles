import argparse

from miles.backends.sglang_utils.sglang_config import SglangConfig
from miles.utils.args.configs.sglang_client import SglangClientConfig


class SglangFieldsConfig(SglangClientConfig):
    sglang: SglangConfig

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> None:
        SglangConfig.add_arguments(parser)
        # required whenever expert projections are LoRA targets, inert otherwise
        # (sglang's own default is False)
        parser.set_defaults(sglang_lora_use_virtual_experts=True)
