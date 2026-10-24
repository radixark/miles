from miles.backends.sglang_utils.sglang_config import SglangConfig
from miles.utils.args.configs.sglang_client import SglangClientConfig


class SglangFieldsConfig(SglangClientConfig):
    sglang: SglangConfig
