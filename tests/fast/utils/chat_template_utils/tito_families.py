"""Real tokenizers for the registered TITO families, shared by the family-wide tests."""

from functools import cache

from huggingface_hub import snapshot_download

from miles.utils.chat_template_utils.tito_tokenizer import (
    TITOTokenizer,
    TITOTokenizerType,
    get_tito_tokenizer,
    resolve_fixed_chat_template,
)
from miles.utils.processing_utils import load_tokenizer

# One checkpoint per registered family class; the renderer is the class's.
FAMILY_CHECKPOINTS = {
    TITOTokenizerType.QWEN3: "Qwen/Qwen3-0.6B",
    TITOTokenizerType.QWEN35: "Qwen/Qwen3.5-0.8B",
    TITOTokenizerType.QWEN36: "Qwen/Qwen3.6-35B-A3B",
    TITOTokenizerType.QWEN38_SMALL: "Qwen/Qwen3.8-27B",
    TITOTokenizerType.QWENNEXT: "Qwen/Qwen3-4B-Thinking-2507",
    TITOTokenizerType.GLM47: "zai-org/GLM-4.7-Flash",
    TITOTokenizerType.GLM53: "zai-org/GLM-5.3",
    TITOTokenizerType.NEMOTRON3: "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
    TITOTokenizerType.KIMI25: "moonshotai/Kimi-K2.5",
    TITOTokenizerType.KIMI26: "moonshotai/Kimi-K2.6",
    TITOTokenizerType.MINIMAX_M25: "MiniMaxAI/MiniMax-M2.5",
    TITOTokenizerType.MINIMAX_M27: "MiniMaxAI/MiniMax-M2.7",
    TITOTokenizerType.DEEPSEEKV32: "deepseek-ai/DeepSeek-V3.2",
    TITOTokenizerType.DEEPSEEKV4: "deepseek-ai/DeepSeek-V4-Flash",
    TITOTokenizerType.DEEPSEEKV41: "deepseek-ai/DeepSeek-V4.1-Flash",
    TITOTokenizerType.INKLING: "thinkingmachines/Inkling-Small",
    TITOTokenizerType.MUSE_SPARK: "RadixArk/Muse-Spark-Open-last4-partial",
}
_TOKENIZER_FILES = [
    "config.json",
    "tokenizer*",
    "special_tokens_map.json",
    "chat_template*",
    "*.py",
    "*.model",
    "*.tiktoken",
    "tiktoken/*",
]


@cache
def load_family_tito(family: TITOTokenizerType) -> TITOTokenizer:
    path = snapshot_download(FAMILY_CHECKPOINTS[family], allow_patterns=_TOKENIZER_FILES)
    chat_template_path, _ = resolve_fixed_chat_template(family)
    tokenizer = load_tokenizer(path, chat_template_path=chat_template_path, trust_remote_code=True)
    return get_tito_tokenizer(tokenizer, tokenizer_type=family)
