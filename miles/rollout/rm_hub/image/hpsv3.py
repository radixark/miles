"""HPSv3 (MizzenAI/HPSv3) of an image: a Qwen2-VL-7B image preference model whose ranknet head reads one special
token; the score is the head's mean (mu), not its sigma. A reward pool scorer: images arrive as encoded bytes (PNG).

Ported from the repo's `inference.py` (`HPSv3RewardInferencer`), which pins transformers 4.45, onto Miles'
transformers. The forward keeps theirs: plain 1D positions in place of Qwen2-VL's 3D rope ones, the head's weights in
the model's dtype but run under float32 autocast, the attention kernel their `create_model_and_processor` picks, and
the image resized by their own `fetch_image`.
"""

import io
import logging
import math

import torch
from PIL import Image, UnidentifiedImageError
from safetensors.torch import load_file
from transformers import AutoProcessor
from transformers.models.qwen2_vl.modeling_qwen2_vl import Qwen2VLModel

from miles.utils.hf_utils.config import load_hf_config

logger = logging.getLogger(__name__)

_SPECIAL_TOKEN = "<|Reward|>"
# HPSv3_7B.yaml's min_pixels and max_pixels, both 256 * 28 * 28.
_PIXELS = 256 * 28 * 28
_SIZE_FACTOR = 28

# HPSv3's INSTRUCTION followed by prompt_with_special_token, verbatim; \x20 keeps the trailing spaces it was trained on.
_PROMPT = """
You are tasked with evaluating a generated image based on Visual Quality and Text Alignment and give a overall score to estimate the human preference. Please provide a rating from 0 to 10, with 0 being the worst and 10 being the best.\x20

**Visual Quality:**\x20\x20
Evaluate the overall visual quality of the image. The following sub-dimensions should be considered:
- **Reasonableness:** The image should not contain any significant biological or logical errors, such as abnormal body structures or nonsensical environmental setups.
- **Clarity:** Evaluate the sharpness and visibility of the image. The image should be clear and easy to interpret, with no blurring or indistinct areas.
- **Detail Richness:** Consider the level of detail in textures, materials, lighting, and other visual elements (e.g., hair, clothing, shadows).
- **Aesthetic and Creativity:** Assess the artistic aspects of the image, including the color scheme, composition, atmosphere, depth of field, and the overall creative appeal. The scene should convey a sense of harmony and balance.
- **Safety:** The image should not contain harmful or inappropriate content, such as political, violent, or adult material. If such content is present, the image quality and satisfaction score should be the lowest possible.\x20

**Text Alignment:**\x20\x20
Assess how well the image matches the textual prompt across the following sub-dimensions:
- **Subject Relevance** Evaluate how accurately the subject(s) in the image (e.g., person, animal, object) align with the textual description. The subject should match the description in terms of number, appearance, and behavior.
- **Style Relevance:** If the prompt specifies a particular artistic or stylistic style, evaluate how well the image adheres to this style.
- **Contextual Consistency**: Assess whether the background, setting, and surrounding elements in the image logically fit the scenario described in the prompt. The environment should support and enhance the subject without contradictions.
- **Attribute Fidelity**: Check if specific attributes mentioned in the prompt (e.g., colors, clothing, accessories, expressions, actions) are faithfully represented in the image. Minor deviations may be acceptable, but critical attributes should be preserved.
- **Semantic Coherence**: Evaluate whether the overall meaning and intent of the prompt are captured in the image. The generated content should not introduce elements that conflict with or distort the original description.
Textual prompt - {text_prompt}



Please provide the overall ratings of this image: <|Reward|>

END
"""


class HPSv3Scorer:
    def __init__(
        self,
        *,
        device: torch.device,
        checkpoint_path: str,
        base_model_path: str,
        dtype: str = "bfloat16",
        attn_implementation: str | None = None,
    ) -> None:
        self.device = device
        # Slow processor, as transformers 4.45 had: the fast one normalises on torch and rounds differently.
        self.processor = AutoProcessor.from_pretrained(base_model_path, padding_side="right", use_fast=False)
        self.processor.tokenizer.add_special_tokens({"additional_special_tokens": [_SPECIAL_TOKEN]})
        self.special_token_id = self.processor.tokenizer.convert_tokens_to_ids(_SPECIAL_TOKEN)

        state = _renamed_state_dict(load_file(checkpoint_path))
        model_config = load_hf_config(base_model_path)
        model_config.text_config.vocab_size = state["language_model.embed_tokens.weight"].shape[0]
        torch_dtype = getattr(torch, dtype)  # config-access-exempt: the torch dtype the dtype argument names
        # Built on the device: a 7B random init on the CPU takes minutes before load_state_dict overwrites it. Built in
        # float32 and cast after loading, as the reference does, so its rope buffers are rounded the same way.
        with torch.device(device):
            self.model = Qwen2VLModel._from_config(
                model_config,
                dtype=torch.float32,
                attn_implementation=attn_implementation or _reference_attention(torch_dtype),
            )
        head = {name[len("rm_head.") :]: state.pop(name) for name in list(state) if name.startswith("rm_head.")}
        self.model.load_state_dict(state, strict=True)
        # HPSv3's ranknet head without rm_head_kwargs; its dropout is inert in eval.
        self.rm_head = torch.nn.Sequential(
            torch.nn.Linear(head["0.weight"].shape[1], head["0.weight"].shape[0]),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.05),
            torch.nn.Linear(head["3.weight"].shape[1], head["3.weight"].shape[0]),
            torch.nn.ReLU(),
            torch.nn.Linear(head["5.weight"].shape[1], head["5.weight"].shape[0]),
        )
        self.rm_head.load_state_dict(head, strict=True)
        self.model.eval().to(torch_dtype)
        _round_vision_rope_as_the_reference(self.model.visual.rotary_pos_emb)
        # The reference casts the whole model, head included, to its dtype before loading the checkpoint.
        self.rm_head.eval().to(device=device, dtype=torch_dtype)

    def score_batch(self, images: list[bytes], prompts: list[str]) -> list[float | None]:
        """Each image's HPSv3 mu against its prompt; None for bytes that do not decode as an image."""
        decoded = [_decode(image) for image in images]
        valid = [index for index, image in enumerate(decoded) if image is not None]
        scores: list[float | None] = [None] * len(images)
        if valid:
            values = self.score_images([decoded[i] for i in valid], [prompts[i] for i in valid])
            for index, value in zip(valid, values, strict=True):
                scores[index] = value
        return scores

    @torch.no_grad()
    def score_images(self, images: list[Image.Image], prompts: list[str]) -> list[float]:
        images = [_fetch_image(image) for image in images]
        texts = [
            self.processor.apply_chat_template(
                [
                    {
                        "role": "user",
                        "content": [{"type": "image"}, {"type": "text", "text": _PROMPT.format(text_prompt=p)}],
                    }
                ],
                tokenize=False,
                add_generation_prompt=True,
            )
            for p in prompts
        ]
        inputs = self.processor(text=texts, images=images, padding=True, return_tensors="pt").to(self.device)
        input_ids = inputs["input_ids"]
        embeds = self.model.get_input_embeddings()(input_ids)
        image_embeds = self.model.get_image_features(inputs["pixel_values"], inputs["image_grid_thw"])
        image_embeds = torch.cat(image_embeds.pooler_output).to(embeds.dtype)
        image_mask = (input_ids == self.model.config.image_token_id).unsqueeze(-1).expand_as(embeds)
        embeds = embeds.masked_scatter(image_mask, image_embeds)
        # No position_ids: transformers 4.45, which HPSv3 trained on, gave its text model 1D positions here.
        hidden = self.model.language_model(inputs_embeds=embeds, attention_mask=inputs["attention_mask"])
        with torch.autocast(device_type=self.device.type, dtype=torch.float32):
            logits = self.rm_head(hidden.last_hidden_state)
        return logits[input_ids == self.special_token_id][:, 0].float().tolist()


def _round_vision_rope_as_the_reference(rope: torch.nn.Module) -> None:
    """transformers 4.45 computed the vision rope's angles and their cos and sin in the model's dtype; 5.x in float32."""

    def forward(x: torch.Tensor, position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        freqs = position_ids[..., None].to(rope.inv_freq.dtype) * rope.inv_freq
        return rope.recomposition_frequencies(freqs.cos()), rope.recomposition_frequencies(freqs.sin())

    rope.forward = forward


def _reference_attention(dtype: torch.dtype) -> str:
    """flash_attention_2 when flash_attn imports and the dtype is half, else sdpa, as the reference picks."""
    try:
        import flash_attn  # noqa: F401
    except ImportError:
        return "sdpa"
    return "flash_attention_2" if dtype in (torch.bfloat16, torch.float16) else "sdpa"


def _decode(image: bytes) -> Image.Image | None:
    try:
        decoded = Image.open(io.BytesIO(image))
        decoded.load()
    except (UnidentifiedImageError, OSError, ValueError) as e:
        logger.warning(f"HPSv3 got bytes that are not an image: {e!r}")
        return None
    return decoded


def _fetch_image(image: Image.Image) -> Image.Image:
    """The reference's `fetch_image` for a PIL image at HPSv3's fixed pixel count."""
    image = image.convert("RGB")
    height, width = _smart_resize(image.height, image.width, min_pixels=_PIXELS, max_pixels=_PIXELS)
    return image.resize((width, height), Image.BICUBIC)


def _smart_resize(height: int, width: int, *, min_pixels: int, max_pixels: int) -> tuple[int, int]:
    """Qwen2-VL's smart_resize as the reference's `dataset/utils.py` computes it, float order included."""
    factor = _SIZE_FACTOR
    h_bar = max(factor, round(height / factor) * factor)
    w_bar = max(factor, round(width / factor) * factor)
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = math.floor(height / beta / factor) * factor
        w_bar = math.floor(width / beta / factor) * factor
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar


def _renamed_state_dict(checkpoint: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """The transformers 4.45 checkpoint as 5.x Qwen2VLModel names (`model.` to `language_model.`), plus `rm_head.*`,
    with the unused `lm_head` dropped."""
    state = {}
    for name, tensor in checkpoint.items():
        if name == "lm_head.weight":
            continue
        if name.startswith("model."):
            name = "language_model." + name[len("model.") :]
        state[name] = tensor
    return state
