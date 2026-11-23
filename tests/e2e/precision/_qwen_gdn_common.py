"""Shared setup of the Qwen GDN TOP gates: a few-layer slice of a public checkpoint and the GRPO arguments."""

import json
import re
import shutil
from pathlib import Path

from safetensors import safe_open
from safetensors.torch import save_file

from miles.true_on_policy.launch import TOP_RECIPE_TRAIN_ARGS

# Engines and trainer read the same override; 512 splits decode keys at lengths a 1,024-token response reaches.
DECODE_KV_SPLIT = "512"
# Serving cuts every decode into context / split key pieces (a CUDA-graph shape) and FA's combine merges at most 256.
CONTEXT_LENGTH = 8192


def slice_checkpoint(src: str, dst: str, num_layers: int, *, keep_mtp: bool = False) -> None:
    """Write ``dst``: ``src`` with its first ``num_layers`` decoder layers and, unless ``keep_mtp``, no MTP head.
    Qwen3.5/3.6/3.8 hybrids put full attention on every fourth layer, so four layers keep three GDN layers and one
    full-attention layer."""
    src_dir, dst_dir = Path(src), Path(dst)
    if (dst_dir / "config.json").exists():
        return
    dst_dir.mkdir(parents=True, exist_ok=True)
    layer_re = re.compile(r"\.layers\.(\d+)\.")

    def keep(name: str) -> bool:
        match = layer_re.search(name)
        if name.startswith("mtp.") or ".mtp." in name:
            return keep_mtp
        return match is None or int(match.group(1)) < num_layers

    by_file: dict[str, list[str]] = {}
    for name, file in json.loads((src_dir / "model.safetensors.index.json").read_text())["weight_map"].items():
        if keep(name):
            by_file.setdefault(file, []).append(name)
    weight_map, total = {}, 0
    for i, (file, names) in enumerate(sorted(by_file.items())):
        out = f"model-{i + 1:05d}-of-{len(by_file):05d}.safetensors"
        with safe_open(src_dir / file, framework="pt") as f:
            tensors = {name: f.get_tensor(name) for name in names}
        save_file(tensors, dst_dir / out, metadata={"format": "pt"})
        weight_map.update(dict.fromkeys(tensors, out))
        total += sum(t.numel() * t.element_size() for t in tensors.values())
    index = {"metadata": {"total_size": total}, "weight_map": weight_map}
    (dst_dir / "model.safetensors.index.json").write_text(json.dumps(index, indent=2))
    config = json.loads((src_dir / "config.json").read_text())
    text = config.get("text_config", config)
    text["num_hidden_layers"] = num_layers
    if "layer_types" in text:
        text["layer_types"] = text["layer_types"][:num_layers]
    if not keep_mtp:
        text["mtp_num_hidden_layers"] = 0
    for path in src_dir.iterdir():
        if path.is_file() and path.suffix != ".safetensors" and path.name != "model.safetensors.index.json":
            shutil.copy2(path, dst_dir / path.name)
    (dst_dir / "config.json").write_text(json.dumps(config, indent=2))


def write_image_prompts(data_dir: str, *, long: bool = False) -> str:
    """Write ``data_dir``/prompts.jsonl, prompts over drawn images, and return its path.

    The images are deterministic drawings at eight sizes and aspect ratios (36 to 1,496 placeholder tokens), so the
    gate needs no download. Two prompts carry two images, one image opens its prompt and another closes it, and
    one prompt has no image: a text sample trains beside image samples and takes mRoPE positions like them.

    ``long``: every image prompt opens with the two largest drawings (2,264 placeholder tokens), so its image
    tokens alone pass 2,048 keys, where Qwen3.8-Flash-Next's QSA starts to select among them."""
    root = Path(data_dir)
    paths = _draw_images(root)
    prompts = [
        ("<image>How many straight lines cross this drawing? Explain each step.", [0]),
        ("Look at the drawing, then list every colour you can see in it. <image>", [1]),
        ("<image><image>Which of the two drawings holds more ellipses, and by how many?", [2, 3]),
        ("Solve x^2 - 5x + 6 = 0 and explain each step.", []),
        ("<image>Estimate the area that the rectangles cover, as a share of the drawing.", [4]),
        ("Compare the first drawing <image> with the second <image> and name three differences.", [5, 6]),
        ("<image>Describe the drawing from its top left corner to its bottom right corner.", [7]),
        ("What is drawn here? <image> Answer in one paragraph.", [0]),
    ]
    if long:
        prompts = [
            (
                f"Two reference drawings: <image><image> Now this task. {prompt}" if images else prompt,
                [7, 4] * bool(images) + images,
            )
            for prompt, images in prompts
        ]
    out = root / "prompts.jsonl"
    with out.open("w") as f:
        for index, (prompt, images) in enumerate(prompts):
            row = {"prompt": prompt, "images": [paths[i] for i in images], "label": str(index)}
            f.write(json.dumps(row) + "\n")
    return str(out)


def _drawing(index: int, width: int, height: int, shift: int = 0):
    """A deterministic drawing; ``shift`` moves its shapes, for the frames of a video."""
    from PIL import Image, ImageDraw

    image = Image.new("RGB", (width, height), (255 - 20 * index, 240, 200 + 5 * index))
    draw = ImageDraw.Draw(image)
    for step in range(index + 3):
        x, y = (37 * step + 11 * index + shift) % width, (53 * step + 7 * index + 2 * shift) % height
        color = (40 * step % 256, 90 * index % 256, 255 - 30 * step % 256)
        draw.line([(x, 0), (width - 1 - x, height - 1)], fill=color, width=1 + step % 4)
        draw.ellipse([x // 2, y // 2, x // 2 + width // 4, y // 2 + height // 4], outline=color, width=2)
        draw.rectangle([x // 3, y // 3, x // 3 + 9 * (step + 1), y // 3 + 5 * (step + 1)], fill=color)
    return image


def _draw_images(root: Path) -> list[str]:
    """Eight drawings at sizes from 96x96 to 1400x1092, in ``root``/images."""
    (root / "images").mkdir(parents=True, exist_ok=True)
    sizes = [(448, 336), (336, 448), (1120, 96), (96, 96), (1024, 768), (640, 640), (224, 896), (1400, 1092)]
    paths = []
    for index, (width, height) in enumerate(sizes):
        path = root / "images" / f"drawing-{index}-{width}x{height}.png"
        _drawing(index, width, height).save(path)
        paths.append(str(path))
    return paths


def _write_videos(root: Path) -> list[str]:
    """Three MPEG-4 clips of moving drawings in ``root``/videos: lengths 2 to 6 seconds at 8 fps, so the sampled
    frames form 2 to 6 temporal patch groups, each its own attention sequence in the tower."""
    import cv2
    import numpy as np

    (root / "videos").mkdir(parents=True, exist_ok=True)
    clips = [(0, 320, 224, 2.0), (5, 256, 256, 4.0), (2, 448, 160, 6.0)]
    paths = []
    for index, width, height, seconds in clips:
        path = root / "videos" / f"clip-{index}-{width}x{height}-{seconds:g}s.mp4"
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 8.0, (width, height))
        for frame in range(int(8 * seconds)):
            image = np.asarray(_drawing(index, width, height, shift=9 * frame))
            writer.write(image[:, :, ::-1].copy())
        writer.release()
        paths.append(str(path))
    return paths


def write_video_prompts(data_dir: str, *, long: bool = False) -> str:
    """Write ``data_dir``/prompts.jsonl, prompts over drawn clips (and drawings), and return its path.

    One clip alone, two clips, a drawing and a clip in one prompt, and one text-only prompt, so video rows, image
    rows and text share batches. ``long``: every media prompt opens with the two largest drawings, as in
    ``write_image_prompts``."""
    root = Path(data_dir)
    images, videos = _draw_images(root), _write_videos(root)
    prompts = [
        ("<video>What moves in this clip, and in which direction?", [], [0]),
        ("Here are two clips: <video> and <video> Which one is longer, and what differs?", [], [1, 2]),
        ("<image> A drawing, then a clip: <video> Does the clip animate the drawing?", [0], [0]),
        ("Solve x^2 - 7x + 12 = 0 and explain each step.", [], []),
        ("<video>Describe the clip frame by frame.", [], [2]),
        ("Count the ellipses in the drawing <image> and in the clip <video>.", [5], [1]),
        ("<video>Name every colour that appears in the clip.", [], [1]),
        ("What is drawn here? <image> And what happens here? <video>", [7], [0]),
    ]
    if long:
        prompts = [
            (
                f"Two reference drawings: <image><image> Now this task. {prompt}" if image or video else prompt,
                [7, 4] * bool(image or video) + image,
                video,
            )
            for prompt, image, video in prompts
        ]
    out = root / "prompts.jsonl"
    with out.open("w") as f:
        for index, (prompt, image, video) in enumerate(prompts):
            row = {"prompt": prompt, "images": [images[i] for i in image], "videos": [videos[i] for i in video]}
            f.write(json.dumps(row | {"label": str(index)}) + "\n")
    return str(out)


def vision_args(*, data_dir: str, video: bool = False) -> str:
    """Images (and with ``video``, video) through SGLang's token-space processor: the engines expand the
    placeholders and preprocess on the CPU (as every TOP engine does), and each trainer rank rebuilds the same
    pixels, placement and mRoPE positions."""
    keys = '{"image": "images", "video": "videos"}' if video else '{"image": "images"}'
    return (
        f"--prompt-data {data_dir}/prompts.jsonl "
        f"--multimodal-keys '{keys}' "
        f"--shared-media-dir {data_dir}/media "
        "--multimodal-token-space-processing "
        "--sglang-enable-token-space-processor "
    )


def grpo_args(*, model_dir: str, prompt_data: str = "/root/datasets/dapo-math-17k/dapo-math-17k.jsonl") -> str:
    """Two GRPO steps of 4 prompts x 2 samples from the HF checkpoint (GDN TOP loads HF weights directly), with the
    trainer run as TOP recipes run it. ``prompt_data`` None leaves the dataset to the caller."""
    return (
        f"--hf-checkpoint {model_dir} --load {model_dir} --ref-load {model_dir} "
        f"{f'--prompt-data {prompt_data} ' if prompt_data else ''}"
        "--input-key prompt "
        "--label-key label "
        "--apply-chat-template "
        "--rollout-shuffle "
        "--num-rollout 2 "
        "--rollout-batch-size 4 "
        "--n-samples-per-prompt 2 "
        "--rollout-max-response-len 1024 "
        "--rollout-temperature 0.8 "
        "--global-batch-size 8 "
        "--advantage-estimator grpo "
        "--kl-loss-coef 0.00 "
        "--kl-loss-type k1 "
        "--kl-coef 0.00 "
        "--entropy-coef 0.00 "
        "--eps-clip 0.2 "
        "--optimizer adam "
        "--lr 1e-6 "
        "--lr-decay-style constant "
        "--weight-decay 0.1 "
        "--adam-beta1 0.9 "
        "--adam-beta2 0.98 "
        "--attention-dropout 0.0 "
        "--hidden-dropout 0.0 "
        "--use-dynamic-batch-size "
        "--max-tokens-per-gpu 16384 "
        # As TOP recipes run: rollout log-probs as the old policy, full recompute (under CP it reuses the received
        # GDN state).
        f"{' '.join(TOP_RECIPE_TRAIN_ARGS)} "
        "--sglang-mem-fraction-static 0.7 "
        # GDN TOP engines size each running request's open-chunk row cache by this bound, and refuse to start without
        # it; 128 covers every decode CUDA graph the gates capture.
        "--sglang-max-running-requests 128 "
        f"--sglang-context-length {CONTEXT_LENGTH} "
        # The checkpoint's vision tower stays on the engines; the text-only trainer never syncs it.
        "--check-weight-update-skip-list visual. "
        "--ci-test "
    )
