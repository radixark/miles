from typing import Any

from miles.utils.processing_utils import encode_image_for_rollout_engine


def build_rollout_engine_multimodal_payload(
    multimodal_inputs: dict[str, Any] | None,
    rollout_video_sources: list[str] | None,
) -> dict[str, list[str]]:
    multimodal_inputs = multimodal_inputs or {}
    unsupported_keys = multimodal_inputs.keys() - {"images", "videos"}
    if unsupported_keys:
        raise ValueError(f"Unsupported multimodal input keys: {sorted(unsupported_keys)}")

    payload = {}
    if image_data := multimodal_inputs.get("images"):
        payload["image_data"] = [encode_image_for_rollout_engine(image) for image in image_data]

    processed_videos = multimodal_inputs.get("videos")
    if rollout_video_sources is not None and any(not isinstance(source, str) for source in rollout_video_sources):
        raise TypeError("Rollout video sources must be paths, URLs, or data URIs")
    processed_video_count = len(processed_videos) if processed_videos is not None else 0
    rollout_video_count = len(rollout_video_sources) if rollout_video_sources is not None else 0
    if processed_video_count != rollout_video_count:
        raise ValueError(
            "Video processor inputs and rollout sources must have the same length: "
            f"processed={processed_video_count}, rollout={rollout_video_count}"
        )
    if rollout_video_sources:
        payload["video_data"] = list(rollout_video_sources)

    return payload


def has_multimodal_inputs(
    multimodal_inputs: dict[str, Any] | None,
    rollout_video_sources: list[str] | None,
) -> bool:
    processor_media = ((multimodal_inputs or {}).get(key) for key in ("images", "videos"))
    return any(value is not None and len(value) > 0 for value in processor_media) or bool(rollout_video_sources)


def build_rollout_input_ids(
    input_ids: list[int],
    *,
    processor_prompt_ids: list[int],
    rollout_prompt_ids: list[int] | None,
) -> list[int]:
    input_ids = list(input_ids)
    processor_prompt_ids = list(processor_prompt_ids)
    if rollout_prompt_ids is None:
        return input_ids

    if input_ids[: len(processor_prompt_ids)] != processor_prompt_ids:
        raise ValueError("Cannot build rollout_input_ids: input IDs do not start with the processed prompt IDs")

    return list(rollout_prompt_ids) + input_ids[len(processor_prompt_ids) :]
