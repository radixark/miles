from typing import Any

from miles.utils.processing_utils import encode_image_for_rollout_engine


def build_rollout_engine_multimodal_payload(
    multimodal_inputs: dict[str, Any] | None,
    rollout_video_inputs: list[dict[str, Any]] | None,
) -> dict[str, list[str]]:
    multimodal_inputs = multimodal_inputs or {}
    unsupported_keys = multimodal_inputs.keys() - {"images", "videos"}
    if unsupported_keys:
        raise ValueError(f"Unsupported multimodal input keys: {sorted(unsupported_keys)}")

    payload = {}
    if image_data := multimodal_inputs.get("images"):
        payload["image_data"] = [encode_image_for_rollout_engine(image) for image in image_data]

    processed_videos = multimodal_inputs.get("videos")
    processed_video_count = len(processed_videos) if processed_videos is not None else 0
    rollout_video_count = len(rollout_video_inputs) if rollout_video_inputs is not None else 0
    if processed_video_count != rollout_video_count:
        raise ValueError(
            "Video processor inputs and rollout inputs must have the same length: "
            f"processed={processed_video_count}, rollout={rollout_video_count}"
        )
    if rollout_video_inputs:
        video_data = []
        for video_input in rollout_video_inputs:
            unrepresented_fields = set(video_input) - {"type", "video"}
            if unrepresented_fields:
                raise NotImplementedError(
                    "The rollout-engine request cannot propagate these per-video fields: "
                    f"{sorted(unrepresented_fields)}"
                )
            source = video_input["video"]
            if not isinstance(source, str):
                raise TypeError("The rollout-engine request requires each video source to be a string")
            video_data.append(source)
        payload["video_data"] = video_data

    return payload


def has_multimodal_inputs(
    multimodal_inputs: dict[str, Any] | None,
    rollout_video_inputs: list[dict[str, Any]] | None,
) -> bool:
    processor_media = ((multimodal_inputs or {}).get(key) for key in ("images", "videos"))
    return any(value is not None and len(value) > 0 for value in processor_media) or bool(rollout_video_inputs)


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
