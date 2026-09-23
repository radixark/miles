import json

from miles.utils.data import Dataset


class _RecordingProcessor:
    def __init__(self) -> None:
        self.prompts = []

    def extract_media(self, prompt):
        self.prompts.append(prompt)
        return {"images": ["image"], "videos": []}


def test_mixed_dataset_uses_processor_only_for_structured_prompts(tmp_path) -> None:
    text_prompt = "Fix the bug"
    multimodal_prompt = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": "image.png"},
                {"type": "text", "text": "Describe it"},
            ],
        }
    ]
    prompt_path = tmp_path / "prompts.jsonl"
    prompt_path.write_text(
        "\n".join(
            [
                json.dumps({"prompt": text_prompt}),
                json.dumps({"prompt": multimodal_prompt}),
            ]
        )
        + "\n"
    )
    processor = _RecordingProcessor()

    dataset = Dataset(
        str(prompt_path),
        tokenizer=None,
        processor=processor,
        max_length=None,
        prompt_key="prompt",
        apply_chat_template=False,
    )

    assert dataset.samples[0].prompt == text_prompt
    assert dataset.samples[0].multimodal_inputs is None
    assert dataset.samples[1].prompt == multimodal_prompt
    assert dataset.samples[1].multimodal_inputs == {
        "images": ["image"],
        "videos": [],
    }
    assert processor.prompts == [multimodal_prompt]
