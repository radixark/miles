"""prepare_avqa_audio.py ROOT: audio multiple-choice prompts for `scripts/run_inkling.py --task avqa_audio`.

Train and eval: AVQA (Joysw909/AVQA: one JSON object per line plus the wav clips), the set
R1-AQA and Omni-R1 trained on. The repack only carries AVQA's training subset, so 2,000 of its
clips, drawn per question type with a fixed seed, are held out with all their questions as
`avqa_heldout.jsonl`, so no held-out audio is ever trained on. Everything becomes a GPQA-style
prompt graded by `examples/avqa_audio/reward.py`.

Every clip is written as 16 kHz mono PCM WAV, the rate Inkling's dMel extractor takes: the engines'
loader and the Miles-side processor then decode the same samples and never resample.

    hf download --repo-type dataset Joysw909/AVQA --local-dir ROOT/avqa
    python examples/avqa_audio/prepare_avqa_audio.py ROOT
"""

import collections
import io
import json
import random
import string
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio.functional as AF

ROOT = Path(sys.argv[1])
OUT = ROOT / "avqa_audio"
SAMPLE_RATE = 16000
INSTRUCTION = "Listen to the audio and answer the question. Answer with the letter of the correct option."
HELDOUT_SIZE = 2000


def prompt(audio_path: str, question: str, choices: list[str]) -> list[dict]:
    options = "\n".join(
        f"({letter}) {choice}" for letter, choice in zip(string.ascii_uppercase[: len(choices)], choices, strict=True)
    )
    return [
        {
            "role": "user",
            "content": [
                {"type": "audio", "audio": audio_path},
                {"type": "text", "text": f"{INSTRUCTION}\n\n{question}\n{options}"},
            ],
        }
    ]


def write_wav(source, path: Path) -> None:
    """Decode ``source`` (bytes or a path) and write it as 16 kHz mono PCM16 WAV."""
    samples, sample_rate = sf.read(
        io.BytesIO(source) if isinstance(source, bytes) else source, dtype="float32", always_2d=True
    )
    mono = samples.mean(axis=1)
    if sample_rate != SAMPLE_RATE:
        mono = AF.resample(torch.from_numpy(mono), orig_freq=sample_rate, new_freq=SAMPLE_RATE).numpy()
    sf.write(path, np.ascontiguousarray(mono, dtype=np.float32), SAMPLE_RATE, subtype="PCM_16")


def write(rows: list[dict], path: Path) -> None:
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    print(f"{path}: {len(rows)} rows")


def avqa_rows() -> tuple[list[dict], list[dict]]:
    """AVQA as (train, held-out): whole clips, the held-out share of each question type matching its share of the data."""
    wav_dir = OUT / "avqa_wav"
    wav_dir.mkdir(parents=True, exist_ok=True)
    by_clip = collections.defaultdict(list)
    for line in (ROOT / "avqa" / "train_r1aqa_line.json").open():
        item = json.loads(line)
        # audio_path is "./Joysw909/AVQA/<dir>/<clip>.wav"; the download holds <dir>/<clip>.wav.
        source = ROOT / "avqa" / Path(item["audio_path"]).parent.name / Path(item["audio_path"]).name
        if not source.is_file():
            continue
        audio_path = wav_dir / source.name
        if not audio_path.is_file():
            write_wav(str(source), audio_path)
        by_clip[audio_path].append(
            (
                item["question_type"],
                {
                    "prompt": prompt(str(audio_path.absolute()), item["question_text"], item["multi_choice"]),
                    "label": string.ascii_uppercase[item["answer"]],
                },
            )
        )
    # A clip with several questions goes to one side whole, under its first question's type.
    by_type = collections.defaultdict(list)
    for questions in by_clip.values():
        by_type[questions[0][0]].append([row for _, row in questions])
    total = sum(len(clips) for clips in by_type.values())
    rng = random.Random(0)
    train, heldout = [], []
    for clips in by_type.values():
        rng.shuffle(clips)
        cut = round(HELDOUT_SIZE * len(clips) / total)
        heldout += [row for clip in clips[:cut] for row in clip]
        train += [row for clip in clips[cut:] for row in clip]
    rng.shuffle(train)
    return train, heldout


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    train, heldout = avqa_rows()
    write(train, OUT / "train.jsonl")
    write(heldout, OUT / "avqa_heldout.jsonl")
