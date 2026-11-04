"""prepare_avqa_audio.py ROOT: audio multiple-choice prompts for `scripts/run_inkling.py --task avqa_audio`.

Train: AVQA as R1-AQA and Omni-R1 trained on it (Joysw909/AVQA: one JSON object per line plus the
wav clips), as GPQA-style prompts scored by `--rm-type gpqa`.

Every clip is written as 16 kHz mono PCM WAV, the rate Inkling's dMel extractor takes: the engines'
loader and the Miles-side processor then decode the same samples and never resample.

    hf download --repo-type dataset Joysw909/AVQA --local-dir ROOT/avqa
    python examples/avqa_audio/prepare_avqa_audio.py ROOT
"""

import io
import json
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


def prompt(audio_path: str, question: str, choices: list[str]) -> list[dict]:
    options = "\n".join(f"({letter}) {choice}" for letter, choice in zip(string.ascii_uppercase[: len(choices)], choices, strict=True))
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


def avqa_rows() -> list[dict]:
    wav_dir = OUT / "avqa_wav"
    wav_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for line in (ROOT / "avqa" / "train_r1aqa_line.json").open():
        item = json.loads(line)
        # audio_path is "./Joysw909/AVQA/<dir>/<clip>.wav"; the download holds <dir>/<clip>.wav.
        source = ROOT / "avqa" / Path(item["audio_path"]).parent.name / Path(item["audio_path"]).name
        if not source.is_file():
            continue
        audio_path = wav_dir / source.name
        write_wav(str(source), audio_path)
        rows.append(
            {
                "prompt": prompt(str(audio_path.resolve()), item["question_text"], item["multi_choice"]),
                "label": string.ascii_uppercase[item["answer"]],
            }
        )
    return rows


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    write(avqa_rows(), OUT / "train.jsonl")
