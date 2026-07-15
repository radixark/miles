from __future__ import annotations

import io
import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F


@dataclass
class InklingAudioEncoderParams:
    sample_rate: int = 16_000
    window_size_multiplier: float = 2.0
    n_fft: int | None = None
    n_mels: int = 80
    num_dmel_bins: int = 16
    dmel_min_value: float = -7.0
    dmel_max_value: float = 2.0
    audio_token_duration_s: float = 0.05


def _load_audio_bytes(audio) -> bytes:
    if isinstance(audio, (bytes, bytearray)):
        return bytes(audio)
    if hasattr(audio, "read"):
        return audio.read()
    if isinstance(audio, str):
        path = audio[len("file://") :] if audio.startswith("file://") else audio
        with open(path, "rb") as f:
            return f.read()
    raise TypeError(f"Unsupported audio input type for Inkling audio extractor: {type(audio)}")


def _to_exact_int(value: float, name: str, tolerance: float = 1e-6) -> int:
    rounded = round(value)
    if abs(value - rounded) > tolerance:
        raise ValueError(f"{name} must resolve to an integer sample count, got {value}")
    return int(rounded)


def _decode_audio(audio_bytes: bytes, sample_rate: int) -> torch.Tensor:
    import soundfile as sf

    samples, src_sample_rate = sf.read(io.BytesIO(audio_bytes), dtype="float32", always_2d=True)
    mono = samples.mean(axis=1)
    if src_sample_rate != sample_rate:
        mono = _resample(mono, src_sample_rate, sample_rate)
    return torch.from_numpy(np.ascontiguousarray(mono, dtype=np.float32))


def _resample(samples: np.ndarray, src_sample_rate: int, sample_rate: int) -> np.ndarray:
    import torchaudio.functional as AF

    audio = torch.from_numpy(np.ascontiguousarray(samples, dtype=np.float32))
    resampled = AF.resample(audio, orig_freq=src_sample_rate, new_freq=sample_rate)
    return resampled.detach().cpu().numpy().astype(np.float32, copy=False)


def _hz_to_mel(frequencies: np.ndarray) -> np.ndarray:
    """Slaney mel scale, matching the librosa/torchaudio convention."""
    frequencies = np.asarray(frequencies, dtype=np.float64)
    f_sp = 200.0 / 3.0
    min_log_hz = 1000.0
    min_log_mel = min_log_hz / f_sp
    logstep = np.log(6.4) / 27.0
    linear = frequencies / f_sp
    log = min_log_mel + np.log(np.maximum(frequencies, min_log_hz) / min_log_hz) / logstep
    return np.where(frequencies >= min_log_hz, log, linear)


def _mel_to_hz(mels: np.ndarray) -> np.ndarray:
    mels = np.asarray(mels, dtype=np.float64)
    f_sp = 200.0 / 3.0
    min_log_hz = 1000.0
    min_log_mel = min_log_hz / f_sp
    logstep = np.log(6.4) / 27.0
    linear = mels * f_sp
    log = min_log_hz * np.exp(logstep * (mels - min_log_mel))
    return np.where(mels >= min_log_mel, log, linear)


_MEL_BASIS_CACHE: dict[tuple[int, int, int], torch.Tensor] = {}


def _mel_basis(sample_rate: int, n_fft: int, n_mels: int) -> torch.Tensor:
    key = (sample_rate, n_fft, n_mels)
    cached = _MEL_BASIS_CACHE.get(key)
    if cached is not None:
        return cached

    fft_bins = n_fft // 2 + 1
    fft_freqs = np.arange(fft_bins, dtype=np.float64) * sample_rate / n_fft
    mel_edges = _mel_to_hz(
        np.linspace(
            _hz_to_mel(np.array([0.0]))[0],
            _hz_to_mel(np.array([sample_rate / 2.0]))[0],
            n_mels + 2,
            dtype=np.float64,
        )
    )
    mel_widths = np.diff(mel_edges)
    lower = (fft_freqs[None, :] - mel_edges[:-2, None]) / mel_widths[:-1, None]
    upper = (mel_edges[2:, None] - fft_freqs[None, :]) / mel_widths[1:, None]
    weights = np.maximum(0.0, np.minimum(lower, upper))

    # Slaney area normalization.
    weights *= (2.0 / (mel_edges[2:] - mel_edges[:-2]))[:, None]
    basis = torch.from_numpy(weights.astype(np.float32, copy=False)).contiguous()
    _MEL_BASIS_CACHE[key] = basis
    return basis


def _dmel_bins(audio: torch.Tensor, params: InklingAudioEncoderParams) -> torch.Tensor:
    hop_length = _to_exact_int(
        params.audio_token_duration_s * params.sample_rate,
        "audio_token_duration_s * sample_rate",
    )
    window_size = _to_exact_int(
        params.audio_token_duration_s * params.window_size_multiplier * params.sample_rate,
        "audio_token_duration_s * window_size_multiplier * sample_rate",
    )
    n_fft = params.n_fft or window_size
    if hop_length <= 0 or window_size <= 0 or n_fft <= 0:
        raise ValueError("audio hop length, window size, and n_fft must be positive")
    if audio.numel() == 0:
        return torch.empty((0, params.n_mels), dtype=torch.int32)

    right_pad = math.ceil(audio.numel() / hop_length) * hop_length - audio.numel()
    left_pad = max(n_fft - hop_length, 0)
    audio = F.pad(audio, (left_pad, right_pad))

    window = torch.hann_window(window_size, periodic=True, dtype=torch.float32)
    spec = torch.stft(
        audio.unsqueeze(0),
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=window_size,
        window=window,
        center=False,
        normalized=False,
        onesided=True,
        return_complex=True,
    )
    spec_ri = torch.view_as_real(spec)
    magnitude = (spec_ri[..., 0].square() + spec_ri[..., 1].square()).clamp_min(1e-10).sqrt().squeeze(0)

    mel = _mel_basis(params.sample_rate, n_fft, params.n_mels).matmul(magnitude).clamp_min(1e-10).log10()
    mel = mel.to(torch.float64).clamp(min=params.dmel_min_value, max=params.dmel_max_value)
    bin_centers = torch.linspace(
        params.dmel_min_value,
        params.dmel_max_value,
        params.num_dmel_bins,
        dtype=torch.float64,
    )
    dmel_bins = (mel.unsqueeze(-1) - bin_centers).abs().argmin(dim=-1)
    return dmel_bins.to(torch.int32).T.contiguous()


class InklingAudioDmelExtractor:
    """audios -> {dmel_bins: [T_i, n_mels] int32 per clip, num_audio_tokens} (no HF deps)."""

    def __init__(self, params: dict | None = None):
        merged = InklingAudioEncoderParams()
        if params:
            for k, v in params.items():
                if hasattr(merged, k):
                    setattr(merged, k, v)
        self.params = merged

    def extract(self, audios: Sequence) -> dict:
        if not isinstance(audios, (list, tuple)):
            audios = [audios]
        dmel_bins = [
            _dmel_bins(_decode_audio(_load_audio_bytes(a), self.params.sample_rate), self.params) for a in audios
        ]
        return {
            "dmel_bins": dmel_bins,
            "num_audio_tokens": [int(b.shape[0]) for b in dmel_bins],
        }
