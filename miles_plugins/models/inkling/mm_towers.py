from __future__ import annotations

import json

import torch
import torch.nn as nn

from miles_plugins.models.inkling.ops.vision_encoder import HMLPPatchEncoder, InklingVisionConfig, RMSNorm
from miles_plugins.models.inkling.options import inkling_opt


class InklingAudioEncoder(nn.Module):
    def __init__(self, audio_cfg: dict):
        super().__init__()
        assert audio_cfg.get("audio_mode", "dmel") == "dmel"
        self.n_mel_bins = audio_cfg["n_mel_bins"]
        self.mel_vocab_size = audio_cfg["mel_vocab_size"]
        self.encoder = nn.Embedding(self.n_mel_bins * self.mel_vocab_size, audio_cfg["decoder_dmodel"])
        self.final_norm = (
            RMSNorm(audio_cfg["decoder_dmodel"], eps=1e-6) if audio_cfg.get("use_audio_norm", True) else None
        )

    def forward(self, dmel: torch.Tensor) -> torch.Tensor:
        assert dmel.shape[1] == self.n_mel_bins
        idx = (torch.arange(self.n_mel_bins, device=dmel.device) * self.mel_vocab_size).unsqueeze(0) + dmel.to(
            torch.int32
        )
        h = self.encoder(idx.reshape(-1)).reshape(dmel.shape[0], self.n_mel_bins, -1).sum(axis=1)
        return self.final_norm(h) if self.final_norm is not None else h


class InklingVisionEncoder(nn.Module):
    def __init__(self, vision_cfg: dict):
        super().__init__()
        assert vision_cfg.get("vision_encoder_type", "hmlp") == "hmlp"
        self.vision_encoder = HMLPPatchEncoder(InklingVisionConfig(vision_cfg))

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        return self.vision_encoder(patches)


def _load_tower_tensors(hf_checkpoint: str) -> dict[str, torch.Tensor]:
    """Read tower tensors from HF safetensors, normalizing keys to visual.* / audio.*."""
    from safetensors import safe_open

    index = json.load(open(f"{hf_checkpoint}/model.safetensors.index.json"))
    by_file: dict[str, list[str]] = {}
    for name, fname in index["weight_map"].items():
        bare = name[len("model.") :] if name.startswith(("model.visual.", "model.audio.")) else name
        if bare.startswith(("visual.", "audio.")):
            by_file.setdefault(fname, []).append(name)
    out: dict[str, torch.Tensor] = {}
    for fname, names in by_file.items():
        with safe_open(f"{hf_checkpoint}/{fname}", framework="pt") as f:
            for name in names:
                bare = name[len("model.") :] if name.startswith(("model.visual.", "model.audio.")) else name
                out[bare] = f.get_tensor(name)
    return out


def build_mm_towers(hf_checkpoint: str, device, dtype) -> tuple[nn.Module | None, nn.Module | None]:
    """Construct + HF-load both towers; None when the checkpoint has no config for a modality."""
    cfg = json.load(open(f"{hf_checkpoint}/config.json"))
    weights = _load_tower_tensors(hf_checkpoint)

    visual = audio = None
    if cfg.get("vision_config") and any(k.startswith("visual.") for k in weights):
        visual = InklingVisionEncoder(cfg["vision_config"])
        sd = {k[len("visual.") :]: v for k, v in weights.items() if k.startswith("visual.")}
        visual.vision_encoder.load_state_dict(sd, strict=True)
    if cfg.get("audio_config") and any(k.startswith("audio.") for k in weights):
        audio = InklingAudioEncoder(cfg["audio_config"])
        sd = {k[len("audio.") :]: v for k, v in weights.items() if k.startswith("audio.")}
        audio.load_state_dict(sd, strict=True)

    for tower in (visual, audio):
        if tower is not None:
            tower.to(device=device, dtype=dtype)
            tower.requires_grad_(inkling_opt("inkling_train_mm_towers"))
            if not inkling_opt("inkling_train_mm_towers"):
                tower.eval()
    return visual, audio


def wire_mm_towers(model, hf_checkpoint: str) -> None:
    """Wrap an InklingGPTModel's forward for multimodal training (pre_process builds+merges; later stages pass through)."""
    import megatron.core.parallel_state as ps

    _orig_forward = model.forward

    if not getattr(model, "pre_process", False):

        def _mm_passthrough(
            *a,
            mm_vision_patches=None,
            mm_vision_positions=None,
            mm_vision_num_patches=None,
            mm_audio_dmel=None,
            mm_audio_positions=None,
            mm_audio_num_tokens=None,
            **kw,
        ):
            return _orig_forward(*a, **kw)

        model.forward = _mm_passthrough
        return

    device = torch.cuda.current_device()
    visual, audio = build_mm_towers(hf_checkpoint, device=device, dtype=model.config.params_dtype)
    model.__dict__["_mm_towers"] = (visual, audio)

    def _scatter(decoder_input, embeds, positions):
        if embeds is None or positions is None or positions.numel() == 0:
            return decoder_input
        s_local = decoder_input.shape[0]
        if model.config.sequence_parallel and ps.get_tensor_model_parallel_world_size() > 1:
            rank = ps.get_tensor_model_parallel_rank()
            local = positions - rank * s_local
            sel = (local >= 0) & (local < s_local)
            local, emb = local[sel], embeds[sel]
        else:
            local, emb = positions, embeds
        # Clone UNCONDITIONALLY on every rank. Guarding the clone behind
        # local.numel() diverges the TP group under sequence parallelism: only
        # ranks whose SP slice contains an image token would clone. Within one TP
        # group that leaves some ranks carrying a plain cloned tensor and others
        # the original SP-region tensor from the embedding all-gather, and the
        # subsequent SP reduce-scatter/all-gather pair mixes the two -> divergent
        # activations (vision blindness) or a wedged collective.
        decoder_input = decoder_input.clone()
        if local.numel():
            decoder_input[local, 0] = emb.to(decoder_input.dtype)
        return decoder_input

    def _mm_forward(
        *a,
        mm_vision_patches=None,
        mm_vision_positions=None,
        mm_vision_num_patches=None,
        mm_audio_dmel=None,
        mm_audio_positions=None,
        mm_audio_num_tokens=None,
        **kw,
    ):
        input_ids = kw.get("input_ids", a[0] if a else None)
        position_ids = kw.get("position_ids")
        if mm_vision_patches is not None:
            assert mm_vision_positions is not None, "mm_vision_patches given without mm_vision_positions"
            assert mm_vision_positions.numel() == mm_vision_patches.shape[0], (
                f"{mm_vision_positions.numel()} vision position(s) vs " f"{mm_vision_patches.shape[0]} patch row(s)"
            )
        if mm_audio_dmel is not None:
            assert mm_audio_positions is not None, "mm_audio_dmel given without mm_audio_positions"
            assert mm_audio_positions.numel() == mm_audio_dmel.shape[0], (
                f"{mm_audio_positions.numel()} audio position(s) vs " f"{mm_audio_dmel.shape[0]} dmel frame(s)"
            )
        decoder_input = model.embedding(input_ids=input_ids, position_ids=position_ids)
        ctx = torch.enable_grad() if inkling_opt("inkling_train_mm_towers") else torch.no_grad()
        with ctx:
            if mm_vision_patches is not None:
                v_emb = model.__dict__["_mm_towers"][0](mm_vision_patches.to(device=decoder_input.device))
                decoder_input = _scatter(decoder_input, v_emb, mm_vision_positions.to(decoder_input.device))
            if mm_audio_dmel is not None:
                a_emb = model.__dict__["_mm_towers"][1](mm_audio_dmel.to(device=decoder_input.device))
                decoder_input = _scatter(decoder_input, a_emb, mm_audio_positions.to(decoder_input.device))
        kw["decoder_input"] = decoder_input
        return _orig_forward(*a, **kw)

    model.forward = _mm_forward
