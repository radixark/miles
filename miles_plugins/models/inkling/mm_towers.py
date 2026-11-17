from __future__ import annotations

import json

import torch
import torch.nn as nn

from miles.backends.training_utils.data.context_parallel import scatter_cp_rows
from miles_plugins.models.inkling.vision_encoder import HMLPPatchEncoder, InklingVisionConfig, RMSNorm


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


def build_mm_towers(
    hf_checkpoint: str, device, dtype, train: bool = False
) -> tuple[nn.Module | None, nn.Module | None]:
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
            tower.requires_grad_(train)
            if not train:
                tower.eval()
    return visual, audio


# Token-space field -> the field name the Inkling towers read. SGLang's Inkling strategy names the
# features as its processors do and places every modality's rows as mm_<modality>_positions.
_TOKEN_SPACE_MEDIA_FIELDS = {
    "vision_patches_bthwc": "mm_vision_patches",
    "mm_image_positions": "mm_vision_positions",
    "dmel_bins": "mm_audio_dmel",
}


def _tower_media_kwargs(kwargs: dict) -> dict:
    """Pop the token-space media fields of ``kwargs`` and return them under the towers' names."""
    return {
        tower_name: kwargs.pop(field) for field, tower_name in _TOKEN_SPACE_MEDIA_FIELDS.items() if field in kwargs
    }


def wire_mm_towers(model, hf_checkpoint: str, train: bool = False) -> None:
    """Wrap an InklingGPTModel's forward for multimodal training (pre_process builds+merges;
    later stages pass through). train=True trains the towers instead of freezing them
    (weight-sync/ckpt for tower params is not wired yet)."""
    import megatron.core.parallel_state as ps

    _orig_forward = model.forward

    if not getattr(
        model, "pre_process", False
    ):  # config-access-exempt: only first pipeline stages construct multimodal towers

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
            _tower_media_kwargs(kw)
            return _orig_forward(*a, **kw)

        model.forward = _mm_passthrough
        return

    device = torch.cuda.current_device()
    visual, audio = build_mm_towers(hf_checkpoint, device=device, dtype=model.config.params_dtype, train=train)
    model.__dict__["_mm_towers"] = (visual, audio)

    def _scatter(decoder_input, embeds, positions):
        if embeds is None or positions is None or positions.numel() == 0:
            return decoder_input
        sp_rank = (
            ps.get_tensor_model_parallel_rank()
            if model.config.sequence_parallel and ps.get_tensor_model_parallel_world_size() > 1
            else 0
        )
        return scatter_cp_rows(decoder_input, embeds, positions, sp_rank=sp_rank)

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
        token_space = _tower_media_kwargs(kw)
        mm_vision_patches = token_space.get("mm_vision_patches", mm_vision_patches)
        mm_vision_positions = token_space.get("mm_vision_positions", mm_vision_positions)
        mm_audio_dmel = token_space.get("mm_audio_dmel", mm_audio_dmel)
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
        # the grad context must cover ONLY the tower forwards: _scatter clones
        # decoder_input, and a clone under no_grad detaches the embedding graph
        ctx = torch.enable_grad() if train else torch.no_grad()
        if mm_vision_patches is not None:
            with ctx:
                v_emb = model.__dict__["_mm_towers"][0](mm_vision_patches.to(device=decoder_input.device))
            decoder_input = _scatter(decoder_input, v_emb, mm_vision_positions.to(decoder_input.device))
        if mm_audio_dmel is not None:
            with ctx:
                a_emb = model.__dict__["_mm_towers"][1](mm_audio_dmel.to(device=decoder_input.device))
            decoder_input = _scatter(decoder_input, a_emb, mm_audio_positions.to(decoder_input.device))
        kw["decoder_input"] = decoder_input
        return _orig_forward(*a, **kw)

    model.forward = _mm_forward
