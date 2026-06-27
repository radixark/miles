"""MiniMax-M3-VL composite model for miles (HF-native vision + Megatron LM).

This is the "simple, less-efficient, end-to-end" VL path, modeled directly on
Megatron-Bridge's ``KimiK25VLModel`` (HF-native vision tower + projector +
Megatron ``language_model``, vision weights *replicated* rather than TP-sharded).

    ┌─────────────────────────────────────────────────────────────┐
    │ MiniMaxM3VLModel(MegatronModule)                            │
    │   vision_tower         <- native HF module (frozen-friendly) │
    │   multi_modal_projector<- native HF module                   │
    │   language_model        = Megatron GPTModel built with       │
    │                           get_minimax_m3_spec (MSA + MoE)     │
    └─────────────────────────────────────────────────────────────┘

forward():
  1. text embeds   : e = language_model.embedding(input_ids)        [t, b, h]
  2. vision feats  : v = projector(vision_tower(pixel_values, grid)) [n_img_tok, h]
  3. merge         : scatter v into e at image/video placeholder positions
  4. decode        : language_model(decoder_input=e, ...)  (skips its embed layer)

Why HF-native vision: M3's tower is Qwen-VL-family (Conv3d patches, 3D-RoPE,
dynamic resolution, 2×2 merge, video) — not in Megatron core (CLIP/RADIO only).
Reimplementing it Megatron-parallel is a large, separate effort; loading the HF
tower verbatim and replicating it gets a correct e2e first. Swap in a parallel
tower later behind the same interface.

Verified in isolation: the embedding-merge logic has a CPU unit test in
``tests`` of the README. The full forward needs GPU + the HF M3 modeling.
"""

from __future__ import annotations

import logging
import os

import torch

logger = logging.getLogger(__name__)

try:
    from megatron.core.transformer.module import MegatronModule
except Exception:  # pragma: no cover - allows import on CPU/no-megatron boxes
    MegatronModule = object  # type: ignore


def merge_vision_into_text(
    text_embeds: torch.Tensor,   # [t, b, h]  (Megatron layout: seq-first)
    vision_embeds: torch.Tensor | None,  # [num_vision_tokens, h]
    input_ids: torch.Tensor,     # [b, t]
    image_token_index: int,
    video_token_index: int,
) -> torch.Tensor:
    """Scatter vision embeddings into the placeholder positions of the text stream.

    The number of placeholder tokens in ``input_ids`` MUST equal
    ``vision_embeds.shape[0]`` (the data-side ``mm_data`` expansion guarantees
    this by expanding one media token into the grid-derived number of slots).

    Returns the merged embeddings in the same ``[t, b, h]`` layout.
    """
    if vision_embeds is None or vision_embeds.numel() == 0:
        return text_embeds

    t, b, h = text_embeds.shape
    # work in [b, t, h] for masked_scatter, then transpose back
    e = text_embeds.transpose(0, 1).contiguous()  # [b, t, h]
    mask = (input_ids == image_token_index) | (input_ids == video_token_index)  # [b, t]
    n_slots = int(mask.sum().item())
    if n_slots != vision_embeds.shape[0]:
        raise ValueError(
            f"vision/token mismatch: {n_slots} placeholder tokens but "
            f"{vision_embeds.shape[0]} vision embeddings. Did mm_data expansion run?"
        )
    e = e.masked_scatter(mask.unsqueeze(-1), vision_embeds.to(e.dtype))
    return e.transpose(0, 1).contiguous()  # [t, b, h]


class MiniMaxM3VLModel(MegatronModule):
    """Composite M3-VL model: HF vision tower + projector + Megatron M3 LM."""

    def __init__(
        self,
        language_model,                 # Megatron GPTModel (built via get_minimax_m3_spec)
        vision_tower: torch.nn.Module,  # native HF module
        multi_modal_projector: torch.nn.Module,  # native HF module
        *,
        image_token_index: int,
        video_token_index: int,
        config=None,
        freeze_vision: bool = True,
    ) -> None:
        if MegatronModule is object:
            super().__init__()
        else:
            super().__init__(config=getattr(language_model, "config", config))
        self.language_model = language_model
        self.vision_tower = vision_tower
        self.multi_modal_projector = multi_modal_projector
        self.image_token_index = image_token_index
        self.video_token_index = video_token_index

        if freeze_vision:
            for p in self.vision_tower.parameters():
                p.requires_grad_(False)
            # projector is usually trained; leave it learnable.

        # Megatron pipeline plumbing: expose pre/post like the inner LM so the
        # scheduler and pipeline wrappers treat this as a drop-in decoder.
        self.pre_process = getattr(language_model, "pre_process", True)
        self.post_process = getattr(language_model, "post_process", True)
        self.share_embeddings_and_output_weights = getattr(
            language_model, "share_embeddings_and_output_weights", False
        )

    # -- vision -------------------------------------------------------------
    def _image_features(self, pixel_values, image_grid_thw, pixel_values_videos=None,
                        video_grid_thw=None):
        """Run the HF vision tower + projector -> [num_vision_tokens, hidden].

        Contract mirrors HF VLM ``get_image_features``. We try the HF model's own
        helper first (most faithful), then fall back to tower+projector. The exact
        M3 API is resolved at runtime to avoid hard-coding submodule internals.
        """
        feats = []
        if pixel_values is not None:
            # mirror HF MiniMaxM3VLModel.get_image_features:
            #   projector(vision_tower(pixel_values, image_grid_thw=...).last_hidden_state.squeeze(0))
            v = self.vision_tower(pixel_values=pixel_values, image_grid_thw=image_grid_thw)
            v = v.last_hidden_state if hasattr(v, "last_hidden_state") else v
            feats.append(self.multi_modal_projector(v.squeeze(0)))
        if pixel_values_videos is not None:
            vv = self.vision_tower(pixel_values=pixel_values_videos, image_grid_thw=video_grid_thw)
            vv = vv.last_hidden_state if hasattr(vv, "last_hidden_state") else vv
            feats.append(self.multi_modal_projector(vv.squeeze(0)))
        if not feats:
            return None
        out = torch.cat(feats, dim=0)
        return out.reshape(-1, out.shape[-1])  # [num_vision_tokens, hidden]

    # -- forward ------------------------------------------------------------
    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids=None,
        attention_mask=None,
        labels=None,
        loss_mask=None,
        packed_seq_params=None,
        pixel_values=None,
        image_grid_thw=None,
        pixel_values_videos=None,
        video_grid_thw=None,
        decoder_input=None,
        **kwargs,
    ):
        # Optional probe (M3_VL_PROBE=1): log a vision + projector weight norm with
        # the TP rank, so a single run shows (a) whether vision TRAINS (norm changes
        # across steps) and (b) whether the replicas stay CONSISTENT across TP ranks
        # (grad-sync). Throttled; cheap when off.
        if os.environ.get("M3_VL_PROBE") and self.pre_process:
            self._probe()

        # On non-first pipeline stages there is no embedding; just pass through.
        if not self.pre_process:
            return self.language_model(
                input_ids=input_ids, position_ids=position_ids,
                attention_mask=attention_mask, labels=labels, loss_mask=loss_mask,
                packed_seq_params=packed_seq_params, decoder_input=decoder_input, **kwargs,
            )

        # 1. text embeddings (seq-first [t, b, h])
        text_embeds = self.language_model.embedding(input_ids=input_ids, position_ids=None)

        # 2 + 3. vision features merged at placeholder positions
        vision_embeds = self._image_features(
            pixel_values, image_grid_thw, pixel_values_videos, video_grid_thw
        )
        merged = merge_vision_into_text(
            text_embeds, vision_embeds, input_ids,
            self.image_token_index, self.video_token_index,
        )

        # 4. decode with precomputed embeddings (LM skips its own embed layer)
        return self.language_model(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            decoder_input=merged,
            labels=labels,
            loss_mask=loss_mask,
            packed_seq_params=packed_seq_params,
            **kwargs,
        )

    def _probe(self):
        self._probe_n = getattr(self, "_probe_n", 0) + 1
        if self._probe_n % 4 != 1:  # throttle to ~once per few microbatches
            return
        try:
            import torch.distributed as dist
            grank = dist.get_rank() if dist.is_initialized() else 0
        except Exception:
            grank = 0
        if grank not in (0, 1):  # tp0 vs tp1 (same dp group) — enough to see divergence
            return
        try:
            from megatron.core import parallel_state as mpu
            tp, dp = mpu.get_tensor_model_parallel_rank(), mpu.get_data_parallel_rank()
        except Exception:
            tp = dp = -1
        def _info(mod):
            p = next((q for _, q in mod.named_parameters() if q.requires_grad), None)
            if p is None:
                p = next((q for _, q in mod.named_parameters()), None)
            if p is None:
                return "n/a"
            wn = float(p.detach().float().norm())
            g = getattr(p, "grad", None)
            mg = getattr(p, "main_grad", None)
            gn = float(g.detach().float().norm()) if g is not None else None
            mgn = float(mg.detach().float().norm()) if mg is not None else None
            return (f"w={wn:.6f} rg={p.requires_grad} dtype={str(p.dtype).split('.')[-1]} "
                    f"has_main_grad={hasattr(p, 'main_grad')} grad={gn} main_grad={mgn}")
        print(f"[M3VL-PROBE] grank={grank} tp={tp} dp={dp} call={self._probe_n} "
              f"| vision[{_info(self.vision_tower)}] | proj[{_info(self.multi_modal_projector)}]",
              flush=True)

    # Delegate the bits Megatron's training loop / checkpoint expect to the LM.
    def set_input_tensor(self, input_tensor):
        return self.language_model.set_input_tensor(input_tensor)

    def shared_embedding_or_output_weight(self):
        return self.language_model.shared_embedding_or_output_weight()

    def sharded_state_dict(self, prefix="", sharded_offsets=(), metadata=None):
        """Delegate the megatron dist checkpoint to the inner LM with the SAME prefix.

        The torch_dist checkpoint is saved/produced as a bare GPTModel
        (``embedding.*`` / ``decoder.*`` — no ``language_model.`` level), so without
        this the default MegatronModule.sharded_state_dict would prefix every LM
        param with ``language_model.`` and ``--load`` finds nothing ("not in state
        dict, will skip" -> rename_mapping assert). The vision tower + projector are
        loaded separately from M3_VISION_CKPT (build_minimax_m3_vl), so they are
        intentionally excluded from the dist checkpoint.
        """
        return self.language_model.sharded_state_dict(
            prefix=prefix, sharded_offsets=sharded_offsets, metadata=metadata
        )


def _load_vision_only(ckpt_dir, hf_cfg, dtype):
    """Build + load ONLY the M3-VL vision tower + projector (skip the 428B LM).

    Loads just the ``vision_tower.*`` (~515) and ``multi_modal_projector.*`` (4)
    tensors straight from the safetensors shards via the weight-map index — never
    materialising the 22.9k ``language_model.*`` keys, so CPU RAM stays small.
    Requires a transformers that has the in-library ``minimax_m3_vl`` model
    (mounted + PYTHONPATH-shadowed for VL mode; see the smoke slurm).
    """
    import glob
    import json
    import os

    from safetensors import safe_open
    from transformers.models.minimax_m3_vl.modeling_minimax_m3_vl import (
        MiniMaxM3VLMultiModalProjector,
        MiniMaxM3VLVisionModel,
    )

    vision_tower = MiniMaxM3VLVisionModel(hf_cfg.vision_config).to(dtype)
    projector = MiniMaxM3VLMultiModalProjector(hf_cfg).to(dtype)

    # Map checkpoint keys (older CLIP-style layout) -> transformers-5.3 module keys.
    def vt_rename(k):  # k already stripped of "vision_tower."
        k = k.replace("vision_model.encoder.layers.", "layers.")
        k = k.replace("vision_model.embeddings.patch_embedding.", "embeddings.proj.")
        k = k.replace("vision_model.", "")  # pre_layrnorm, post_layernorm, ...
        return k

    def pj_collect(k, t, pj_sd):  # k is the full checkpoint key
        if k.startswith("multi_modal_projector."):
            pj_sd[k[len("multi_modal_projector."):]] = t        # linear_1/linear_2 (direct)
        elif k.startswith("patch_merge_mlp."):                  # separate top-level module ->
            pj_sd["merge_" + k[len("patch_merge_mlp."):]] = t   # projector.merge_linear_1/2

    wanted = ("vision_tower.", "multi_modal_projector.", "patch_merge_mlp.")
    idx_path = os.path.join(ckpt_dir, "model.safetensors.index.json")
    if os.path.exists(idx_path):
        weight_map = json.load(open(idx_path))["weight_map"]
        shard_of = {}
        for k, f in weight_map.items():
            if k.startswith(wanted):
                shard_of.setdefault(f, []).append(k)
        shards = shard_of.items()
    else:  # single-file checkpoint
        only = glob.glob(os.path.join(ckpt_dir, "*.safetensors"))[0]
        shards = [(os.path.basename(only), None)]

    vt_sd, pj_sd = {}, {}
    for fname, keys in shards:
        with safe_open(os.path.join(ckpt_dir, fname), framework="pt") as sf:
            it = keys if keys is not None else [k for k in sf.keys() if k.startswith(wanted)]
            for k in it:
                t = sf.get_tensor(k).to(dtype)
                if k.startswith("vision_tower."):
                    vt_sd[vt_rename(k[len("vision_tower."):])] = t
                else:
                    pj_collect(k, t, pj_sd)

    missing_vt, unexpected_vt = vision_tower.load_state_dict(vt_sd, strict=False)
    missing_pj, unexpected_pj = projector.load_state_dict(pj_sd, strict=False)
    if missing_vt or missing_pj:
        logger.warning(
            "M3-VL vision load: %d missing vision keys, %d missing projector keys "
            "(unexpected vt=%d pj=%d). Check the checkpoint->module key rename.",
            len(missing_vt), len(missing_pj), len(unexpected_vt), len(unexpected_pj),
        )
    else:
        logger.info("M3-VL vision load: all vision_tower + projector keys loaded.")
    return vision_tower, projector


def build_minimax_m3_vl(language_model, args, *, freeze_vision: bool = True):
    """Assemble the composite: Megatron M3 ``language_model`` + HF vision tower.

    Loads ONLY the vision tower + projector from the HF checkpoint (the HF LM is
    discarded — Megatron's ``language_model`` replaces it). Vision runs in bf16,
    replicated across TP ranks (the "less efficient" e2e path). Call this from the
    miles model_provider after the GPTModel is built (see README hook).
    """
    import os

    import torch

    # Tell the M3 bridge to use the composite ("language_model.") param prefix
    # and add vision ReplicatedMappings (see miles_plugins/megatron_bridge/minimax_m3.py).
    os.environ["MINIMAX_M3_VL"] = "1"

    # Load the config via the IN-LIBRARY 5.3 MiniMaxM3VLConfig (not the checkpoint's
    # bundled remote-code config, which is older and lacks fields the 5.3 vision
    # modeling reads, e.g. temporal_patch_size).
    from transformers.models.minimax_m3_vl.configuration_minimax_m3_vl import (
        MiniMaxM3VLConfig,
    )

    hf_cfg = MiniMaxM3VLConfig.from_pretrained(args.hf_checkpoint)
    dev = torch.device("cuda", torch.cuda.current_device())
    # Vision weights come from a checkpoint that actually has the safetensors. For
    # reduced-layer smokes, --hf-checkpoint is a config-only 4-layer dir (so
    # hf_validate_args passes), and M3_VISION_CKPT points at the full checkpoint.
    vision_ckpt = os.environ.get("M3_VISION_CKPT") or args.hf_checkpoint
    vision_tower, projector = _load_vision_only(vision_ckpt, hf_cfg, torch.bfloat16)
    vision_tower = vision_tower.to(dev)
    projector = projector.to(dev)

    image_token_index = getattr(hf_cfg, "image_token_index", 200025)
    video_token_index = getattr(hf_cfg, "video_token_index", 200026)
    return MiniMaxM3VLModel(
        language_model, vision_tower, projector,
        image_token_index=image_token_index,
        video_token_index=video_token_index,
        config=getattr(language_model, "config", None),
        freeze_vision=freeze_vision,
    )
