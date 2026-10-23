"""Differentiable text-only DiffusionGemma with one FSDP-owned parameter stack.

The decoder checkpoint namespace is canonical. Encoder and decoder reuse the same
layers through their ordinary module calls; the encoder KV tensors remain in the
training graph. Masks must be explicit dictionaries keyed by HF layer type, with
True meaning an allowed attention edge for boolean masks.
"""

import torch
from torch import nn
from torch.nn import functional as F
from transformers import DiffusionGemmaConfig
from transformers.models.diffusion_gemma.modeling_diffusion_gemma import (
    DiffusionGemmaDecoderModel,
    DiffusionGemmaDecoderTextAttention,
    DiffusionGemmaDecoderTextLayer,
    DiffusionGemmaPreTrainedModel,
    DiffusionGemmaRMSNorm,
    DiffusionGemmaSelfConditioning,
    DiffusionGemmaTextRotaryEmbedding,
    DiffusionGemmaTextScaledWordEmbedding,
    apply_rotary_pos_emb,
    eager_attention_forward,
    repeat_kv,
)

KVPair = tuple[torch.Tensor, torch.Tensor]
AttentionMasks = dict[str, torch.Tensor]


class DiffusionGemmaSharedTextAttention(DiffusionGemmaDecoderTextAttention):
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor,
        encoder_kv: KVPair | None = None,
    ) -> tuple[torch.Tensor, KVPair]:
        input_shape = hidden_states.shape[:-1]
        head_shape = (*input_shape, -1, self.head_dim)
        cos, sin = position_embeddings
        query = self.q_norm(self.q_proj(hidden_states).view(head_shape))
        query = apply_rotary_pos_emb(query, cos, sin, unsqueeze_dim=2).transpose(1, 2)
        raw_key = self.k_proj(hidden_states).view(head_shape)
        raw_value = self.v_proj(hidden_states).view(head_shape) if self.v_proj is not None else raw_key
        key = apply_rotary_pos_emb(self.k_norm(raw_key), cos, sin, unsqueeze_dim=2).transpose(1, 2)
        value = self.v_norm(raw_value).transpose(1, 2)
        own_kv = (key, value)
        if encoder_kv is not None:
            key = torch.cat((encoder_kv[0], key), dim=-2)
            value = torch.cat((encoder_kv[1], value), dim=-2)
        dropout = self.attention_dropout if self.training else 0.0
        if self.config._attn_implementation == "sdpa":
            attention_output = (
                F.scaled_dot_product_attention(
                    query,
                    repeat_kv(key, self.num_key_value_groups),
                    repeat_kv(value, self.num_key_value_groups),
                    attn_mask=attention_mask,
                    dropout_p=dropout,
                    is_causal=False,
                    scale=self.scaling,
                )
                .transpose(1, 2)
                .contiguous()
            )
        else:
            if attention_mask.dtype == torch.bool:
                attention_mask = torch.zeros_like(attention_mask, dtype=query.dtype).masked_fill(
                    ~attention_mask, torch.finfo(query.dtype).min
                )
            attention_output, _ = eager_attention_forward(
                self,
                query,
                key,
                value,
                attention_mask,
                scaling=self.scaling,
                dropout=dropout,
            )
        output = self.o_proj(attention_output.reshape(*input_shape, -1).contiguous())
        return output, own_kv


class DiffusionGemmaSharedTextLayer(DiffusionGemmaDecoderTextLayer):
    """HF differentiable MoE/MLP/norm layers, with explicit functional KV output."""

    def __init__(self, config, layer_idx: int):
        super().__init__(config, layer_idx)
        self.self_attn = DiffusionGemmaSharedTextAttention(config, layer_idx)
        self.register_buffer("encoder_layer_scalar", torch.ones(1))

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor,
        encoder_kv: KVPair | None = None,
        encoder_mode: bool = False,
    ) -> tuple[torch.Tensor, KVPair]:
        residual = hidden_states
        hidden_states, own_kv = self.self_attn(
            self.input_layernorm(hidden_states),
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            encoder_kv=encoder_kv,
        )
        hidden_states = residual + self.post_attention_layernorm(hidden_states)
        residual = hidden_states
        mlp_output = self.mlp(self.pre_feedforward_layernorm(hidden_states))
        mlp_output = self.post_feedforward_layernorm_1(mlp_output)
        routing_input = residual.reshape(-1, residual.shape[-1])
        _, top_k_weights, top_k_index = self.router(routing_input)
        expert_output = self.experts(
            self.pre_feedforward_layernorm_2(routing_input), top_k_index, top_k_weights
        ).reshape(residual.shape)
        expert_output = self.post_feedforward_layernorm_2(expert_output)
        hidden_states = residual + self.post_feedforward_layernorm(mlp_output + expert_output)
        scalar = self.encoder_layer_scalar if encoder_mode else self.layer_scalar
        return hidden_states * scalar.to(hidden_states.dtype), own_kv


class _SharedDecoder(DiffusionGemmaDecoderModel):
    def __init__(self, config: DiffusionGemmaConfig):
        # Build only one stack. Do not first construct HF decoder layers and replace
        # them: that doubles peak construction memory for the 26B checkpoint.
        DiffusionGemmaPreTrainedModel.__init__(self, config)
        text = config.text_config
        self.text_config = text
        self.embed_tokens = DiffusionGemmaTextScaledWordEmbedding(
            text.vocab_size, text.hidden_size, text.pad_token_id, embed_scale=text.hidden_size**0.5
        )
        self.layers = nn.ModuleList(
            DiffusionGemmaSharedTextLayer(text, layer_idx) for layer_idx in range(text.num_hidden_layers)
        )
        self.norm = DiffusionGemmaRMSNorm(text.hidden_size, eps=text.rms_norm_eps)
        self.rotary_emb = DiffusionGemmaTextRotaryEmbedding(text)
        self.self_conditioning = DiffusionGemmaSelfConditioning(text)
        self.unique_layer_types = set(text.layer_types)
        self.post_init()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_masks: AttentionMasks,
        position_ids: torch.Tensor,
        encoder_kvs: tuple[KVPair, ...] | None = None,
        self_conditioning_logits: torch.Tensor | None = None,
        self_conditioning_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, tuple[KVPair, ...]]:
        hidden_states = self.embed_tokens(input_ids)
        encoder_mode = encoder_kvs is None
        if not encoder_mode:
            soft_embeddings = torch.zeros_like(hidden_states)
            if self_conditioning_logits is not None:
                probabilities = self_conditioning_logits.softmax(dim=-1, dtype=torch.float32)
                soft_embeddings = probabilities.to(self.embed_tokens.weight.dtype) @ self.embed_tokens.weight
                soft_embeddings = soft_embeddings * self.embed_tokens.embed_scale.to(hidden_states.dtype)
                soft_embeddings = soft_embeddings * self_conditioning_mask[:, None, None].to(soft_embeddings.dtype)
            hidden_states = self.self_conditioning(hidden_states, soft_embeddings)
        positions = {kind: self.rotary_emb(hidden_states, position_ids, kind) for kind in self.unique_layer_types}
        output_kvs = []
        for index, layer in enumerate(self.layers):
            kind = self.text_config.layer_types[index]
            hidden_states, own_kv = layer(
                hidden_states,
                position_embeddings=positions[kind],
                attention_mask=attention_masks[kind],
                encoder_kv=None if encoder_mode else encoder_kvs[index],
                encoder_mode=encoder_mode,
            )
            output_kvs.append(own_kv)
        return self.norm(hidden_states), tuple(output_kvs)


class _SharedModel(nn.Module):
    def __init__(self, config: DiffusionGemmaConfig):
        super().__init__()
        self.decoder = _SharedDecoder(config)


class DiffusionGemmaForBlockDiffusion(DiffusionGemmaPreTrainedModel):
    """Text SFT model; no generation cache, multimodal inputs, or router aux loss.

    ``forward`` consumes clean ``[B,L]`` and noisy canvas ``[B,R]`` IDs, explicit
    local/full attention masks and positions, and a boolean ``[B]`` self-conditioning
    gate. It returns encoder and decoder logits. The first decoder pass always runs
    without gradients, including when every gate is false, so FSDP ranks execute
    the same module sequence. The final pass reads gradient-bearing encoder KV.
    """

    _tied_weights_keys = {"lm_head.weight": "model.decoder.embed_tokens.weight"}
    _no_split_modules = ["DiffusionGemmaSharedTextLayer"]
    supports_gradient_checkpointing = True
    _supports_flash_attn = False
    _supports_flex_attn = False
    _keys_to_ignore_on_load_unexpected = [r"model\.encoder\..*"]
    input_modalities = ("text",)

    def __init__(self, config: DiffusionGemmaConfig):
        if not config.tie_word_embeddings:
            raise ValueError("The shared-stack SFT model requires tied encoder/decoder weights.")
        super().__init__(config)
        self.model = _SharedModel(config)
        self.lm_head = nn.Linear(config.text_config.hidden_size, config.text_config.vocab_size, bias=False)
        self.final_logit_softcapping = config.text_config.final_logit_softcapping
        self.post_init()

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        mapping = dict(kwargs.pop("key_mapping", None) or {})
        mapping[r"^model\.encoder\.language_model\.layers\.(\d+)\.layer_scalar$"] = (
            r"model.decoder.layers.\1.encoder_layer_scalar"
        )
        return super().from_pretrained(pretrained_model_name_or_path, *model_args, key_mapping=mapping, **kwargs)

    def _init_weights(self, module):
        super()._init_weights(module)
        if isinstance(module, DiffusionGemmaSharedTextLayer):
            module.encoder_layer_scalar.fill_(1.0)

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        options = {"use_reentrant": False, **(gradient_checkpointing_kwargs or {})}
        if options["use_reentrant"]:
            raise ValueError("DiffusionGemma encoder KV outputs require non-reentrant gradient checkpointing.")
        return super().gradient_checkpointing_enable(options)

    def get_input_embeddings(self) -> nn.Embedding:
        return self.model.decoder.embed_tokens

    def set_input_embeddings(self, value: nn.Embedding) -> None:
        self.model.decoder.embed_tokens = value

    def _logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        logits = self.lm_head(hidden_states).float()
        if self.final_logit_softcapping is not None:
            logits = torch.tanh(logits / self.final_logit_softcapping) * self.final_logit_softcapping
        return logits

    def forward(
        self,
        input_ids: torch.Tensor,
        decoder_input_ids: torch.Tensor,
        encoder_attention_mask: AttentionMasks,
        decoder_attention_mask: AttentionMasks,
        position_ids: torch.Tensor,
        decoder_position_ids: torch.Tensor,
        self_conditioning_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self_conditioning_mask.dtype != torch.bool or self_conditioning_mask.shape != (input_ids.shape[0],):
            raise ValueError("self_conditioning_mask must be a boolean tensor with shape [batch_size].")
        encoder_hidden, encoder_kvs = self.model.decoder(
            input_ids, attention_masks=encoder_attention_mask, position_ids=position_ids
        )
        encoder_logits = self._logits(encoder_hidden)
        with torch.no_grad():
            first_hidden, _ = self.model.decoder(
                decoder_input_ids,
                attention_masks=decoder_attention_mask,
                position_ids=decoder_position_ids,
                encoder_kvs=encoder_kvs,
            )
            first_logits = self._logits(first_hidden)
        decoder_hidden, _ = self.model.decoder(
            decoder_input_ids,
            attention_masks=decoder_attention_mask,
            position_ids=decoder_position_ids,
            encoder_kvs=encoder_kvs,
            self_conditioning_logits=first_logits,
            self_conditioning_mask=self_conditioning_mask,
        )
        return encoder_logits, self._logits(decoder_hidden)
