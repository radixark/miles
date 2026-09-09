"""Bridge for Qwen3.8-Next (HF ``architectures: [Qwen4ExpForConditionalGeneration]``).

Subclasses ``Qwen3_5Bridge`` to mirror sglang, where the model is literally
``Qwen4ExpModel(Qwen3_5ForCausalLM)`` wrapped in
``Qwen4ExpForConditionalGeneration(Qwen3VLForConditionalGeneration)``. Everything
Qwen3.5 already maps carries over unchanged: the 3:1 linear/full attention split,
the fused 3-D MoE experts, the shared expert and its gate, MTP.

What Qwen3.8-Next adds, all verified against the 1658-tensor safetensors index of
Qwen/Qwen3.8-Flash-Next:

  * **Hyper-Connection**, two per layer (``attn_hyper_connection`` before
    attention, ``mlp_hyper_connection`` before the MLP) plus one model-level
    ``hyper_connection_mixer`` contracting n*C -> C before the LM head.
  * **PLE** on exactly one layer (index 1 -- ``ple_layer_ids`` is 1-based, see
    ``_ple_layer_ids``), including a 102 GB n-gram table sharded 128 ways.
  * a sparse **indexer** on the full-attention layers.

And what it *removes*, which matters more than what it adds:

  * no ``input_layernorm`` and no ``post_attention_layernorm`` anywhere (0 of each
    in the index), because each HC's ``hc_norm`` is the pre-block norm;
  * no final norm either -- there is no ``model.language_model.norm.weight``,
    because the final mixer's ``hc_norm`` plays that role.

So the inherited entries for those have to be popped, not left to fail: mapping a
nonexistent HF key makes the loader raise on a missing tensor, and a spec that
keeps the norms would leave them at init values and quietly wreck the forward.

The mcore names below are the plain ``nn.Parameter`` attributes on the HC module
(``hc_norm_weight``), while HF stores each as a submodule's ``.weight``
(``hc_norm.weight``). Translating is the bridge's job; renaming the Megatron
params to match HF would mean inventing submodules that hold a single tensor.
"""

import torch
from mbridge.core import register_model

from miles_plugins.mbridge.qwen3_5 import Qwen3_5Bridge


@register_model(["qwen3.8_next", "qwen3_8_next", "qwen4_exp"])
class Qwen38NextBridge(Qwen3_5Bridge):
    """Weight mapping + Megatron config for Qwen3.8-Next."""

    _DIRECT_MAPPING = Qwen3_5Bridge._DIRECT_MAPPING.copy()

    # Qwen3.5 maps this to model.language_model.norm.weight, which Qwen3.8-Next
    # does not have -- the final mixer's hc_norm is the final norm.
    _DIRECT_MAPPING.pop("decoder.final_layernorm.weight", None)

    # The final contraction lives in TransformerBlockSubmodules.hc_head_contraction,
    # filled with Qwen38NextHCHeadContraction by the spec, so its parameters are
    # under decoder.hc_head_contraction.
    _DIRECT_MAPPING.update(
        {
            "decoder.hc_head_contraction.hc_norm_weight": "model.language_model.hyper_connection_mixer.hc_norm.weight",
            "decoder.hc_head_contraction.input_mix_weight_down": "model.language_model.hyper_connection_mixer.input_mix_weight_down.weight",
            "decoder.hc_head_contraction.input_mix_weight_up": "model.language_model.hyper_connection_mixer.input_mix_weight_up.weight",
        }
    )

    _ATTENTION_MAPPING = Qwen3_5Bridge._ATTENTION_MAPPING.copy()

    # Qwen3.5 fuses the pre-attention norm into linear_qkv, producing
    # self_attention.linear_qkv.layer_norm_weight. There is nothing to fill it
    # with here, and the spec correspondingly stops asking TE to fuse it.
    _ATTENTION_MAPPING.pop("self_attention.linear_qkv.layer_norm_weight", None)
    _ATTENTION_MAPPING.pop("self_attention.input_layernorm.weight", None)

    _ATTENTION_MAPPING.update(
        {
            "self_attention.indexer.index_qk_proj.weight": [
                "model.language_model.layers.{layer_number}.self_attn.indexer.index_qk_proj.weight"
            ],
            # Plain nn.Parameters on our indexer; HF wraps each in a submodule.
            "self_attention.indexer.q_layernorm": [
                "model.language_model.layers.{layer_number}.self_attn.indexer.q_layernorm.weight"
            ],
            "self_attention.indexer.k_layernorm": [
                "model.language_model.layers.{layer_number}.self_attn.indexer.k_layernorm.weight"
            ],
        }
    )

    _MLP_MAPPING = Qwen3_5Bridge._MLP_MAPPING.copy()

    _MLP_MAPPING.pop("mlp.linear_fc1.layer_norm_weight", None)
    _MLP_MAPPING.pop("pre_mlp_layernorm", None)

    # The mcore names come from TransformerLayerSubmodules' slot names --
    # self_attention_hyper_connection and mlp_hyper_connection -- while HF calls the
    # first one attn_hyper_connection. Only the attention side differs, which is why
    # mlp_hyper_connection looks like an identity mapping.
    _OTHER_MAPPING = {
        "self_attention_hyper_connection.hc_norm_weight": [
            "model.language_model.layers.{layer_number}.attn_hyper_connection.hc_norm.weight"
        ],
        "self_attention_hyper_connection.input_mix_weight_down": [
            "model.language_model.layers.{layer_number}.attn_hyper_connection.input_mix_weight_down.weight"
        ],
        "self_attention_hyper_connection.input_mix_weight_up": [
            "model.language_model.layers.{layer_number}.attn_hyper_connection.input_mix_weight_up.weight"
        ],
        "self_attention_hyper_connection.block_inject_weight": [
            "model.language_model.layers.{layer_number}.attn_hyper_connection.block_inject_weight.weight"
        ],
        "mlp_hyper_connection.hc_norm_weight": [
            "model.language_model.layers.{layer_number}.mlp_hyper_connection.hc_norm.weight"
        ],
        "mlp_hyper_connection.input_mix_weight_down": [
            "model.language_model.layers.{layer_number}.mlp_hyper_connection.input_mix_weight_down.weight"
        ],
        "mlp_hyper_connection.input_mix_weight_up": [
            "model.language_model.layers.{layer_number}.mlp_hyper_connection.input_mix_weight_up.weight"
        ],
        "mlp_hyper_connection.block_inject_weight": [
            "model.language_model.layers.{layer_number}.mlp_hyper_connection.block_inject_weight.weight"
        ],
        "self_attention_hyper_connection.ple.key_proj.weight": [
            "model.language_model.layers.{layer_number}.ple.key_proj.weight"
        ],
        "self_attention_hyper_connection.ple.value_proj.weight": [
            "model.language_model.layers.{layer_number}.ple.value_proj.weight"
        ],
        # HF wraps each of these in a submodule holding a .weight; on our side they
        # are plain nn.Parameters, so the names differ by that one level.
        "self_attention_hyper_connection.ple.conv1d_weight": [
            "model.language_model.layers.{layer_number}.ple.conv1d.weight"
        ],
        "self_attention_hyper_connection.ple.norm_conv": [
            "model.language_model.layers.{layer_number}.ple.norm_conv.weight"
        ],
        "self_attention_hyper_connection.ple.norm_key": [
            "model.language_model.layers.{layer_number}.ple.norm_key.weight"
        ],
        "self_attention_hyper_connection.ple.norm_query": [
            "model.language_model.layers.{layer_number}.ple.norm_query.weight"
        ],
        # The n-gram table and its hash metadata are deliberately absent here: the
        # table is a non-persistent buffer read straight from the HF safetensors on
        # first use (so it never enters the torch_dist checkpoint and never needs
        # resharding when TP changes), and load_from_hf pulls the three metadata
        # tensors along with it.
    }

    def _ple_layer_ids(self) -> list[int]:
        """0-based decoder layer indices carrying PLE.

        ``ple_layer_ids`` in the HF config is **1-based**: sglang's
        ``Qwen4ExpTextConfig`` computes ``{int(i) - 1 for i in ple_layer_ids}``
        and its model checks ``(layer_id + 1) in config.ple_layer_ids``. The
        released config says ``[2]`` and the checkpoint carries the tensors under
        ``layers.1``, which agrees. Off by one here silently moves PLE to the
        wrong layer: the shapes are identical, so nothing would complain.
        """
        text_config = self._get_text_config()
        return sorted({int(i) - 1 for i in getattr(text_config, "ple_layer_ids", None) or []})

    def _ngram_rows_per_shard(self) -> int | None:
        """Height of one n-gram shard, read from the checkpoint's tensor shapes.

        Returns None when the index is unavailable (from-scratch init), in which
        case the embedding module falls back to ceil(total_rows / num_shards).
        """
        layer_ids = self._ple_layer_ids()
        if not layer_ids:
            return None
        key = f"model.language_model.layers.{layer_ids[0]}.ple.ple_embedding" ".ngram_embedding.shard_0.weight"
        try:
            shape = self.safetensor_io.get_tensor_shape(key)
        except Exception:
            return None
        return int(shape[0]) if shape else None

    def _weight_name_mapping_other(self, mcore_weights_name: str) -> list[str]:
        """HC / indexer / PLE names, including the sharded n-gram embedding.

        The n-gram table ships as ``split_ngram_parts`` separate tensors
        (128 x [2500012, 160] = 102 GB, ~31% of the checkpoint), so it is expanded
        here rather than written out as 128 dict entries.
        """
        layer_number = None
        name = mcore_weights_name
        # A GPTModel prefixes its block with "decoder."; a bare TransformerBlock
        # (as built by the audit and timing harnesses) does not. Accept either --
        # the remaining keys are unambiguous, so this cannot mask a real prefix bug.
        if name.startswith("decoder."):
            name = name[len("decoder.") :]
        if name.startswith("layers."):
            parts = name.split(".")
            layer_number = int(parts[1])
            name = ".".join(parts[2:])

        if name in self._OTHER_MAPPING:
            if layer_number is None:
                raise NotImplementedError(f"{mcore_weights_name} needs a layer index")
            return [t.format(layer_number=layer_number) for t in self._OTHER_MAPPING[name]]

        raise NotImplementedError(f"Unsupported parameter name: {mcore_weights_name}")

    def _weight_name_mapping_mcore_to_hf(self, mcore_weights_name: str) -> list[str]:
        try:
            return super()._weight_name_mapping_mcore_to_hf(mcore_weights_name)
        except NotImplementedError:
            return self._weight_name_mapping_other(mcore_weights_name)

    # Megatron's grouped-GEMM MoE materialises experts as individual parameters --
    # linear_fc1.weight0 .. weight511 -- while the checkpoint stores two fused
    # tensors per layer. mbridge's load loop is per mcore parameter, so for a
    # 512-expert model it runs ~1046 times per layer (~50,000 for 48 layers), and
    # every one of those iterations resolves to the *same* fused HF tensor and hands
    # it to _weight_to_mcore_format to slice one expert out of.
    #
    # Touching the whole fused tensor once per expert is what made the load take
    # hours: gate_up_proj is [512, 1280, 2560] = 3.36 GB, and moving it to the device
    # costs ~0.4 s, so 512 experts is ~205 s of pure transfer per layer -- 3.4
    # min/layer, which is exactly what was measured. Doing that work on the CPU
    # instead (the unpatched path) is the same cost in a different place.
    #
    # So cache the moved tensor by HF name. One layer's 512 experts then share a
    # single upload. The cache holds two entries because a layer has two fused
    # expert tensors (gate_up and down) and the loop alternates between them.
    _GPU_CACHE_SIZE = 2

    def _weight_to_mcore_format(self, mcore_weights_name: str, hf_weights: list[torch.Tensor]):
        """Slice on the device, reusing one upload across a layer's experts."""
        if not torch.cuda.is_available():
            return super()._weight_to_mcore_format(mcore_weights_name, hf_weights)

        cache = getattr(self, "_gpu_weight_cache", None)
        if cache is None:
            cache = self._gpu_weight_cache = {}

        moved = []
        for w in hf_weights:
            # data_ptr identifies the mmap-backed source, and is stable for repeated
            # reads of the same tensor out of the same safetensors file.
            key = (w.data_ptr(), tuple(w.shape), w.dtype)
            hit = cache.get(key)
            if hit is None:
                if len(cache) >= self._GPU_CACHE_SIZE:
                    cache.pop(next(iter(cache)))
                hit = w.to(torch.cuda.current_device(), non_blocking=False)
                cache[key] = hit
            moved.append(hit)
        return super()._weight_to_mcore_format(mcore_weights_name, moved)

    def load_weights(self, model, *args, **kwargs):
        """Memoise state_dict() for the load, then drop the device cache.

        mbridge's loop does ``param = model.state_dict()[local_name]`` once per
        parameter (bridge.py:199). state_dict() is O(number of parameters), so
        inside an n-iteration loop the load is O(n^2) -- and n is not the ~35 per
        layer the checkpoint suggests but 512 experts x 2, about 1046 per layer, so
        ~50,000 for 48 layers. Measured on a 4-layer model: 23.4 s of the 34 s load
        is mbridge internals dominated by this, which scales to roughly an hour at
        full size.

        Caching is safe here because the load only ever writes *into* existing
        parameters (``param.copy_``); the parameter set does not change, and the
        cached dict holds the same tensor objects, so in-place writes are visible
        through it.
        """
        chunks = model if isinstance(model, (list, tuple)) else [model]
        originals = []
        for chunk in chunks:
            cached = chunk.state_dict()
            originals.append((chunk, chunk.state_dict))
            chunk.state_dict = lambda *a, _c=cached, **k: _c
        try:
            return super().load_weights(model, *args, **kwargs)
        finally:
            for chunk, orig in originals:
                try:
                    del chunk.state_dict
                except AttributeError:
                    chunk.state_dict = orig
            self._gpu_weight_cache = {}
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def _build_config(self):
        text_config = self._get_text_config()
        config = super()._build_config()

        # enable_hyper_connections is what makes get_gpt_decoder_block_spec emit
        # HyperConnectionTransformerLayer and fill the two HC ModuleSpec slots; the
        # spec function then swaps HyperConnectionModule for Qwen3.8-Next's gating.
        config.enable_hyper_connections = True
        config.num_residual_streams = getattr(text_config, "hc_count", 4)
        config.qwen3_8_next_hc_lowrank = getattr(text_config, "hc_lowrank", 320)

        config.qwen3_8_next_ple_layer_ids = self._ple_layer_ids()
        config.qwen3_8_next_ple_embed_dim = getattr(text_config, "ple_embed_dim", 2560)
        config.qwen3_8_next_ngram_size = getattr(text_config, "ngram_size", 3)
        config.qwen3_8_next_heads_per_ngram = getattr(text_config, "heads_per_ngram", 8)
        config.qwen3_8_next_ngram_vocab_size_base = getattr(text_config, "ngram_vocab_size_base", 20000000)
        config.qwen3_8_next_split_ngram_parts = getattr(text_config, "split_ngram_parts", 128)
        config.qwen3_8_next_ple_conv_kernel_size = getattr(text_config, "ple_conv_kernel_size", 4)
        config.qwen3_8_next_ple_conv_dilation = getattr(text_config, "ple_conv_dilation", 3)
        # The n-gram hash resets at EOS so n-grams never straddle a document.
        config.qwen3_8_next_eos_token_id = getattr(text_config, "eos_token_id", 0)
        # Shard height is not derivable from the config: the checkpoint rounds the
        # 320,001,446 hashed rows up to 128 x 2,500,012, so the last shard carries
        # padding. Read it off the checkpoint instead of recomputing a ceil.
        config.qwen3_8_next_ngram_rows_per_shard = self._ngram_rows_per_shard()

        config.qwen3_8_next_indexer_budget = getattr(text_config, "indexer_budget", 2048)
        config.qwen3_8_next_indexer_compress_ratio = getattr(text_config, "indexer_compress_ratio", 4)
        config.qwen3_8_next_indexer_n_heads = getattr(text_config, "indexer_n_heads", 4)
        config.qwen3_8_next_indexer_head_dim = getattr(text_config, "indexer_head_dim", 128)
        config.qwen3_8_next_indexer_kv_heads = getattr(text_config, "indexer_kv_heads", 1)

        # hc_norm is the pre-block norm and the final mixer's hc_norm is the final
        # norm, so nothing is left for TE's fused LayerNormColumnParallelLinear or
        # for pre_mlp_layernorm to load. The spec reads this to swap both out.
        config.qwen3_8_next_no_block_layernorms = True

        return config
