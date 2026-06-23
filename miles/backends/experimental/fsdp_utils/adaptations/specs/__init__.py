"""Per-arch adaptation specs — the one place a new architecture plugs into the FSDP backend.

Importing this package registers every arch's hooks across the mechanism registries. Each
``<arch>.py`` module declares everything that arch needs; the mechanism modules (weight_bridge,
class_patches, packing, post_load_fixups, precision) never change.

To add a new architecture, create ``specs/<arch>.py`` that registers only the hooks it needs, and
add it to the import line at the bottom. The available registrations (import each from ``..<module>``):

  * register_param_transform(model_type, matches, expand)      [weight_bridge]
        Train->rollout param rename/reshape, applied at weight sync. Use when the training param
        names/shapes differ from what the sglang loader expects (e.g. transformers>=5.6 batches MoE
        experts into one tensor but sglang wants per-expert names). matches(name, param)->bool,
        expand(name, full_tensor)->Iterable[(name, tensor)]. The shared batched-MoE-expert split lives
        in weight_bridge as ``_qwen3_moe_matches`` / ``_qwen3_moe_expand`` and is reused by qwen3_moe
        and glm4_moe_lite.

  * register_model_patch(ModelPatchHook(name, applies_to, apply))   [class_patches]
        Config-time patch of the transformers classes, applied BEFORE construction. applies_to(cfg)
        ->bool, apply(cfg, args). Use to monkeypatch a class forward (e.g. the qwen3_moe MoE block).

  * register_packing_patch(PackingPatch(name, applies_to, lifetime, apply))   [packing.registry]
        Packed-sequence (THD) per-document state reset. lifetime "config" patches classes before
        construction (apply()); "post_load" patches the instantiated model (apply(model)). Use for
        stateful attention/SSM (GatedDeltaNet, Mamba2) that must reset per packed document.

  * register_post_load_fixup(PostLoadFixup(name, applies_to, apply))   [post_load_fixups]
        Post-load weight correction. apply(model, ckpt_path)->int. Use when from_pretrained corrupts
        loaded weights (e.g. NemotronH _init_weights re-inits Mamba params after loading).

  * register_fp32_master_type(model_type)   [precision]
        Keep an fp32 master copy of the weights. Use when the FSDP2 bf16 reshard perturbs weights
        enough to open a train<->rollout logprob gap (glm4_moe_lite).

An arch that needs none of these (a dense model whose names/shapes/precision already match) registers
nothing. See the four existing specs for worked examples.
"""

from . import glm4_moe_lite, nemotron_h, qwen3_5_moe, qwen3_moe  # noqa: F401  (imports trigger registration)
