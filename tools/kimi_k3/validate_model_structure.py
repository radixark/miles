import os
from types import SimpleNamespace

import torch
import torch.distributed as dist
from megatron.core import parallel_state
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed

import miles_plugins.mbridge  # noqa: F401
from mbridge.core.parallel_states import ParallelStates
from miles_plugins.mbridge.kimi_k3 import KimiK3Bridge


def make_hf_config() -> SimpleNamespace:
    text_config = SimpleNamespace(
        activation_situ_beta=4.0,
        activation_situ_linear_beta=25.0,
        attn_res_block_size=2,
        first_k_dense_replace=1,
        head_dim=32,
        hidden_size=128,
        intermediate_size=256,
        kv_lora_rank=16,
        latent_moe_use_norm=True,
        linear_attn_config={
            "full_attn_layers": [4],
            "gate_lower_bound": -5.0,
            "head_dim": 16,
            "kda_layers": [1, 2, 3],
            "num_heads": 4,
            "short_conv_kernel_size": 4,
            "use_full_rank_gate": True,
        },
        max_position_embeddings=1024,
        mla_use_nope=True,
        mla_use_output_gate=True,
        moe_intermediate_size=64,
        moe_layer_freq=1,
        moe_renormalize=True,
        moe_router_activation_func="sigmoid",
        num_attention_heads=4,
        num_experts=8,
        num_experts_per_token=2,
        num_hidden_layers=4,
        num_key_value_heads=4,
        num_shared_experts=1,
        q_lora_rank=32,
        qk_nope_head_dim=16,
        qk_rope_head_dim=8,
        rms_norm_eps=1e-5,
        routed_expert_hidden_size=64,
        routed_scaling_factor=1.0,
        tie_word_embeddings=False,
        v_head_dim=16,
        vocab_size=256,
    )
    return SimpleNamespace(model_type="kimi_k3", text_config=text_config)


def main() -> None:
    torch.cuda.set_device(0)
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29500")
    dist.init_process_group("nccl", rank=0, world_size=1)
    parallel_state.initialize_model_parallel(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        context_parallel_size=1,
        expert_model_parallel_size=1,
        expert_tensor_parallel_size=1,
    )
    model_parallel_cuda_manual_seed(1234)

    mpu = ParallelStates(
        tp_size=1,
        tp_rank=0,
        pp_size=1,
        pp_rank=0,
        vpp_size=None,
        cp_size=1,
        cp_rank=0,
        ep_size=1,
        ep_rank=0,
        etp_size=1,
        etp_rank=0,
        tp_group=parallel_state.get_tensor_model_parallel_group(),
        pp_group=parallel_state.get_pipeline_model_parallel_group(),
        cp_group=parallel_state.get_context_parallel_group(),
        ep_group=parallel_state.get_expert_model_parallel_group(),
        etp_group=parallel_state.get_expert_tensor_parallel_group(),
    )
    bridge = KimiK3Bridge(make_hf_config(), parallel_states=mpu)
    model = GPTModel(
        config=bridge.config,
        transformer_layer_spec=bridge._get_transformer_layer_spec(),
        vocab_size=bridge.text_config.vocab_size,
        max_sequence_length=bridge.text_config.max_position_embeddings,
        position_embedding_type="none",
        share_embeddings_and_output_weights=False,
    )

    local_to_global = bridge._weight_name_mapping_mcore_local_to_global(model)
    mapped = {}
    for local_name, global_name in local_to_global.items():
        mapped[local_name] = bridge._weight_name_mapping_mcore_to_hf(global_name)
    assert len(mapped) == len(local_to_global)

    print(f"parameters={sum(parameter.numel() for parameter in model.parameters())}")
    print(f"mapped_tensors={len(mapped)}")
    for layer in model.decoder.layers:
        print(
            f"layer={layer.layer_number - 1} attention={type(layer.self_attention).__name__} "
            f"is_kda={layer.self_attention.is_kda} mlp={type(layer.mlp).__name__}"
        )

    model.train()
    input_ids = torch.randint(0, bridge.text_config.vocab_size, (2, 16), device="cuda")
    position_ids = torch.arange(16, device="cuda").unsqueeze(0).expand(2, -1)
    logits = model(input_ids, position_ids, attention_mask=None)
    assert torch.isfinite(logits).all()
    loss = logits.float().square().mean()
    loss.backward()
    print(f"forward_shape={tuple(logits.shape)} loss={loss.item():.6f}")

    parallel_state.destroy_model_parallel()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
