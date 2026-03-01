# WARNING: Do NOT relax any assert logic in this file. All assertions must remain strict.
# The comparator must report all-passed with zero failures — no exceptions.
# NOTE: Dump raw tensors as-is — never reshape/squeeze/flatten before dumping.
# The comparator is responsible for alignment and shape reconciliation.

import os
import subprocess
import sys
from pathlib import Path

import torch

# ---------------------------------------------------------------------------
# Shared source patcher configs (used by test_dumper.py and test_run_megatron.py)
# ---------------------------------------------------------------------------

SOURCE_PATCHED_FIELDS: list[str] = [
    "layer_input",
    "attn_output",
    "attn_q",
    "attn_k",
    "attn_v",
    "attn_pre_o_proj",
    "pre_mlp_residual",
    "pre_mlp_layernorm_output",
    "mlp_output",
    "moe_router_logits",
    "moe_expert_output",
]

# Megatron replicated axes: axes that are active (size > 1) but not sharded
# for most tensors.  Declared once so every dims string stays concise.
_MEG_REPL = "tp:replicated ep:replicated pp:replicated etp:replicated"

MEGATRON_SOURCE_PATCHER_CONFIG_YAML: str = """\
patches:
  - target: megatron.core.transformer.transformer_layer.TransformerLayer._forward_attention
    edits:
      - match: |
          inference_context = deprecate_inference_params(inference_context, inference_params)
        append: "dumper.dump('layer_input', hidden_states, dims='t[cp:zigzag,sp] 1 h # """ + _MEG_REPL + """')"
      - match: "nvtx_range_pop(suffix=\\"self_attention\\")"
        append: "dumper.dump('attn_output', attention_output_with_bias[0], dims='t[cp:zigzag,sp] 1 h # """ + _MEG_REPL + """')"
  - target: megatron.core.transformer.transformer_layer.TransformerLayer._forward_mlp
    edits:
      - match: "residual = hidden_states"
        append: "dumper.dump('pre_mlp_residual', residual, dims='t[cp:zigzag,sp] 1 h # """ + _MEG_REPL + """')"
      - match: "pre_mlp_layernorm_output = self.pre_mlp_layernorm(hidden_states)"
        append: "dumper.dump('pre_mlp_layernorm_output', pre_mlp_layernorm_output, dims='t[cp:zigzag,sp] 1 h # """ + _MEG_REPL + """')"
      - match: "return self._forward_post_mlp(mlp_output_with_bias, residual)"
        prepend: "dumper.dump('mlp_output', mlp_output_with_bias[0], dims='t[cp:zigzag,sp] 1 h # """ + _MEG_REPL + """')"

  # --- attention internals ---
  - target: megatron.core.transformer.attention.Attention.forward
    edits:
      - match: "nvtx_range_pop(suffix=\\"rotary_pos_emb\\")"
        append: |
          dumper.dump('attn_q', query, dims='t[cp:zigzag,sp] num_heads[tp] head_dim # ep:replicated pp:replicated etp:replicated')
          dumper.dump('attn_k', key, dims='t[cp:zigzag,sp] num_kv_heads[tp] head_dim # ep:replicated pp:replicated etp:replicated')
          dumper.dump('attn_v', value, dims='t[cp:zigzag,sp] num_kv_heads[tp] head_dim # ep:replicated pp:replicated etp:replicated')
      - match: "nvtx_range_push(suffix=\\"linear_proj\\")"
        prepend: "dumper.dump('attn_pre_o_proj', core_attn_out, dims='t[cp:zigzag,sp] 1 (num_heads*head_dim)[tp] # ep:replicated pp:replicated etp:replicated')"

  # --- moe internals ---
  - target: megatron.core.transformer.moe.router.TopKRouter.forward
    edits:
      - match: "logits = self.gating(input)"
        append: "dumper.dump('moe_router_logits', logits, dims='t[cp:zigzag,sp] 1 num_experts # """ + _MEG_REPL + """')"

  - target: megatron.core.transformer.moe.moe_layer.MoELayer.routed_experts_compute
    edits:
      - match: "expert_output, mlp_bias = self.experts(dispatched_input, tokens_per_expert, permuted_probs)"
        append: "dumper.dump('moe_expert_output', expert_output, dims='t h[tp:partial] # ep:replicated pp:replicated etp:replicated')"
"""

MEGATRON_SOURCE_PATCHER_CONFIG_BSHD_YAML: str = """\
patches:
  - target: megatron.core.transformer.transformer_layer.TransformerLayer._forward_attention
    edits:
      - match: |
          inference_context = deprecate_inference_params(inference_context, inference_params)
        append: "dumper.dump('layer_input', hidden_states, dims='s[cp:zigzag,sp] b h # """ + _MEG_REPL + """')"
      - match: "nvtx_range_pop(suffix=\\"self_attention\\")"
        append: "dumper.dump('attn_output', attention_output_with_bias[0], dims='s[cp:zigzag,sp] b h # """ + _MEG_REPL + """')"
  - target: megatron.core.transformer.transformer_layer.TransformerLayer._forward_mlp
    edits:
      - match: "residual = hidden_states"
        append: "dumper.dump('pre_mlp_residual', residual, dims='s[cp:zigzag,sp] b h # """ + _MEG_REPL + """')"
      - match: "pre_mlp_layernorm_output = self.pre_mlp_layernorm(hidden_states)"
        append: "dumper.dump('pre_mlp_layernorm_output', pre_mlp_layernorm_output, dims='s[cp:zigzag,sp] b h # """ + _MEG_REPL + """')"
      - match: "return self._forward_post_mlp(mlp_output_with_bias, residual)"
        prepend: "dumper.dump('mlp_output', mlp_output_with_bias[0], dims='s[cp:zigzag,sp] b h # """ + _MEG_REPL + """')"

  # --- attention internals ---
  - target: megatron.core.transformer.attention.Attention.forward
    edits:
      - match: "nvtx_range_pop(suffix=\\"rotary_pos_emb\\")"
        append: |
          dumper.dump('attn_q', query, dims='s[cp:zigzag,sp] b num_heads[tp] head_dim # ep:replicated pp:replicated etp:replicated')
          dumper.dump('attn_k', key, dims='s[cp:zigzag,sp] b num_kv_heads[tp] head_dim # ep:replicated pp:replicated etp:replicated')
          dumper.dump('attn_v', value, dims='s[cp:zigzag,sp] b num_kv_heads[tp] head_dim # ep:replicated pp:replicated etp:replicated')
      - match: "nvtx_range_push(suffix=\\"linear_proj\\")"
        prepend: "dumper.dump('attn_pre_o_proj', core_attn_out, dims='s[cp:zigzag,sp] b (num_heads*head_dim)[tp] # ep:replicated pp:replicated etp:replicated')"

  # --- moe internals ---
  - target: megatron.core.transformer.moe.router.TopKRouter.forward
    edits:
      - match: "logits = self.gating(input)"
        append: "dumper.dump('moe_router_logits', logits, dims='s[cp:zigzag,sp] b num_experts # """ + _MEG_REPL + """')"

  - target: megatron.core.transformer.moe.moe_layer.MoELayer.routed_experts_compute
    edits:
      - match: "expert_output, mlp_bias = self.experts(dispatched_input, tokens_per_expert, permuted_probs)"
        append: "dumper.dump('moe_expert_output', expert_output, dims='t h[tp:partial] # ep:replicated pp:replicated etp:replicated')"
"""

SGLANG_SOURCE_PATCHER_CONFIG_YAML: str = """\
patches:
  - target: sglang.srt.models.qwen3_moe.Qwen3MoeDecoderLayer.forward
    edits:
      - match: |
          hidden_states, residual = (
              self.layer_communicator.prepare_attn_and_capture_last_layer_outputs(
                  hidden_states,
                  residual,
                  forward_batch,
                  captured_last_layer_outputs=captured_last_layer_outputs,
                  **kwargs,
              )
          )
        append: "dumper.dump('layer_input', residual, dims='t h # tp:replicated')"
      - match: |
          if hidden_states.shape[0] != 0:
              hidden_states = self.self_attn(
                  positions=positions,
                  hidden_states=hidden_states,
                  forward_batch=forward_batch,
              )
        append: "dumper.dump('attn_output', hidden_states, dims='t h[tp:partial]')"
      - match: |
          hidden_states, residual = self.layer_communicator.prepare_mlp(
              hidden_states, residual, forward_batch
          )
        append: |
          dumper.dump('pre_mlp_residual', residual, dims='t h # tp:replicated')
          dumper.dump('pre_mlp_layernorm_output', hidden_states, dims='t h # tp:replicated')
      - match: |
          hidden_states = self.mlp(
              hidden_states, forward_batch, should_allreduce_fusion, use_reduce_scatter
          )
        append: "dumper.dump('mlp_output', hidden_states, dims='t h[tp:partial]')"

  # --- attention internals ---
  - target: sglang.srt.models.qwen3_moe.Qwen3MoeAttention.forward_core
    edits:
      - match: |
          attn_output = self.attn(
        prepend: |
          dumper.dump('attn_q', q, dims='t (num_heads*head_dim)[tp]')
          dumper.dump('attn_k', k, dims='t (num_kv_heads*head_dim)[tp]')
          dumper.dump('attn_v', v, dims='t (num_kv_heads*head_dim)[tp]')
      - match: "output, _ = self.o_proj(attn_output)"
        prepend: "dumper.dump('attn_pre_o_proj', attn_output, dims='t (num_heads*head_dim)[tp]')"

  # --- moe internals ---
  - target: sglang.srt.models.qwen3_moe.Qwen3MoeSparseMoeBlock.forward_normal
    edits:
      - match: "router_logits, _ = self.gate(hidden_states)"
        append: "dumper.dump('moe_router_logits', router_logits, dims='t num_experts # tp:replicated')"
      - match: "final_hidden_states = self.experts(hidden_states, topk_output)"
        append: "dumper.dump('moe_expert_output', final_hidden_states, dims='t h[tp:partial]')"
"""


def clear_proxy_env() -> None:
    for proxy_var in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        os.environ.pop(proxy_var, None)


def check_dump_dir(
    phase_dir: Path,
    exp_pattern: str,
    expected_fields: list[str] | None = None,
) -> None:
    """Verify dump directory structure: .pt files exist, contain value + meta keys, contain expected fields."""
    assert phase_dir.exists(), f"Missing dump dir: {phase_dir}"
    dump_subdirs: list[Path] = list(phase_dir.glob(exp_pattern))
    assert len(dump_subdirs) > 0, f"No {exp_pattern} subdirs in {phase_dir}"
    dump_files: list[Path] = list(dump_subdirs[0].glob("*.pt"))
    assert len(dump_files) > 0, f"No .pt files in {dump_subdirs[0]}"
    sample: dict = torch.load(dump_files[0], weights_only=False)
    assert isinstance(sample, dict), f"Unexpected type: {type(sample)}"
    assert "value" in sample and "meta" in sample, f"Missing keys: {sample.keys()}"

    if expected_fields:
        for field in expected_fields:
            matches: list[Path] = list(phase_dir.rglob(f"*name={field}*.pt"))
            assert len(matches) > 0, f"Expected field '{field}' not found under {phase_dir}"


def log_comparator_output(stdout: str, stderr: str) -> None:
    if stdout.strip():
        print(f"[comparator stdout]\n{stdout}")
    if stderr.strip():
        print(f"[comparator stderr]\n{stderr}")


def run_and_verify_comparator(
    baseline_dir: Path,
    target_dir: Path,
    extra_args: list[str] | None = None,
) -> None:
    """Run comparator subprocess and rely on its exit code (--allow-skip-pattern limits allowed skips)."""
    cmd: list[str] = [
        sys.executable,
        "-m",
        "sglang.srt.debug_utils.comparator",
        "--baseline-path",
        str(baseline_dir),
        "--target-path",
        str(target_dir),
        "--output-format",
        "json",
        "--preset",
        "sglang_megatron",
        "--allow-skipped-pattern",
        "input_ids|positions|cu_seqlens_q|cu_seqlens_kv|qkv_format",
    ]
    if extra_args:
        cmd.extend(extra_args)

    result: subprocess.CompletedProcess[str] = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
    )

    log_comparator_output(stdout=result.stdout, stderr=result.stderr)

    assert result.returncode == 0, f"Comparator failed (rc={result.returncode})\nstderr: {result.stderr[-2000:]}"
