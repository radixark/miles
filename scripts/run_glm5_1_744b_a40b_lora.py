"""
GLM-5.1 744B-A40B GRPO LoRA training script (Megatron-Bridge / bridge mode).

Per-model split of ``scripts/run_glm5_lora.py`` following the PR #1376 convention
(one run script per model + a dedicated ``scripts/models/*.sh`` registry). This file
covers GLM-5.1 only; GLM-5.2 lives in ``scripts/run_glm5_2_744b_a40b_lora.py``.
The model-name -> megatron_model_type mapping points at the LoRA registries:
  GLM-5.1        -> scripts/models/glm5.1-744B-A40B_lora.sh
  GLM-5.1-6layer -> scripts/models/glm5.1-744B-A40B_6layer_lora.sh

NOTE (intentional drop vs run_glm5_lora.py): the GLM-5.1-4layer / GLM-5.1-20layer prune
entries are NOT carried over into this split. Their base MODEL_ARGS registries survive as
scripts/models/glm5-744B-A40B_4layer.sh / glm5-744B-A40B_20layer.sh (still usable via the
combined scripts/run_glm5_lora.py); resurrect them here by adding the model-name entries
plus matching *_lora.sh registries if they are ever needed on this path.

GLM-5.1 is MoE + MLA + DSA (DeepSeek Sparse Attention). LoRA trains through the
Megatron-Bridge path (``--megatron-to-hf-mode bridge``); the "dsa" experimental-attention
spec is provided by the Megatron-Bridge GLM5 provider, so NO ``--spec`` is consumed here
(the registry ``.sh`` may still list it; it is provably inert under bridge LoRA).

Unlike GLM-5.2, GLM-5.1 has NO DSA cross-layer index sharing (every layer carries its own
indexer); the ``glm5.1-744B-A40B*_lora`` registries are identical to the GLM-5.2 ones except
``--rotary-base`` 1e6 (5.2 uses 8e6). sglang serves GLM-5.1 natively, so GLM-5.1 is also the
dump SOURCE for the GLM-5.2 train-only replay workflow (see the GLM-5.2 sibling's docstring).

Modeled on ``scripts/run_deepseek_v4.py`` (typer app + ScriptArgs(ExecuteTrainConfig)
+ model-name -> megatron_model_type registry + ``_get_parallel_config`` + ``U.execute_train``).

DSA kernel backend (``--dsa-attention-backend``; bridge path only). Orthogonal to model version
and to LoRA -- BOTH backends run GLM-5.1 *and* GLM-5.2, full or LoRA:
  * ``glm-native`` (default): the vendored fused TileLang kernels (SparseMLA + lighting_indexer),
    matching slime's rollout kernels for rollout<->train numerical parity (incl. R3 indexer
    replay). Needs the optional ``tilelang`` dep and the ``thd`` (packed) layout.
    Training/forward-only (no KV cache): it cannot serve generation -- the rollout is always
    served by sglang.
  * ``megatron-bridge-native``: the portable unfused megatron-core DSA kernels (DSAttention /
    CrossLayerDSAttention). No extra deps. Uses the ``bshd`` query layout.
  This launcher selects the matching ``--qkv-format`` automatically from the backend (see
  ``_get_parallel_config``); just pass ``--dsa-attention-backend megatron-bridge-native`` or leave the
  default. See the Megatron-Bridge ``models/glm5/__init__.py`` docstring for the full
  backend matrix.

Two DSA specifics:
  * ``--target-modules`` excludes the 3 DSA indexer modules (wq_b/wk/weights_proj) by default --
    the indexer stays a frozen base capability; this run does not train it. On the glm-native backend the
    indexer adapter gets no gradient anyway (a genuine no-op); on the megatron-bridge-native backend it
    would get a tiny aux-loss gradient (~1e-5), so excluding it there is a deliberate choice.
  * ``--micro-batch-size 1`` (no ``--use-dynamic-batch-size``): both backends pin a static
    micro-batch. The query layout follows the backend -- ``bshd`` (megatron-core's DSA
    core-attention needs a 4D query; the default ``thd`` packing yields a 3D query and raises "not
    enough values to unpack") for ``megatron-bridge-native``, ``thd`` (packed) for ``glm-native``.

Supported model variants (HF checkpoint must be the native config,
model_type=glm_moe_dsa / GlmMoeDsaForCausalLM):
  GLM-5.1          full 744B model (zai-org/GLM-5.1)
  GLM-5.1-6layer   6-layer GLM-5.1 prune (jybsuper/GLM-5.1-6layer; 3 dense + 3 MoE)

Usage (run ON the devbox; miles editable-installed under /personal):
  python scripts/run_glm5_1_744b_a40b_lora.py prepare    --model-name GLM-5.1-6layer   # download model + task dataset (default gsm8k)
  # default (glm-native / fused TileLang) backend:
  python scripts/run_glm5_1_744b_a40b_lora.py full-train --model-name GLM-5.1-6layer --num-gpus-per-node 4
  # unfused megatron-bridge-native backend:
  python scripts/run_glm5_1_744b_a40b_lora.py full-train --model-name GLM-5.1-6layer \\
      --dsa-attention-backend megatron-bridge-native --num-gpus-per-node 4
  # DAPO-Math example (zhuzilin/dapo-math-17k, long-CoT competition math -- use a longer response len):
  python scripts/run_glm5_1_744b_a40b_lora.py prepare --model-name GLM-5.1-6layer --task dapo-math
  python scripts/run_glm5_1_744b_a40b_lora.py train   --model-name GLM-5.1-6layer --task dapo-math \\
      --rollout-max-response-len 4096 --num-gpus-per-node 4   # add --dapo-dynamic-sampling on a real model
  # dump a rollout for the GLM-5.2 train-only replay (both toys share the GLM tokenizer/vocab):
  python scripts/run_glm5_1_744b_a40b_lora.py full-train --model-name GLM-5.1-6layer \\
      --extra-args "--dump-details /personal/dump51"
"""

from dataclasses import dataclass
from typing import Literal

import os

import typer

import miles.utils.external_utils.command_utils as U

app = typer.Typer()

# HF repos to download from (full model from zai-org; pruned toy from jybsuper).
_HF_REPO = {
    "GLM-5.1": "zai-org/GLM-5.1",
    "GLM-5.1-6layer": "jybsuper/GLM-5.1-6layer",
}

# LoRA-dedicated registries (scripts/models/<name>.sh); same MODEL_ARGS as the full-FT
# glm5-744B-A40B.sh / glm5.1-744B-A40B_6layer.sh registries plus a documented LoRA section.
_MEGATRON_MODEL_TYPE = {
    "GLM-5.1": "glm5.1-744B-A40B_lora",
    "GLM-5.1-6layer": "glm5.1-744B-A40B_6layer_lora",
}

# Explicit LoRA targets: standard attn + MLA + MLP/MoE, EXCLUDING the DSA indexer
# (wq_b/wk/weights_proj). Set --target-modules all-linear to also cover the indexer.
_DEFAULT_TARGET_MODULES = "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj,q_a_proj,kv_a_proj_with_mqa,q_b_proj,kv_b_proj"


@dataclass
class ScriptArgs(U.ExecuteTrainConfig):
    run_id: str = U.create_run_id()
    model_name: Literal[
        "GLM-5.1",
        "GLM-5.1-6layer",
    ] = "GLM-5.1-6layer"
    # Example task / dataset (math RL, --rm-type math for both):
    #   gsm8k     -> zhuzilin/gsm8k, short-answer grade-school math (~256-tok responses).
    #   dapo-math -> zhuzilin/dapo-math-17k (DAPO-Math-17k), hard long-CoT competition math.
    #                Pass a larger --rollout-max-response-len (e.g. 4096); a >2048 total seq is
    #                also what makes the GLM-5 DSA indexer go genuinely SPARSE, unlike the dense
    #                short gsm8k case. Optionally enable DAPO dynamic sampling (see
    #                dapo_dynamic_sampling below). Mirrors run_deepseek_v4.py.
    task: Literal["gsm8k", "dapo-math"] = "gsm8k"

    hf_checkpoint: str | None = None
    model_dir: str = "/root/models"
    save_dir: str = "/personal/checkpoints"
    data_dir: str = "/root/datasets"
    megatron_path: str = "/root/Megatron-LM"

    # DSA sparse-MLA kernel backend (bridge path only). Orthogonal to model version & LoRA: BOTH
    # backends support GLM-5.1 AND GLM-5.2, full or LoRA.
    #   "glm-native" (default):  vendored fused TileLang SparseMLA + lighting_indexer; thd layout; needs
    #                       tilelang; matches slime's rollout kernels for rollout<->train parity
    #                       (incl. R3 indexer replay); training/forward-only (no KV cache).
    #   "megatron-bridge-native":  portable unfused megatron-core DSA kernels; bshd layout; no extra deps.
    # The matching --qkv-format is chosen from this automatically (see _get_parallel_config).
    dsa_attention_backend: Literal["megatron-bridge-native", "glm-native"] = "glm-native"

    # R3 (rollout routing replay, arxiv 2510.11370): during training, replay the rollout's recorded
    # MoE top-8 so the train-side expert selection matches the rollout (on-policy). Adds ONLY
    # --use-rollout-routing-replay. The DSA indexer top-k replay (--use-rollout-indexer-replay) is
    # NOT added: it is a debug-only parity check (the glm-native kernel recomputes the top-k) and it
    # triggers sglang's ~78-128 GB/rank IndexerTopkCapturer host buffer that OOM'd the colocate pod.
    # See _train r3_args for the full rationale.
    use_r3: bool = True

    # performance
    num_gpus_per_node: int = 4

    # LoRA
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.0
    target_modules: str = _DEFAULT_TARGET_MODULES
    # REQUIRED for true on-policy under colocate: without the sglang host-RAM base-weight mirror
    # (enable_weights_cpu_backup) the base mis-maps across the torch_memory_saver pause/resume and
    # the rollout serves a corrupted base+LoRA -> train<->rollout abs_diff ~1.46 / KL ~1.0 instead
    # of ~0.01 / ~1e-4 (measured on GLM-5.2_5layer; exp_weight harness A/B). Default ON. Opt out
    # (--no-lora-base-cpu-backup) ONLY for full-744B on the glm-native train backend when host RAM is
    # tight: the ~372 GB/node mirror + glm-native-backend init can push the pod past its cgroup memory.max
    # (host OOM -> RolloutManager SIGTERM); trade-off when OFF is the trainer re-ships the base
    # per swap AND the on-policy guarantee is lost.
    lora_base_cpu_backup: bool = True
    # MoE-expert LoRA adapter layout — only takes effect when MoE-expert LoRA is active
    # (KEEP_MOE_LORA=1 and the expert projections are in --target-modules; no-op otherwise).
    #   True  (default): shared-outer grouped-expert layout (--experts-shared-outer-loras;
    #          arguments.py auto-pairs the serve-side --sglang-experts-shared-outer-loras).
    #   False: regular per-expert MoE LoRA (the launch_glm_rl_att_*_moe.sh layout).
    experts_shared_outer_loras: bool = True

    # rollout
    num_rollout: int = 1
    rollout_batch_size: int = 4
    n_samples_per_prompt: int = 4
    # NB: the GLM-5 DSA indexer has index_topk=2048. It performs genuine SPARSE top-k selection
    # only when the full sequence (prompt + response) EXCEEDS 2048; at shorter seq (the gsm8k
    # default below) the indexer degenerates to DENSE (top-k >= seq -> selects all keys). Use a
    # longer prompt/response (> 2048) to hit sparse indexing.
    rollout_max_response_len: int = 0  # 0 => per-task default (gsm8k 512, dapo-math 4096); see __post_init__
    # seq window: emitted as --seq-length + --rollout-max-context-len when > 0; 0 => per-task default
    # in __post_init__. The rollout caps generation at min(response_len, seq_window - prompt), so a
    # window > response_len HARD-bounds prompt+response (and the colocate memory window):
    #   dapo-math -> 8192 (resp 4096 fits with prompt headroom; total can never exceed 8192)
    #   gsm8k     -> 0 (UNSET: 512-tok responses sit well within megatron's default 4096 train window)
    seq_window: int = 0
    global_batch_size: int = 16

    # DAPO dynamic sampling (only used when task="dapo-math") -- DAPO's signature trick: drop
    # prompt-groups whose samples ALL get the same reward (no learning signal), oversampling to
    # refill the batch. OFF by default: on a model that scores 0 on every sample (e.g. the toy
    # pruned checkpoints) it would reject every batch and resample forever. Enable only with a
    # model that actually solves some of the problems.
    dapo_dynamic_sampling: bool = False
    over_sampling_batch_size: int = 32  # used only when dapo_dynamic_sampling; should exceed rollout_batch_size

    # rollout engine
    rollout_num_gpus_per_engine: int = 2  # rollout tp=2
    sglang_mem_fraction_static: float = 0.5
    # sglang LoRA kernel backend (triton|csgmv|ascend|torch_native). sglang's OWN default is csgmv,
    # but csgmv has crashed the GLM-5 DSA MoE-LoRA rollout (gate_up slice miscount under
    # dp-attention -> "scheduler died") and is less robust; triton is the kernel the multinode .sh
    # wrappers already pin. Default it to triton HERE so every entrypoint that flows through this
    # launcher inherits it -- a bare `python run_glm5_1_744b_a40b_lora.py`, the multinode .sh
    # wrappers, and the CI smoke tests all pick it up. Emitted as --sglang-lora-backend; override
    # per-run with --sglang-lora-backend csgmv (or, via the .sh wrappers, SGLANG_LORA_BACKEND=...).
    sglang_lora_backend: str = "triton"
    # fp8 rollout: serve sglang from a pre-converted _fp8 ckpt (tools/convert_hf_to_fp8.py). fp8
    # halves rollout weight mem so the 744B model fits engine=8 (1 node); megatron TRAIN stays bf16
    # (it dequantizes the fp8 HF base via the bridge). Point --hf-checkpoint at the _fp8 dir.
    fp8_rollout: bool = False

    enable_wandb: bool = True
    # pass any extra miles/megatron/sglang args through, e.g. --extra-args '--lora-base-cpu-backup'
    extra_args: str = ""

    def __post_init__(self):
        if self.hf_checkpoint is None:
            # NB: must be a LOCAL path -- miles sets args.load = hf_checkpoint and asserts it
            # is an existing directory (a HF repo id is not accepted here).
            self.hf_checkpoint = f"{self.model_dir}/{self.model_name}"
        if self.rollout_max_response_len == 0:
            # per-task default response budget (the .sh wrappers also set this; this keeps a bare
            # `python run_glm5_1_744b_a40b_lora.py --task ...` correct on its own). NB: no
            # --seq-length / --rollout-max-context-len is set here -- the total seq window
            # auto-derives from the model config; only the generation budget is capped:
            #   gsm8k     -> 512  (short-answer; seq < index_topk 2048 so the DSA indexer is DENSE)
            #   dapo-math -> 4096 (long-CoT; >2048 seq makes the GLM-5 DSA indexer go SPARSE)
            self.rollout_max_response_len = 4096 if self.task == "dapo-math" else 512
        if self.seq_window == 0 and self.task == "dapo-math":
            # dapo response (4096) must leave room for the prompt within the train window, and the
            # colocate window must be bounded -> set an 8192 window (gsm8k stays 0/unset: tiny seq).
            self.seq_window = 8192

    @property
    def megatron_model_type(self) -> str:
        return _MEGATRON_MODEL_TYPE[self.model_name]


def _get_parallel_config(args: ScriptArgs) -> str:
    """Single-node MoE layout: TP = EP = num_gpus_per_node, DP1 (mirrors run_glm5_744b_a40b).

    The DSA kernel backend dictates the query layout; both forbid --use-dynamic-batch-size, hence
    --micro-batch-size 1:
      * megatron-bridge-native (unfused megatron-core DSA core-attention): needs bshd (4D query); the
        default thd packing yields a 3D query and raises "not enough values to unpack".
      * glm-native (fused TileLang SparseMLA + lighting_indexer): needs thd (packed) -- the fused
        kernels index by cu_seqlens and use a [t, heads, dim] layout; bshd has no cu_seqlens.
    """
    ngpu = args.num_gpus_per_node
    # Canonical GLM-5 layout TP=EP=ngpu + sequence-parallel, CP=1 here (the full-scale recipe adds
    # context-parallel + --allgather-cp). BOTH backends run with sequence-parallel: the glm-native fused
    # thd indexer / SparseMLA path is SP/CP-aware in the bridge port -- glm_native_mla.py reconciles
    # index_q/index_k/head_weights via SP(+CP) all-gather and cu_seqlens (starts/ends) via CP-scatter,
    # mirroring native slime glm5.py, so the fused kernel sees matching token dims under SP. The only
    # per-backend difference is the query layout: thd (packed, cu_seqlens-indexed) for glm-native, bshd
    # for the unfused megatron-core core-attention.
    qkv_format = "thd" if args.dsa_attention_backend == "glm-native" else "bshd"
    return (
        f"--tensor-model-parallel-size {ngpu} --sequence-parallel --pipeline-model-parallel-size 1 "
        f"--context-parallel-size 1 --expert-model-parallel-size {ngpu} --expert-tensor-parallel-size 1 "
        f"--qkv-format {qkv_format} --micro-batch-size 1 "
    )


def _download_dataset(args: ScriptArgs):
    match args.task:
        case "gsm8k":
            U.hf_download_dataset("zhuzilin/gsm8k", data_dir=args.data_dir)
        case "dapo-math":
            U.hf_download_dataset("zhuzilin/dapo-math-17k", data_dir=args.data_dir)


def _prepare_download(args: ScriptArgs):
    U.exec_command(f"mkdir -p {args.data_dir} {args.model_dir}")
    repo = _HF_REPO.get(args.model_name)
    if repo is not None:
        U.exec_command(f"hf download {repo} --local-dir {args.model_dir}/{args.model_name}")
    _download_dataset(args)


def _train(args: ScriptArgs):
    print(f"[run] GLM-5.1 LoRA: model={args.model_name} (megatron_model_type={args.megatron_model_type}), dsa-backend={args.dsa_attention_backend}, r3={args.use_r3}, {args.num_gpus_per_node} GPUs, rollout tp={args.rollout_num_gpus_per_engine}")
    load_save_path = f"{args.save_dir}/{args.run_id}"

    ckpt_args = (
        f"--hf-checkpoint {args.hf_checkpoint} --megatron-to-hf-mode bridge "
        f"--dsa-attention-backend {args.dsa_attention_backend} "
    )

    # Full GLM-5 (non-toy) gets the MoE-LoRA rollout mechanism flags (Kimi K2.5 LoRA recipe,
    # MINUS --experts-shared-outer-loras per directive): virtual-experts MoE-LoRA path, frozen
    # base kept on CPU under colocate, no grad-accum-fusion. target-modules stays GLM's own
    # default (the correct GLM MLA superset; NOT Kimi's DeepSeek-named list). The toy keeps
    # the simple path unchanged.
    # Apply the FULL GLM-5 rollout config (dp-attention / nsa / EP + MLP-target drop + MoE-LoRA
    # flags + no reasoning/tool parser) to ALL GLM-5 models INCLUDING the pruned toy (per request).
    # The toy is glm_moe_dsa too, so it exercises the same DSA / dp-attention / LoRA serving
    # path -> a fast (~seconds to load) way to validate the rollout sglang config without the full
    # model's ~15-min load. Engine size / mem-fraction still come from the C0 script per model
    # (toy -> rollout-num-gpus-per-engine 2, mem-frac 0.5; full -> 32, 0.70). Set to
    # `"layer" not in args.model_name` to restore the old simple csgmv toy path.
    _is_full = True
    # (B) Full model: DROP gate_proj/up_proj/down_proj from the rollout LoRA targets. sglang's
    # mem_pool LoRA-B buffer probe (mem_pool.py:359-383) fails to detect the fully-DP dense MLP
    # gate_up (tp=1 under --sglang-moe-dense-tp-size 1) and falls back to the GLOBAL tp_size,
    # sizing B as 24576/tp_size -> "LoRA B output dim != base partition prefix dim 24576" scheduler
    # crash (orthogonal to fp8/engine-size/virtual-experts/max-lora-rank). LoRA only attention
    # (q/k/v/o + MLA q_a/kv_a/q_b/kv_b) until the sglang probe is patched.
    _tm = args.target_modules
    # MoE/MLP LoRA on gate_proj/up_proj/down_proj is KEPT BY DEFAULT now (regular per-expert MoE LoRA;
    # the bridge layout is selected by --experts-shared-outer-loras via lora_utils.create_lora_instance).
    # Set KEEP_MOE_LORA=0 to restore the attention-only drop -- needed ONLY when serving via the sglang
    # colocate rollout, which cannot yet serve MoE-expert LoRA (the dense gate_up "12288 vs 24576"
    # scheduler crash). Training / train-only / dumped-rollout do not need the drop.
    _keep_moe_lora = os.environ.get("KEEP_MOE_LORA", "1") != "0"
    if _is_full and not _keep_moe_lora:
        _tm = ",".join(m for m in _tm.split(",") if m.strip() not in ("gate_proj", "up_proj", "down_proj"))
    # MOE_LORA_LAYERS: restrict the MoE-EXPERT LoRA (MLP gate/up/down -> linear_fc1/linear_fc2) to a
    # SUBSET of layers, to cut the actor_train backward-activation / optimizer memory of the many-layer
    # expert grouped-GEMM (attention LoRA stays on ALL layers). Empty (default) = every layer (current
    # behavior). Accepts ranges and/or comma lists, e.g. "58-77" or "60,65,70". Mechanism: emit
    # Megatron-Bridge ModuleMatcher wildcard patterns "*.layers.<N>.*.linear_fc1/linear_fc2" for ONLY
    # the selected layers and drop the bare gate/up/down (which match every layer). miles passes
    # "*"-patterns through unchanged (convert_target_modules_to_megatron); the bare attention names
    # (q/k/v/o + MLA q_a/kv_a/q_b/kv_b) still map to all-layer attention LoRA.
    # ─────────────────────────────────────────────────────────────────────────────
    # [DEBUG-DISABLED 2026-06-28] MOE_LORA_LAYERS subset-rewrite COMMENTED OUT to rule
    # it out as the source of a suspected bug. While disabled, MoE-expert LoRA always
    # targets ALL MoE layers via the plain module names (gate_proj/up_proj/down_proj);
    # the "*.layers.<N>.*.linear_fc1/linear_fc2" wildcard patterns are NEVER emitted.
    # The env var is still read ONLY to warn if it is set (so it isn't silently ignored).
    # To re-enable: delete the warning shim and un-comment the block below.
    # ─────────────────────────────────────────────────────────────────────────────
    _moe_lora_layers = os.environ.get("MOE_LORA_LAYERS", "").strip()
    if _moe_lora_layers:
        print(
            f"[run_glm5_1_744b_a40b_lora] WARNING: MOE_LORA_LAYERS={_moe_lora_layers} is SET but the subset-rewrite "
            "feature is DISABLED (commented out for debugging) -> MoE-expert LoRA stays on ALL layers."
        )
    # if _keep_moe_lora and _moe_lora_layers:
    #     def _parse_layers(spec):
    #         out = []
    #         for part in spec.split(","):
    #             part = part.strip()
    #             if not part:
    #                 continue
    #             if "-" in part:
    #                 a, b = part.split("-", 1)
    #                 out.extend(range(int(a), int(b) + 1))
    #             else:
    #                 out.append(int(part))
    #         return sorted(set(out))
    #
    #     _layers = _parse_layers(_moe_lora_layers)
    #     _attn_only = [m for m in _tm.split(",") if m.strip() not in ("gate_proj", "up_proj", "down_proj")]
    #     _moe_patterns = []
    #     for _n in _layers:
    #         _moe_patterns.append(f"*.layers.{_n}.*.linear_fc1")
    #         _moe_patterns.append(f"*.layers.{_n}.*.linear_fc2")
    #     _tm = ",".join(_attn_only + _moe_patterns)
    #     print(
    #         f"[run_glm5_1_744b_a40b_lora] MOE_LORA_LAYERS={_moe_lora_layers} -> MoE-expert LoRA on layers {_layers} "
    #         f"only (attention LoRA still on all layers); target-modules: {_tm}"
    #     )
    # MoE-expert LoRA needs TWO INDEPENDENT flags, each controlling its own thing -- not a symmetric
    # pair, but both must be ON for serving to work. Enabled whenever the MoE expert projections
    # (gate_proj/up_proj/down_proj) are LoRA targets (KEEP_MOE_LORA=1, the default):
    #   --experts-shared-outer-loras      (TRAIN side: adapter laid out shared-outer -- gate_up lora_A
    #       / down lora_B shared across experts, expert_dim=1; arguments.py also auto-sets the serve-
    #       side --sglang-experts-shared-outer-loras so sglang knows the layout -- SGLang PR #21466).
    #   --sglang-lora-use-virtual-experts (SERVE side: sglang serves the expert LoRA via the
    #       virtual-experts path).
    # Turn on only one and serving breaks (e.g. expert gate_up LoRA-B dim 768 vs 24576 = 32x ep ->
    # "scheduler died during init"), so both go on together here. KEEP_MOE_LORA=0 turns both off
    # (attention-only LoRA: q/k/v/o + MLA q_a/kv_a/q_b/kv_b), the previous bring-up default.
    # (experts_shared_outer_loras=False keeps the per-expert layout instead; virtual-experts
    # serving stays on either way.)
    lora_args = f'--lora-rank {args.lora_rank} --lora-alpha {args.lora_alpha} --lora-dropout {args.lora_dropout} --target-modules "{_tm}" '
    if _keep_moe_lora and args.experts_shared_outer_loras:
        lora_args += "--experts-shared-outer-loras "
    if _is_full:
        # NOTE: when KEEP_MOE_LORA=1 (default) both MoE-expert-LoRA flags are on -- lora_args got the
        # train-side --experts-shared-outer-loras above and sglang_args gets the serve-side
        # --sglang-lora-use-virtual-experts below. KEEP_MOE_LORA=0 -> attention-only, neither emitted.
        lora_args += "--no-gradient-accumulation-fusion "
    # --lora-base-cpu-backup: default ON (see the ScriptArgs field comment — OFF silently breaks
    # on-policy with abs_diff ~1.46 / KL ~1.0; opt out only for full-744B + glm-native when host RAM
    # cannot take the ~372 GB/node mirror).
    if args.lora_base_cpu_backup:
        lora_args += "--lora-base-cpu-backup "

    # Math RL (both tasks score with the boxed/SymPy verifier --rm-type math). Shared rollout
    # args; the per-task block sets --prompt-data + --input-key. Same task-dispatch shape as
    # run_deepseek_v4.py.
    rollout_args = (
        "--label-key label "
        "--apply-chat-template "
        "--rollout-shuffle "
        "--rm-type math "
        f"--num-rollout {args.num_rollout} "
        f"--rollout-batch-size {args.rollout_batch_size} "
        f"--n-samples-per-prompt {args.n_samples_per_prompt} "
        f"--rollout-max-response-len {args.rollout_max_response_len} "
        "--rollout-temperature 1.0 "
        f"--global-batch-size {args.global_batch_size} "
    )
    match args.task:
        case "gsm8k":  # zhuzilin/gsm8k ships {messages, label} parquet
            rollout_args += f"--prompt-data {args.data_dir}/gsm8k/train.parquet --input-key messages "
        case "dapo-math":  # zhuzilin/dapo-math-17k ships {prompt, label} jsonl (prompt = chat messages)
            rollout_args += f"--prompt-data {args.data_dir}/dapo-math-17k/dapo-math-17k.jsonl --input-key prompt "
    # DAPO dynamic sampling (opt-in; see knob above). The filter drops zero-std (all-same-reward)
    # groups, so over-sampling-batch-size must exceed rollout-batch-size to refill the batch.
    if args.dapo_dynamic_sampling:
        rollout_args += (
            f"--over-sampling-batch-size {args.over_sampling_batch_size} "
            "--dynamic-sampling-filter-path miles.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonzero_std "
        )

    grpo_args = "--advantage-estimator grpo --kl-loss-coef 0.00 --kl-loss-type low_var_kl --kl-coef 0.00 --entropy-coef 0.00 --eps-clip 0.2 --eps-clip-high 0.28 "

    # R3 = rollout ROUTING replay ONLY: replay the rollout's MoE top-8 in training so the train-side
    # expert selection matches the rollout (on-policy parity). Cheap (sglang routed-experts capturer
    # ~0.5 GB/rank) and is the R3 we want for a faithful RL run.
    #
    # We deliberately DO NOT add --use-rollout-indexer-replay (the DSA indexer top-k replay):
    #   * It is a DEBUG aid -- it only verifies rollout-vs-train indexer top-k parity. The glm-native
    #     kernel recomputes the indexer top-k itself, so TRAINING DOES NOT NEED IT.
    #   * Enabling it makes sglang allocate the IndexerTopkCapturer HOST pinned buffer
    #     -- shape (max_total_num_tokens, num_layers, index_topk=2048) int32 ~= 78-128 GB/rank,
    #     x8 ranks/node -- which blew the ~1.78 TB colocate pod cgroup -> RolloutManager host-OOM.
    # So: R3 on => routing replay only (no indexer replay, no indexer host buffer).
    # --use-rollout-routing-replay now DEFAULTS ON (arguments.py BooleanOptionalAction), so emit it
    # explicitly on BOTH branches -- the R3 knob must still fully control it: use_r3=False has to
    # emit --no-use-rollout-routing-replay to actually disable the (now default-on) routing replay.
    if args.use_r3:
        r3_args = "--use-rollout-routing-replay "
    else:
        r3_args = "--no-use-rollout-routing-replay "

    optimizer_args = "--optimizer adam --lr 1e-5 --lr-decay-style constant --weight-decay 0.1 --adam-beta1 0.9 --adam-beta2 0.98 "
    # CPU Adam — offload the optimizer state to host RAM. DEFAULT ON (set OPTIMIZER_CPU_OFFLOAD=0 to
    # disable, e.g. for the pruned toy). The three flags MUST go together (offload + D2H/H2D overlap +
    # precision-aware optimizer), matching the full-FT recipe run_glm5_744b_a40b.py and the deepseek/
    # qwen3 siblings. Requires --use-distributed-optimizer (already on via _get_parallel_config).
    # NOTE: under LoRA only the small adapter params carry optimizer state, so the GPU-mem saving is
    # modest; kept on for parity with the full-FT recipe and easy full-FT reuse.
    if os.environ.get("OPTIMIZER_CPU_OFFLOAD", "1") != "0":
        optimizer_args += "--optimizer-cpu-offload --overlap-cpu-optimizer-d2h-h2d --use-precision-aware-optimizer "

    perf_args = _get_parallel_config(args)

    if _is_full:
        # FULL bf16 GLM-5 rollout: full-model sglang BASE (dp-attention + EP/DP + nsa/DSA +
        # dp-lm-head, mirrors run_glm5_744b_a40b.py) + GLM parsers + MoE-LoRA triton backend.
        # engine = ep = dp = rollout_num_gpus_per_engine (32 for full; bf16 ~1488GB needs >=~22/engine,
        # NOT the full-FT's fp8-only 8). NO deepep (fp8-gated there + conflicts with the triton LoRA
        # MoE runner), NO MTP. cuda-graph-max-bs stays 64 (256 risks bf16+dp graph-pool OOM).
        # fp8 rollout (PR#1376 GLM-5.2 recipe): engine = min(8, ngpu) — fp8 halves the weights so
        # the 744B model fits 1 node; fp8 KV cache + flashmla_kv decode + cuda-graph-max-bs 256.
        # bf16 rollout: engine = rollout_num_gpus_per_engine (32; bf16 ~1488GB needs ~22+/engine),
        # flashmla_sparse decode, cuda-graph-max-bs 64.
        _eng = min(8, args.num_gpus_per_node) if args.fp8_rollout else args.rollout_num_gpus_per_engine
        _decode = "flashmla_kv" if args.fp8_rollout else "flashmla_sparse"
        _cg = 256 if args.fp8_rollout else 64
        _kv = "--sglang-kv-cache-dtype fp8_e4m3 " if args.fp8_rollout else ""
        # MoE-expert-LoRA serve-side flag (the virtual-experts path); goes on together with the
        # train-side --experts-shared-outer-loras above. Empty unless KEEP_MOE_LORA=1.
        _ve = "--sglang-lora-use-virtual-experts " if _keep_moe_lora else ""
        sglang_args = (
            f"--rollout-num-gpus-per-engine {_eng} --sglang-mem-fraction-static {args.sglang_mem_fraction_static} "
            f"--sglang-enable-dp-attention --sglang-ep-size {_eng} --sglang-dp-size {_eng} "
            "--sglang-moe-dense-tp-size 1 --sglang-enable-dp-lm-head "
            f"--sglang-attention-backend nsa --sglang-nsa-decode-backend {_decode} "
            f"--sglang-nsa-prefill-backend flashmla_sparse --sglang-page-size 64 {_kv}"
            f"--sglang-cuda-graph-max-bs {_cg} --sglang-max-running-requests 512 "
            f"--sglang-chunked-prefill-size {2048 * _eng} --sglang-watchdog-timeout 3600 "
            f"--sglang-moe-runner-backend triton --sglang-disable-shared-experts-fusion {_ve}"
            # NOTE: --sglang-reasoning-parser/--sglang-tool-call-parser intentionally REMOVED
            # (per directive; the PR #1376 GLM-5.2 recipe does not set them).
            # sglang must size the LoRA buffers to the real rank. Without --max-lora-rank,
            # _get_lora_n_slices (= A_buffer.shape[-2] // lora_rank) miscounts the gate_up slices
            # under dp-attention -> partition-prefix(24576) != B-out(768) "scheduler died" crash.
            f"--sglang-max-lora-rank {args.lora_rank} "
            # LoRA kernel backend (default triton; see sglang_lora_backend field). csgmv is fragile
            # on the GLM DSA MoE-LoRA dp-attention path; triton is the robust default.
            f"--sglang-lora-backend {args.sglang_lora_backend} "
        )
    else:
        sglang_args = f"--rollout-num-gpus-per-engine {args.rollout_num_gpus_per_engine} --sglang-mem-fraction-static {args.sglang_mem_fraction_static} --sglang-cuda-graph-max-bs 64 --sglang-moe-runner-backend triton --sglang-disable-shared-experts-fusion --sglang-lora-backend {args.sglang_lora_backend} --sglang-reasoning-parser glm45 --sglang-tool-call-parser glm47 "

    save_args = f"--save-interval 1 --save {load_save_path} "

    misc_args = f"--attention-dropout 0.0 --hidden-dropout 0.0 --accumulate-allreduce-grads-in-fp32 --attention-softmax-in-fp32 --attention-backend flash --calculate-per-token-loss --use-miles-router --actor-num-nodes 1 --actor-num-gpus-per-node {args.num_gpus_per_node} --colocate "

    wandb_args = U.get_default_wandb_args(__file__, run_id=args.run_id) if args.enable_wandb else ""

    # seq window (training --seq-length + rollout --rollout-max-context-len). Omitted when seq_window
    # == 0 so megatron falls back to its default 4096 window (fine for the short gsm8k case). The .sh
    # wrappers may also pass these via --extra-args; argparse takes the last occurrence (env wins).
    seq_args = f"--seq-length {args.seq_window} --rollout-max-context-len {args.seq_window} " if args.seq_window > 0 else ""

    train_args = f"{ckpt_args} {lora_args} {rollout_args} {seq_args} {optimizer_args} {grpo_args} {r3_args} {wandb_args} {perf_args} {sglang_args} {save_args} {misc_args} {args.extra_args} "

    U.execute_train(
        train_args=train_args,
        config=args,
        num_gpus_per_node=args.num_gpus_per_node,
        megatron_model_type=args.megatron_model_type,
        extra_env_vars={
            "MILES_EXPERIMENTAL_ROLLOUT_REFACTOR": "1",
            # GLM-5 (glm_moe_dsa) DSA indexer uses INTERLEAVED RoPE, not NeoX. sglang must
            # match or the sparse-attention top-k index is computed on wrongly-rotated q/k ->
            # the indexer selects the wrong tokens -> rollout produces gibberish on long
            # sequences (short prompts < index_topk don't trigger the indexer, which is why a
            # short-prompt base server looked fine). run_glm5_744b_a40b.py:403 sets this for the
            # full model; the LoRA wrapper was missing it (the pruned toy masked it -- a heavy
            # prune is gibberish regardless). See run_deepseek_v32.py:262 (v3.2=NeoX, GLM-5=interleaved).
            "INDEXER_ROPE_NEOX_STYLE": "0",
            # Force the NSA/DSA MLA path in sglang (matches run_glm5_744b_a40b.py:357). Needed when
            # serving the full GLM-5 with --sglang-attention-backend nsa. Harmless on the toy.
            "SGLANG_NSA_FORCE_MLA": "1",
            # NOTE: do NOT set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True here. It is mutually
            # exclusive with torch_memory_saver (colocate offload): the saver's _sanity_checks()
            # raises "TorchMemorySaver is disabled ... expandable_segments is not supported yet".
            # The 744B DSA indexer OOM (mcore _compute_index_scores einsum, O(seq^2) [s,b,h,t] fp32
            # ~8 GiB at seq 8192) must be solved by REDUCING the peak (CP>1 halves seq -> halves the
            # indexer intermediate), not by changing the allocator backend.
        },
        megatron_path=args.megatron_path,
    )


@app.command()
@U.dataclass_cli
def prepare(args: ScriptArgs):
    """Download the model checkpoint (for a known HF repo) and the task dataset (gsm8k or dapo-math). Run once per node before training."""
    _prepare_download(args)


@app.command()
@U.dataclass_cli
def train(args: ScriptArgs):
    """Run GRPO LoRA training (assumes the dataset is already prepared)."""
    _train(args)


@app.command()
@U.dataclass_cli
def full_train(args: ScriptArgs):
    """Download the model checkpoint + dataset, then run GRPO LoRA training."""
    _prepare_download(args)
    _train(args)


@app.callback()
def _callback() -> None:
    pass


if __name__ == "__main__":
    app()
