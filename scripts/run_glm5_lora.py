"""
GLM-5 / 5.1 / 5.2 GRPO LoRA training script (Megatron-Bridge / bridge mode).

These models are MoE + MLA + DSA (DeepSeek Sparse Attention). LoRA trains through the
Megatron-Bridge path (``--megatron-to-hf-mode bridge``); the "dsa" experimental-attention
spec is provided by the Megatron-Bridge GLM5 provider, so NO ``--spec`` is consumed here
(the registry ``.sh`` may still list it; it is provably inert under bridge LoRA).

GLM-5.2 additionally has DSA *cross-layer index sharing* (only "computing" layers carry the
indexer; "skip" layers reuse the most recent computing layer's top-k). The Megatron-Bridge
GLM5 provider reads that schedule (``index_topk_freq``) from the HF config and builds
``CrossLayerDSAttention`` -- nothing extra is needed here beyond selecting a GLM-5.2 model
name (which maps to the ``glm5.2-744B-A40B*`` registry, identical to GLM-5.1 except
``--rotary-base`` 8e6).

Modeled on ``scripts/run_deepseek_v4.py`` (typer app + ScriptArgs(ExecuteTrainConfig)
+ model-name -> megatron_model_type registry + ``_get_parallel_config`` + ``U.execute_train``).

DSA kernel backend (``--dsa-attention-backend``; bridge path only). Orthogonal to model version
and to LoRA -- BOTH backends run GLM-5.1 *and* GLM-5.2, full or LoRA:
  * ``megatron-bridge`` (default): the portable unfused megatron-core DSA kernels (DSAttention /
    CrossLayerDSAttention). No extra deps. Uses the ``bshd`` query layout.
  * ``slime``: the vendored fused TileLang kernels (SparseMLA + lighting_indexer), matching slime's
    rollout kernels for rollout<->train numerical parity (incl. R3 indexer replay). Needs the
    optional ``tilelang`` dep and the ``thd`` (packed) layout. Training/forward-only (no KV cache):
    it cannot serve generation -- the rollout is always served by sglang.
  This launcher selects the matching ``--qkv-format`` automatically from the backend (see
  ``_get_parallel_config``); just pass ``--dsa-attention-backend slime`` or leave the default. See
  the Megatron-Bridge ``models/glm_moe_dsa/__init__.py`` docstring for the full backend matrix.

Two DSA specifics:
  * ``--target-modules`` excludes the 3 DSA indexer modules (wq_b/wk/weights_proj) by default --
    the indexer stays a frozen base capability; this run does not train it. On the slime backend the
    indexer adapter gets no gradient anyway (a genuine no-op); on the default backend it would get a
    tiny aux-loss gradient (~1e-5), so excluding it there is a deliberate choice.
  * ``--micro-batch-size 1`` (no ``--use-dynamic-batch-size``): both backends pin a static
    micro-batch. The query layout follows the backend -- ``bshd`` (megatron-core's DSA
    core-attention needs a 4D query; the default ``thd`` packing yields a 3D query and raises "not
    enough values to unpack") for ``megatron-bridge``, ``thd`` (packed) for ``slime``.

Supported model variants (HF checkpoint must be the native config,
model_type=glm_moe_dsa / GlmMoeDsaForCausalLM):
  GLM-5.1 / GLM-5.2            full models
  GLM-5.1-6layer              6-layer GLM-5.1 prune (jybsuper/GLM-5.1-6layer)
  GLM-5.2_5layer              5-layer GLM-5.2 prune (Pinaster/GLM-5.2_5layer; 3 dense + 2 MoE)
  GLM-5.1-4layer / -20layer   other GLM-5.1 prunes

Usage (run ON the devbox; miles editable-installed under /personal):
  python scripts/run_glm5_lora.py prepare    --model-name GLM-5.1-6layer   # download model + task dataset (default gsm8k)
  # default (megatron-bridge / unfused) backend:
  python scripts/run_glm5_lora.py full-train --model-name GLM-5.1-6layer --num-gpus-per-node 4
  # fused slime backend (GLM-5.2 shown; works for GLM-5.1 too):
  python scripts/run_glm5_lora.py full-train --model-name GLM-5.2_5layer \\
      --dsa-attention-backend slime --num-gpus-per-node 4
  # DAPO-Math example (zhuzilin/dapo-math-17k, long-CoT competition math -- use a longer response len):
  python scripts/run_glm5_lora.py prepare --model-name GLM-5.2_5layer --task dapo-math
  python scripts/run_glm5_lora.py train   --model-name GLM-5.2_5layer --task dapo-math \\
      --rollout-max-response-len 4096 --num-gpus-per-node 4   # add --dapo-dynamic-sampling on a real model

GLM-5.2 rollout caveat: sglang does not yet serve the GLM-5.2 cross-layer (subset-indexer)
checkpoint, so a full rollout->train loop is blocked on the rollout side. The *training* side
is validated train-only by replaying a dumped rollout (both toys share the GLM tokenizer/vocab):
  # 1) dump a rollout from GLM-5.1 (sglang serves 5.1 fine)
  python scripts/run_glm5_lora.py full-train --model-name GLM-5.1-6layer \\
      --extra-args "--dump-details /personal/dump51"
  # 2) train GLM-5.2 on that dump (no sglang)
  python scripts/run_glm5_lora.py train --model-name GLM-5.2_5layer \\
      --extra-args "--load-debug-rollout-data /personal/dump51/rollout_data/0.pt"
"""

from dataclasses import dataclass
from typing import Literal

import typer

import miles.utils.external_utils.command_utils as U

app = typer.Typer()

# HF repos to download from (full models from zai-org; pruned toys from jybsuper / Pinaster).
# Variants absent here (e.g. -4layer/-20layer) are assumed already present at --hf-checkpoint.
_HF_REPO = {
    "GLM-5.1": "zai-org/GLM-5.1",
    "GLM-5.1-6layer": "jybsuper/GLM-5.1-6layer",
    "GLM-5.2": "zai-org/GLM-5.2",
    "GLM-5.2_5layer": "Pinaster/GLM-5.2_5layer",
}

_MEGATRON_MODEL_TYPE = {
    "GLM-5.1": "glm5-744B-A40B",
    "GLM-5.1-6layer": "glm5.1-744B-A40B_6layer",
    "GLM-5.1-4layer": "glm5-744B-A40B_4layer",
    "GLM-5.1-20layer": "glm5-744B-A40B_20layer",
    "GLM-5.2": "glm5.2-744B-A40B",
    "GLM-5.2_5layer": "glm5.2-744B-A40B_5layer",
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
        "GLM-5.1-4layer",
        "GLM-5.1-20layer",
        "GLM-5.2",
        "GLM-5.2_5layer",
    ] = "GLM-5.1-6layer"
    # Example task / dataset (math RL, --rm-type math for both):
    #   gsm8k     -> zhuzilin/gsm8k, short-answer grade-school math (~256-tok responses).
    #   dapo-math -> zhuzilin/dapo-math-17k (DAPO-Math-17k), hard long-CoT competition math.
    #                Pass a larger --rollout-max-response-len (e.g. 4096); a >2048 total seq is
    #                also what makes the GLM-5.2 DSA indexer go genuinely SPARSE (cross-layer
    #                path), unlike the dense short gsm8k case. Optionally enable DAPO dynamic
    #                sampling (see dapo_dynamic_sampling below). Mirrors run_deepseek_v4.py.
    task: Literal["gsm8k", "dapo-math"] = "gsm8k"

    hf_checkpoint: str | None = None
    model_dir: str = "/root/models"
    save_dir: str = "/personal/checkpoints"
    data_dir: str = "/root/datasets"
    megatron_path: str = "/root/Megatron-LM"

    # DSA sparse-MLA kernel backend (bridge path only). Orthogonal to model version & LoRA: BOTH
    # backends support GLM-5.1 AND GLM-5.2, full or LoRA.
    #   "slime" (default):  vendored fused TileLang SparseMLA + lighting_indexer; thd layout; needs
    #                       tilelang; matches slime's rollout kernels for rollout<->train parity
    #                       (incl. R3 indexer replay); training/forward-only (no KV cache).
    #   "megatron-bridge":  portable unfused megatron-core DSA kernels; bshd layout; no extra deps.
    # The matching --qkv-format is chosen from this automatically (see _get_parallel_config).
    dsa_attention_backend: Literal["megatron-bridge", "slime"] = "slime"

    # R3 (rollout routing replay, arxiv 2510.11370): during training, replay the rollout's recorded
    # MoE top-8 so the train-side expert selection matches the rollout (on-policy). Adds ONLY
    # --use-rollout-routing-replay. The DSA indexer top-k replay (--use-rollout-indexer-replay) is
    # NOT added: it is a debug-only parity check (the slime kernel recomputes the top-k) and it
    # triggers sglang's ~78-128 GB/rank IndexerTopkCapturer host buffer that OOM'd the colocate pod.
    # See _build_train_args r3_args for the full rationale.
    use_r3: bool = True

    # performance
    num_gpus_per_node: int = 4

    # LoRA
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.0
    target_modules: str = _DEFAULT_TARGET_MODULES

    # rollout
    num_rollout: int = 1
    rollout_batch_size: int = 4
    n_samples_per_prompt: int = 4
    # NB: the GLM-5.2 DSA indexer has index_topk=2048. It performs genuine SPARSE top-k selection
    # (and thus exercises the cross-layer DSA path) only when the full sequence (prompt + response)
    # EXCEEDS 2048; at shorter seq (the gsm8k default below) the indexer degenerates to DENSE
    # (top-k >= seq -> selects all keys). Use a longer prompt/response (> 2048) to hit sparse indexing.
    rollout_max_response_len: int = 256
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

    @property
    def megatron_model_type(self) -> str:
        return _MEGATRON_MODEL_TYPE[self.model_name]


def _get_parallel_config(args: ScriptArgs) -> str:
    """Single-node MoE layout: TP = EP = num_gpus_per_node, DP1 (mirrors run_glm5_744b_a40b).

    The DSA kernel backend dictates the query layout; both forbid --use-dynamic-batch-size, hence
    --micro-batch-size 1:
      * megatron-bridge (unfused megatron-core DSA core-attention): needs bshd (4D query); the
        default thd packing yields a 3D query and raises "not enough values to unpack".
      * slime (fused TileLang SparseMLA + lighting_indexer): needs thd (packed) -- the fused
        kernels index by cu_seqlens and use a [t, heads, dim] layout; bshd has no cu_seqlens.
    """
    ngpu = args.num_gpus_per_node
    # Canonical GLM-5 layout TP=EP=ngpu + sequence-parallel, CP=1 here (the full-scale recipe adds
    # context-parallel + --allgather-cp). BOTH backends run with sequence-parallel: the slime fused
    # thd indexer / SparseMLA path is SP/CP-aware in the bridge port -- slime_mla.py reconciles
    # index_q/index_k/head_weights via SP(+CP) all-gather and cu_seqlens (starts/ends) via CP-scatter,
    # mirroring native slime glm5.py, so the fused kernel sees matching token dims under SP. The only
    # per-backend difference is the query layout: thd (packed, cu_seqlens-indexed) for slime, bshd
    # for the unfused megatron-core core-attention.
    qkv_format = "thd" if args.dsa_attention_backend == "slime" else "bshd"
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
    print(f"[run] GLM-5 LoRA: model={args.model_name} (megatron_model_type={args.megatron_model_type}), dsa-backend={args.dsa_attention_backend}, r3={args.use_r3}, {args.num_gpus_per_node} GPUs, rollout tp={args.rollout_num_gpus_per_engine}")
    load_save_path = f"{args.save_dir}/{args.run_id}"

    ckpt_args = (
        f"--hf-checkpoint {args.hf_checkpoint} --megatron-to-hf-mode bridge "
        f"--dsa-attention-backend {args.dsa_attention_backend} "
    )

    # Full GLM-5 (non-5layer) gets the MoE-LoRA rollout mechanism flags (Kimi K2.5 LoRA recipe,
    # MINUS --experts-shared-outer-loras per directive): virtual-experts MoE-LoRA path, frozen
    # base kept on CPU under colocate, no grad-accum-fusion. target-modules stays GLM's own
    # default (the correct GLM-5.2 MLA superset; NOT Kimi's DeepSeek-named list). The 5-layer toy
    # keeps the simple path unchanged.
    # Apply the FULL GLM-5.2 rollout config (dp-attention / nsa / EP + MLP-target drop + MoE-LoRA
    # flags + no reasoning/tool parser) to ALL GLM-5 models INCLUDING the 5-layer toy (per request).
    # The 5-layer is glm_moe_dsa too, so it exercises the same DSA / dp-attention / LoRA serving
    # path -> a fast (~seconds to load) way to validate the rollout sglang config without the full
    # model's ~15-min load. Engine size / mem-fraction still come from the C0 script per model
    # (5layer -> rollout-num-gpus-per-engine 2, mem-frac 0.5; full -> 32, 0.70). Set to
    # `"5layer" not in args.model_name` to restore the old simple csgmv toy path.
    _is_full = True
    # (B) Full model: DROP gate_proj/up_proj/down_proj from the rollout LoRA targets. sglang's
    # mem_pool LoRA-B buffer probe (mem_pool.py:359-383) fails to detect the fully-DP dense MLP
    # gate_up (tp=1 under --sglang-moe-dense-tp-size 1) and falls back to the GLOBAL tp_size,
    # sizing B as 24576/tp_size -> "LoRA B output dim != base partition prefix dim 24576" scheduler
    # crash (orthogonal to fp8/engine-size/virtual-experts/max-lora-rank). LoRA only attention
    # (q/k/v/o + MLA q_a/kv_a/q_b/kv_b) until the sglang probe is patched.
    _tm = args.target_modules
    if _is_full:
        _tm = ",".join(m for m in _tm.split(",") if m.strip() not in ("gate_proj", "up_proj", "down_proj"))
    lora_args = f'--lora-rank {args.lora_rank} --lora-alpha {args.lora_alpha} --lora-dropout {args.lora_dropout} --target-modules "{_tm}" '
    if _is_full:
        # NOTE: --sglang-lora-use-virtual-experts is DROPPED — it requires --experts-shared-outer-loras
        # (matched pair; without it the MoE expert gate_up LoRA-B dim mismatches sglang, 768 vs 24576
        # = 32x ep -> "scheduler died during init"). User opted to drop BOTH. The MoE expert LoRA then
        # rides the default (non-virtual-experts) sglang path; at rollout 0 the expert-LoRA delta is
        # zero anyway (B=0), so this isolates the rollout config (dp-attention+nsa+INDEXER_ROPE_NEOX_STYLE=0).
        # NOTE: --lora-base-cpu-backup is intentionally NOT added here (opt-in only). It keeps a
        # HOST-RAM mirror of the base weights on the sglang side (enable_weights_cpu_backup) so they
        # survive torch_memory_saver.pause() without re-ship; but at full 744B scale on the slime
        # backend that mirror (~372 GB/node) + megatron's slime init pushed the colocate pod past
        # its ~1.78 TB cgroup memory.max -> RolloutManager SIGTERM'd (host OOM, NOT GPU) -> sglang
        # dp-attn all_gather peers saw "connection reset" -> cluster-wide scheduler crash. Enable it
        # via the run_glm5_lora_multinode.sh knob LORA_BASE_CPU_BACKUP=on (passed through --extra-args)
        # only when host RAM allows. Trade-off when OFF: skip_base_sync=False -> trainer re-ships base per swap.
        lora_args += "--no-gradient-accumulation-fusion "

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
    #   * It is a DEBUG aid -- it only verifies rollout-vs-train indexer top-k parity. The slime
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

    perf_args = _get_parallel_config(args)

    if _is_full:
        # FULL bf16 GLM-5.2 rollout: full-model sglang BASE (dp-attention + EP/DP + nsa/DSA +
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
        sglang_args = (
            f"--rollout-num-gpus-per-engine {_eng} --sglang-mem-fraction-static {args.sglang_mem_fraction_static} "
            f"--sglang-enable-dp-attention --sglang-ep-size {_eng} --sglang-dp-size {_eng} "
            "--sglang-moe-dense-tp-size 1 --sglang-enable-dp-lm-head "
            f"--sglang-attention-backend nsa --sglang-nsa-decode-backend {_decode} "
            f"--sglang-nsa-prefill-backend flashmla_sparse --sglang-page-size 64 {_kv}"
            f"--sglang-cuda-graph-max-bs {_cg} --sglang-max-running-requests 512 "
            f"--sglang-chunked-prefill-size {2048 * _eng} --sglang-watchdog-timeout 3600 "
            "--sglang-moe-runner-backend triton --sglang-disable-shared-experts-fusion "
            # NOTE: --sglang-reasoning-parser/--sglang-tool-call-parser intentionally REMOVED
            # (per directive; the PR #1376 GLM-5.2 recipe does not set them).
            # sglang must size the LoRA buffers to the real rank. Without --max-lora-rank,
            # _get_lora_n_slices (= A_buffer.shape[-2] // lora_rank) miscounts the gate_up slices
            # under dp-attention -> partition-prefix(24576) != B-out(768) "scheduler died" crash.
            f"--sglang-max-lora-rank {args.lora_rank} "
        )
    else:
        sglang_args = f"--rollout-num-gpus-per-engine {args.rollout_num_gpus_per_engine} --sglang-mem-fraction-static {args.sglang_mem_fraction_static} --sglang-cuda-graph-max-bs 64 --sglang-moe-runner-backend triton --sglang-disable-shared-experts-fusion --sglang-reasoning-parser glm45 --sglang-tool-call-parser glm47 "

    save_args = f"--save-interval 1 --save {load_save_path} "

    misc_args = f"--attention-dropout 0.0 --hidden-dropout 0.0 --accumulate-allreduce-grads-in-fp32 --attention-softmax-in-fp32 --attention-backend flash --calculate-per-token-loss --use-miles-router --actor-num-nodes 1 --actor-num-gpus-per-node {args.num_gpus_per_node} --colocate "

    wandb_args = U.get_default_wandb_args(__file__, run_id=args.run_id) if args.enable_wandb else ""

    train_args = f"{ckpt_args} {lora_args} {rollout_args} {optimizer_args} {grpo_args} {r3_args} {wandb_args} {perf_args} {sglang_args} {save_args} {misc_args} {args.extra_args} "

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
            # full model; the LoRA wrapper was missing it (the 5-layer toy masked it -- a 5/78
            # prune is gibberish regardless). See run_deepseek_v32.py:262 (v3.2=NeoX, GLM-5=interleaved).
            "INDEXER_ROPE_NEOX_STYLE": "0",
            # Force the NSA/DSA MLA path in sglang (matches run_glm5_744b_a40b.py:357). Needed when
            # serving the full GLM-5.2 with --sglang-attention-backend nsa. Harmless on the toy.
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


if __name__ == "__main__":
    app()
