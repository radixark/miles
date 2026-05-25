from __future__ import annotations

from scripts import run_qwen3_30b_a3b_deterministic as run_qwen3_30b_a3b


def test_qwen3_moe_script_true_on_policy_tp1_ep4_cp2_contract(monkeypatch):
    captured = {}

    def fake_execute_train(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(run_qwen3_30b_a3b.U, "execute_train", fake_execute_train)
    monkeypatch.setattr(run_qwen3_30b_a3b.U, "get_default_wandb_args", lambda *args, **kwargs: "")

    args = run_qwen3_30b_a3b.ScriptArgs(
        mode="debug_one_sample",
        run_id="unit-test-moe",
        true_on_policy=True,
        enable_eval=False,
        tensor_model_parallel_size=1,
        context_parallel_size=2,
        cp_comm_type="a2a",
        expert_model_parallel_size=4,
        expert_tensor_parallel_size=1,
        rollout_num_gpus=8,
        rollout_num_gpus_per_engine=4,
        use_sequence_parallel=True,
    )

    assert args.use_sequence_parallel is True

    run_qwen3_30b_a3b.execute(args)

    train_args = captured["train_args"]
    env_vars = captured["extra_env_vars"]
    config = captured["config"]

    assert "--tensor-model-parallel-size 1" in train_args
    assert "--context-parallel-size 2" in train_args
    assert "--cp-comm-type a2a" in train_args
    assert "--expert-model-parallel-size 4" in train_args
    assert "--expert-tensor-parallel-size 1" in train_args
    assert "--rollout-num-gpus 8" in train_args
    assert "--rollout-num-gpus-per-engine 4" in train_args
    assert "--num-rollout 1" in train_args
    assert "--rollout-batch-size 4" in train_args
    assert "--sglang-ep-size 4" in train_args
    assert "--sglang-data-parallel-size" not in train_args
    assert "--sglang-enable-dp-attention" not in train_args
    assert "--sglang-true-on-policy-contract qwen3_moe_true_on_policy_v1" in train_args
    assert "--true-on-policy-contract qwen3_moe_true_on_policy_v1" in train_args
    assert "--recompute-logprobs-via-prefill" not in train_args
    assert "--sequence-parallel" not in train_args
    assert "--no-gradient-accumulation-fusion" not in train_args
    assert "--use-sglang" not in train_args
    assert "ROW_LINEAR_ENABLE_INV" not in env_vars
    assert "NCCL_ALGO" not in env_vars
    assert "NCCL_ALGO=Ring" in config.extra_env_vars
    assert "MODEL_ARGS_DISABLE_MOE_PERMUTE_FUSION=1" in config.extra_env_vars


def test_qwen3_moe_script_true_on_policy_tp2_ep4_enables_sequence_parallel(monkeypatch):
    captured = {}

    def fake_execute_train(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(run_qwen3_30b_a3b.U, "execute_train", fake_execute_train)
    monkeypatch.setattr(run_qwen3_30b_a3b.U, "get_default_wandb_args", lambda *args, **kwargs: "")

    args = run_qwen3_30b_a3b.ScriptArgs(
        mode="debug_one_sample",
        run_id="unit-test-moe-tp2-ep4",
        true_on_policy=True,
        enable_eval=False,
        tensor_model_parallel_size=2,
        context_parallel_size=1,
        expert_model_parallel_size=4,
        expert_tensor_parallel_size=1,
        rollout_num_gpus=8,
        rollout_num_gpus_per_engine=4,
        use_sequence_parallel=True,
    )

    run_qwen3_30b_a3b.execute(args)

    train_args = captured["train_args"]

    assert "--tensor-model-parallel-size 2" in train_args
    assert "--sequence-parallel" in train_args
    assert "--expert-model-parallel-size 4" in train_args
    assert "--sglang-ep-size 4" in train_args
    assert "--make-vocab-size-divisible-by 1" in train_args
    assert "--no-gradient-accumulation-fusion" in train_args
    assert "--sglang-true-on-policy-contract qwen3_moe_true_on_policy_v1" in train_args
    assert "--true-on-policy-contract qwen3_moe_true_on_policy_v1" in train_args


def test_qwen3_moe_script_true_on_policy_pp2_ep4_contract(monkeypatch):
    captured = {}

    def fake_execute_train(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(run_qwen3_30b_a3b.U, "execute_train", fake_execute_train)
    monkeypatch.setattr(run_qwen3_30b_a3b.U, "get_default_wandb_args", lambda *args, **kwargs: "")

    args = run_qwen3_30b_a3b.ScriptArgs(
        mode="debug_one_sample",
        run_id="unit-test-moe-pp2-ep4",
        true_on_policy=True,
        enable_eval=False,
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=2,
        context_parallel_size=1,
        expert_model_parallel_size=4,
        expert_tensor_parallel_size=1,
        rollout_num_gpus=8,
        rollout_num_gpus_per_engine=4,
        use_sequence_parallel=False,
    )

    run_qwen3_30b_a3b.execute(args)

    train_args = captured["train_args"]

    assert "--tensor-model-parallel-size 1" in train_args
    assert "--pipeline-model-parallel-size 2" in train_args
    assert "Qwen3-30B-A3B_torch_dist_tp1_pp2_ep4_etp1" in train_args
    assert "--megatron-to-hf-mode bridge" not in train_args
    assert "--sequence-parallel" not in train_args
    assert "--no-gradient-accumulation-fusion" not in train_args
    assert "--expert-model-parallel-size 4" in train_args
    assert "--sglang-ep-size 4" in train_args
    assert "--sglang-true-on-policy-contract qwen3_moe_true_on_policy_v1" in train_args
    assert "--true-on-policy-contract qwen3_moe_true_on_policy_v1" in train_args


def test_qwen3_moe_script_true_on_policy_sp_pp_ep_contract(monkeypatch):
    captured = {}

    def fake_execute_train(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(run_qwen3_30b_a3b.U, "execute_train", fake_execute_train)
    monkeypatch.setattr(run_qwen3_30b_a3b.U, "get_default_wandb_args", lambda *args, **kwargs: "")

    args = run_qwen3_30b_a3b.ScriptArgs(
        mode="debug_one_sample",
        run_id="unit-test-moe-sp-pp-ep",
        true_on_policy=True,
        enable_eval=False,
        tensor_model_parallel_size=2,
        pipeline_model_parallel_size=2,
        context_parallel_size=1,
        expert_model_parallel_size=2,
        expert_tensor_parallel_size=1,
        rollout_num_gpus=8,
        rollout_num_gpus_per_engine=2,
        sglang_expert_parallel_size=2,
        use_sequence_parallel=True,
    )

    run_qwen3_30b_a3b.execute(args)

    train_args = captured["train_args"]

    assert "--tensor-model-parallel-size 2" in train_args
    assert "--sequence-parallel" in train_args
    assert "--pipeline-model-parallel-size 2" in train_args
    assert "--expert-model-parallel-size 2" in train_args
    assert "--sglang-ep-size 2" in train_args
    assert "--rollout-num-gpus-per-engine 2" in train_args
    assert "--rollout-batch-size 2" in train_args
    assert "--global-batch-size 2" in train_args
    assert "Qwen3-30B-A3B_torch_dist_tp2_pp2_ep2_etp1" in train_args
    assert "--make-vocab-size-divisible-by 1" in train_args
    assert "--no-gradient-accumulation-fusion" in train_args
    assert "--sglang-true-on-policy-contract qwen3_moe_true_on_policy_v1" in train_args
    assert "--true-on-policy-contract qwen3_moe_true_on_policy_v1" in train_args


def test_qwen3_moe_default_rollout_engine_size_matches_sglang_ep():
    args = run_qwen3_30b_a3b.ScriptArgs(
        mode="debug_one_sample",
        run_id="unit-test-moe-defaults",
        true_on_policy=True,
        enable_eval=False,
        tensor_model_parallel_size=1,
        context_parallel_size=2,
        cp_comm_type="a2a",
        expert_model_parallel_size=4,
        expert_tensor_parallel_size=1,
        rollout_num_gpus=8,
        use_sequence_parallel=True,
    )

    assert args.sglang_expert_parallel_size == 4
    assert args.rollout_num_gpus_per_engine == 4


def test_qwen3_moe_sppp_default_rollout_engine_size_matches_sglang_moe_tp():
    args = run_qwen3_30b_a3b.ScriptArgs(
        mode="debug_one_sample",
        run_id="unit-test-moe-sppp-defaults",
        true_on_policy=True,
        enable_eval=False,
        tensor_model_parallel_size=2,
        pipeline_model_parallel_size=2,
        context_parallel_size=1,
        expert_model_parallel_size=2,
        expert_tensor_parallel_size=1,
        rollout_num_gpus=8,
        use_sequence_parallel=True,
    )

    assert args.sglang_expert_parallel_size == 2
    assert args.rollout_num_gpus_per_engine == 2


def test_qwen3_moe_pp_uses_topology_specific_torch_dist_checkpoint():
    args = run_qwen3_30b_a3b.ScriptArgs(
        mode="debug_one_sample",
        run_id="unit-test-moe-pp-checkpoint",
        true_on_policy=True,
        enable_eval=False,
        model_dir="/models",
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=2,
        context_parallel_size=1,
        expert_model_parallel_size=4,
        expert_tensor_parallel_size=1,
        rollout_num_gpus=8,
        rollout_num_gpus_per_engine=4,
        use_sequence_parallel=False,
    )

    checkpoint_path = run_qwen3_30b_a3b._megatron_torch_dist_path(args)
    conversion_args = run_qwen3_30b_a3b._megatron_torch_dist_conversion_args(args)

    assert checkpoint_path == "/models/Qwen3-30B-A3B_torch_dist_tp1_pp2_ep4_etp1"
    assert "--tensor-model-parallel-size 1" in conversion_args
    assert "--pipeline-model-parallel-size 2" in conversion_args
    assert "--expert-model-parallel-size 4" in conversion_args
    assert "--expert-tensor-parallel-size 1" in conversion_args


def test_qwen3_moe_pp1_uses_topology_specific_torch_dist_checkpoint_path():
    args = run_qwen3_30b_a3b.ScriptArgs(
        mode="debug_one_sample",
        run_id="unit-test-moe-pp1-checkpoint",
        true_on_policy=True,
        enable_eval=False,
        model_dir="/models",
        tensor_model_parallel_size=2,
        pipeline_model_parallel_size=1,
        context_parallel_size=1,
        expert_model_parallel_size=4,
        expert_tensor_parallel_size=1,
        rollout_num_gpus=8,
        rollout_num_gpus_per_engine=4,
        use_sequence_parallel=True,
    )

    checkpoint_path = run_qwen3_30b_a3b._megatron_torch_dist_path(args)
    conversion_args = run_qwen3_30b_a3b._megatron_torch_dist_conversion_args(args)

    assert checkpoint_path == "/models/Qwen3-30B-A3B_torch_dist_tp2_pp1_ep4_etp1"
    assert "--tensor-model-parallel-size 2" in conversion_args
    assert "--pipeline-model-parallel-size 1" in conversion_args
    assert "--expert-model-parallel-size 4" in conversion_args
    assert "--expert-tensor-parallel-size 1" in conversion_args
    assert "--make-vocab-size-divisible-by 1" in conversion_args
