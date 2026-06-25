from types import SimpleNamespace

import pytest
import torch

from miles.backends.megatron_utils.predictive_router_replay import (
    PREDICTIVE_LAYER_SCALE_SCHEDULES,
    PredictiveRouterReplayState,
    RouterPredictiveAction,
    _ensure_bias_predictor_runtime_placement,
    build_synthetic_predictive_loss,
    calculate_topk_accuracy,
    clear_predictive_optimizer_grads,
    compute_predictive_layer_scale,
    compute_predictive_bias_ratio,
    compute_predictive_loss,
    disable_predictive_param_groups,
    get_predictive_replay_controller,
    restore_predictive_param_groups,
    stabilize_predictive_delta_logits,
)
from miles.backends.megatron_utils.predictive_router_utils import (
    RecordedPredictiveMicrobatch,
    build_local_predictive_sample_lengths,
    pack_recorded_predictive_microbatch,
)


def test_build_synthetic_predictive_loss_returns_zero_with_intact_param_graph():
    """The synthetic loss exists so a DP rank that received no predictive
    microbatch this rollout can still participate in the gradient all-reduce
    over the bias-predictor parameters. The loss value must be exactly 0 and
    backward must populate (zero) gradients on every parameter without
    raising — otherwise the no-data rank would diverge from the data ranks.
    """

    torch.manual_seed(0)
    bias_predictor = torch.nn.Sequential(
        torch.nn.Linear(8, 16, bias=True),
        torch.nn.GELU(),
        torch.nn.Linear(16, 4, bias=True),
    )
    for param in bias_predictor.parameters():
        param.grad = None
    input_tensor = torch.randn(3, 8, requires_grad=True)

    loss = build_synthetic_predictive_loss(
        bias_predictor=bias_predictor,
        input_tensor=input_tensor,
    )

    assert loss.shape == ()
    assert loss.item() == pytest.approx(0.0)
    # Loss must be differentiable so the collective participates correctly.
    assert loss.requires_grad
    loss.backward()
    # Every bias-predictor parameter must have a (zero) grad — i.e. the
    # autograd graph reached every leaf.
    for param in bias_predictor.parameters():
        assert param.grad is not None
        assert torch.equal(param.grad, torch.zeros_like(param))
    # Input is .detach()'d inside, so its grad must remain None.
    assert input_tensor.grad is None


def test_pack_recorded_predictive_microbatch_builds_mask_and_storage():
    parallel_state = SimpleNamespace(cp_rank=0, cp_size=1)
    recorded_old_inputs = [
        torch.arange(18, dtype=torch.float32).reshape(6, 3),
        torch.arange(18, 36, dtype=torch.float32).reshape(6, 3),
    ]
    recorded_old_logits = [
        torch.arange(24, dtype=torch.float32).reshape(6, 4),
        torch.arange(24, 48, dtype=torch.float32).reshape(6, 4),
    ]

    packed = pack_recorded_predictive_microbatch(
        recorded_old_inputs=recorded_old_inputs,
        recorded_old_logits=recorded_old_logits,
        total_lengths=[2, 4],
        parallel_state=parallel_state,
        qkv_format="thd",
        downsample_batch_size=1,
        max_len_limit=2,
        storage_dtype="fp16",
    )

    assert packed.sample_lengths == [2, 4]
    assert packed.sampled_indices == [0]
    assert packed.valid_mask.tolist() == [True, True, False, False, False, False]
    assert packed.old_inputs_concat.dtype == torch.float16
    assert packed.old_logits_concat.dtype == torch.float16
    assert packed.old_inputs_concat.shape == torch.Size([2, 2, 3])
    assert packed.old_logits_concat.shape == torch.Size([2, 2, 4])
    assert packed.original_total_tokens == 2
    assert packed.selected_total_tokens == 2
    assert packed.predictive_loss_scale == 1.0


def test_pack_recorded_predictive_microbatch_caps_total_tokens_after_sampling():
    parallel_state = SimpleNamespace(cp_rank=0, cp_size=1)
    recorded_old_inputs = [
        torch.arange(18, dtype=torch.float32).reshape(6, 3),
        torch.arange(18, 36, dtype=torch.float32).reshape(6, 3),
    ]
    recorded_old_logits = [
        torch.arange(24, dtype=torch.float32).reshape(6, 4),
        torch.arange(24, 48, dtype=torch.float32).reshape(6, 4),
    ]

    packed = pack_recorded_predictive_microbatch(
        recorded_old_inputs=recorded_old_inputs,
        recorded_old_logits=recorded_old_logits,
        total_lengths=[2, 4],
        parallel_state=parallel_state,
        qkv_format="thd",
        downsample_batch_size=2,
        max_total_tokens=3,
        storage_dtype="fp32",
    )

    assert packed.sampled_indices == [0, 1]
    assert packed.selected_sample_lengths == [2, 1]
    assert packed.valid_mask.tolist() == [True, True, True, False, False, False]
    assert packed.old_inputs_concat.shape == torch.Size([3, 2, 3])
    assert packed.old_logits_concat.shape == torch.Size([3, 2, 4])
    assert packed.original_total_tokens == 6
    assert packed.selected_total_tokens == 3
    assert packed.predictive_loss_scale == 0.5


def test_build_local_predictive_sample_lengths_uses_arithmetic_layout():
    parallel_state = SimpleNamespace(cp_rank=0, cp_size=2)

    assert build_local_predictive_sample_lengths(
        total_lengths=[5, 8],
        parallel_state=parallel_state,
        qkv_format="thd",
    ) == [4, 4]

    assert build_local_predictive_sample_lengths(
        total_lengths=[3, 7],
        parallel_state=parallel_state,
        qkv_format="bshd",
        max_seq_lens=[8, 8],
    ) == [4, 4]


def test_predictive_router_replay_registry_and_metrics():
    PredictiveRouterReplayState.reset_registry()
    state0 = PredictiveRouterReplayState()
    state1 = PredictiveRouterReplayState()

    PredictiveRouterReplayState.set_global_predictive_action(RouterPredictiveAction.COMPUTE_PREDICTIVE_LOSS)
    assert state0.predictive_action == RouterPredictiveAction.COMPUTE_PREDICTIVE_LOSS
    assert state1.predictive_action == RouterPredictiveAction.COMPUTE_PREDICTIVE_LOSS

    old_inputs_concat = torch.randn(5, 2, 3)
    old_logits_concat = torch.randn(5, 2, 4)
    valid_mask = torch.tensor([True, False, True, True, False], dtype=torch.bool)
    PredictiveRouterReplayState.set_global_predictive_data(
        old_inputs_concat=old_inputs_concat,
        old_logits_concat=old_logits_concat,
        valid_mask=valid_mask,
        loss_scale=0.25,
    )

    state0_inputs, state0_logits, state0_mask, state0_loss_scale = state0.get_predictive_data()
    state1_inputs, state1_logits, state1_mask, state1_loss_scale = state1.get_predictive_data()
    assert state0_inputs.shape == torch.Size([5, 1, 3])
    assert state0_logits.shape == torch.Size([5, 1, 4])
    assert state1_inputs.shape == torch.Size([5, 1, 3])
    assert state1_logits.shape == torch.Size([5, 1, 4])
    assert torch.equal(state0_mask, valid_mask)
    assert torch.equal(state1_mask, valid_mask)
    assert state0_loss_scale == 0.25
    assert state1_loss_scale == 0.25

    PredictiveRouterReplayState.record_predictive_loss(0, 1.0)
    PredictiveRouterReplayState.record_predictive_loss(1, 3.0)
    PredictiveRouterReplayState.record_predictive_bias_ratio(0, 2.0)
    PredictiveRouterReplayState.record_predictive_topk_accuracy(0, 0.25)
    PredictiveRouterReplayState.record_predictive_scalar_metric("predictive_stabilizer_scale", 1, 0.5)
    metrics = PredictiveRouterReplayState.get_and_clear_predictive_metrics()
    assert metrics == {
        "predictive_loss": 2.0,
        "predictive_bias_to_logits_ratio": 2.0,
        "predictive_topk_accuracy": 0.25,
        "predictive_stabilizer_scale": 0.5,
    }
    assert PredictiveRouterReplayState.get_and_clear_predictive_metrics() == {}

    PredictiveRouterReplayState.clear_global_predictive_data()
    assert state0.get_predictive_data() == (None, None, None, 1.0)
    assert state1.get_predictive_data() == (None, None, None, 1.0)
    PredictiveRouterReplayState.clear_global_predictive_action()
    assert state0.predictive_action == RouterPredictiveAction.DISABLED
    assert state1.predictive_action == RouterPredictiveAction.DISABLED


def test_predictive_controller_applies_compute_and_skip_modes():
    controller = get_predictive_replay_controller()
    controller.reset_registry()
    state0 = PredictiveRouterReplayState()
    state1 = PredictiveRouterReplayState()

    skipped_microbatch = RecordedPredictiveMicrobatch(
        old_inputs_concat=torch.full((2, 2, 4), 1.0),
        old_logits_concat=torch.full((2, 2, 5), 2.0),
        valid_mask=torch.tensor([True, True], dtype=torch.bool),
        sampled_indices=[0],
        sample_lengths=[2],
        total_token_count=2,
        predictive_loss_scale=1.0,
    )
    valid_microbatch = RecordedPredictiveMicrobatch(
        old_inputs_concat=torch.full((3, 2, 4), 7.0),
        old_logits_concat=torch.full((3, 2, 5), 9.0),
        valid_mask=torch.tensor([True, False, True], dtype=torch.bool),
        sampled_indices=[1],
        sample_lengths=[3],
        total_token_count=3,
        predictive_loss_scale=0.5,
    )
    empty_microbatch = RecordedPredictiveMicrobatch(
        old_inputs_concat=None,
        old_logits_concat=None,
        valid_mask=torch.zeros(0, dtype=torch.bool),
        sampled_indices=[],
        sample_lengths=[],
        total_token_count=0,
        predictive_loss_scale=1.0,
    )

    controller.clear_microbatch_buffer()
    controller.append_microbatch(skipped_microbatch)
    controller.append_microbatch(valid_microbatch)
    controller.apply_predictive_train_mode("skip", consume_microbatch=True)
    assert controller.get_global_predictive_action() == RouterPredictiveAction.SKIP_PREDICTIVE
    assert controller.used_valid_predictive_data is False
    assert controller.remaining_microbatch_count() == 1
    assert state0.get_predictive_data() == (None, None, None, 1.0)

    controller.apply_predictive_train_mode("compute")
    assert controller.get_global_predictive_action() == RouterPredictiveAction.COMPUTE_PREDICTIVE_LOSS
    assert controller.used_valid_predictive_data is True
    assert state0.get_predictive_data()[0].shape == torch.Size([3, 1, 4])
    assert state1.get_predictive_data()[1].shape == torch.Size([3, 1, 5])
    assert torch.all(state0.get_predictive_data()[0] == 7.0)
    assert torch.all(state1.get_predictive_data()[1] == 9.0)
    assert state0.get_predictive_data()[3] == 0.5
    assert state1.get_predictive_data()[3] == 0.5

    controller.reset_train_step_usage()
    controller.clear_global_predictive_data()
    controller.clear_global_predictive_action()
    controller.clear_microbatch_buffer()
    controller.append_microbatch(empty_microbatch)
    controller.apply_predictive_train_mode("compute")
    assert controller.get_global_predictive_action() == RouterPredictiveAction.COMPUTE_PREDICTIVE_LOSS
    assert controller.used_valid_predictive_data is True
    assert state0.get_predictive_data() == (None, None, None, 1.0)


def test_predictive_router_replay_buffer_cursor():
    controller = get_predictive_replay_controller()
    controller.reset_registry()

    controller.clear_microbatch_buffer()
    controller.append_microbatch("mb0")
    controller.append_microbatch("mb1")

    assert controller.buffered_microbatch_count() == 2
    assert controller.pop_next_microbatch() == "mb0"
    assert controller.pop_next_microbatch() == "mb1"

    controller.clear_microbatch_buffer()
    assert controller.buffered_microbatch_count() == 0


def test_predictive_optimizer_group_disable_and_restore():
    predictor_param = torch.nn.Parameter(torch.ones(2, dtype=torch.float32))
    predictor_param.is_bias_predictor = True
    predictor_param.grad = torch.ones_like(predictor_param)
    predictor_param.main_grad = torch.ones_like(predictor_param)

    normal_param = torch.nn.Parameter(torch.ones(2, dtype=torch.float32))
    normal_param.grad = torch.ones_like(normal_param)

    optimizer = SimpleNamespace(
        param_groups=[
            {"params": [predictor_param], "lr": 3.0, "weight_decay": 0.1},
            {"params": [normal_param], "lr": 1.0, "weight_decay": 0.01},
        ]
    )

    clear_predictive_optimizer_grads(optimizer)
    assert predictor_param.grad is None
    assert torch.equal(predictor_param.main_grad, torch.zeros_like(predictor_param.main_grad))
    assert normal_param.grad is not None

    saved_groups = disable_predictive_param_groups(optimizer)
    assert optimizer.param_groups[0]["lr"] == 0.0
    assert optimizer.param_groups[0]["weight_decay"] == 0.0
    assert optimizer.param_groups[1]["lr"] == 1.0

    restore_predictive_param_groups(saved_groups)
    assert optimizer.param_groups[0]["lr"] == 3.0
    assert optimizer.param_groups[0]["weight_decay"] == 0.1


def test_compute_predictive_loss_variants_and_metrics():
    old_logits = torch.tensor([[0.0, 1.0], [1.0, 0.0]], dtype=torch.float32)
    current_logits = torch.tensor([[0.5, 1.5], [1.5, 0.5]], dtype=torch.float32)
    predicted_delta_logits = torch.tensor([[0.25, 0.75], [0.75, 0.25]], dtype=torch.float32)

    kl_loss = compute_predictive_loss(
        old_logits=old_logits,
        current_logits=current_logits,
        predicted_delta_logits=predicted_delta_logits,
        loss_type="kl",
    )
    kl_post_loss = compute_predictive_loss(
        old_logits=old_logits,
        current_logits=current_logits,
        predicted_delta_logits=predicted_delta_logits,
        loss_type="kl-post",
    )

    assert kl_loss.item() >= 0
    assert kl_post_loss.item() >= 0
    assert compute_predictive_bias_ratio(predicted_delta_logits, old_logits) > 0
    assert calculate_topk_accuracy(topk=1, logits1=old_logits + predicted_delta_logits, logits2=current_logits) == 1.0


def test_compute_predictive_layer_scale_schedules_decay_with_depth():
    assert PREDICTIVE_LAYER_SCALE_SCHEDULES == ("none", "linear_decay", "sqrt_decay", "cosine_decay")
    assert compute_predictive_layer_scale(layer_idx=0, num_layers=48, schedule="none", min_scale=0.25) == 1.0
    assert compute_predictive_layer_scale(layer_idx=0, num_layers=48, schedule="linear_decay", min_scale=0.25) == 1.0
    assert compute_predictive_layer_scale(layer_idx=47, num_layers=48, schedule="linear_decay", min_scale=0.25) == 0.25
    assert compute_predictive_layer_scale(layer_idx=47, num_layers=48, schedule="sqrt_decay", min_scale=0.4) == 0.4
    mid_linear = compute_predictive_layer_scale(layer_idx=24, num_layers=48, schedule="linear_decay", min_scale=0.5)
    mid_sqrt = compute_predictive_layer_scale(layer_idx=24, num_layers=48, schedule="sqrt_decay", min_scale=0.5)
    assert 0.5 < mid_sqrt < mid_linear < 1.0


def test_stabilize_predictive_delta_logits_applies_depth_gate():
    predicted_delta_logits = torch.full((2, 4), 4.0, dtype=torch.float32)
    reference_logits = torch.full((2, 4), 10.0, dtype=torch.float32)

    stabilized_delta_logits, metrics = stabilize_predictive_delta_logits(
        predicted_delta_logits=predicted_delta_logits,
        reference_logits=reference_logits,
        layer_idx=47,
        num_layers=48,
        layer_scale_schedule="linear_decay",
        layer_scale_min=0.5,
    )

    assert torch.allclose(stabilized_delta_logits, torch.full((2, 4), 2.0, dtype=torch.float32))
    assert metrics["predictive_raw_bias_to_logits_ratio"] == pytest.approx(0.4)
    assert metrics["predictive_stabilized_bias_to_logits_ratio"] == pytest.approx(0.2)
    assert metrics["predictive_layer_gate_scale"] == pytest.approx(0.5)
    assert metrics["predictive_stabilizer_scale"] == pytest.approx(0.5)


def test_stabilize_predictive_delta_logits_is_identity_without_layer_scale():
    reference_logits = torch.tensor([[3.0, 2.0, 0.0, -1.0]], dtype=torch.float32)
    predicted_delta_logits = torch.tensor([[2.0, -2.0, 2.0, -2.0]], dtype=torch.float32)

    stabilized_delta_logits, metrics = stabilize_predictive_delta_logits(
        predicted_delta_logits=predicted_delta_logits,
        reference_logits=reference_logits,
        layer_idx=0,
        num_layers=1,
        layer_scale_schedule="none",
        layer_scale_min=1.0,
    )

    assert torch.allclose(stabilized_delta_logits, predicted_delta_logits)
    assert metrics["predictive_layer_gate_scale"] == pytest.approx(1.0)
    assert metrics["predictive_stabilizer_scale"] == pytest.approx(1.0)


def test_ensure_bias_predictor_runtime_placement_preserves_parameter_identity():
    bias_predictor = torch.nn.Linear(4, 3, bias=False)
    predictor_weight = bias_predictor.weight
    predictor_weight.grad = torch.ones_like(predictor_weight)
    predictor_weight.main_grad = torch.ones_like(predictor_weight)
    reference_tensor = torch.randn(2, 4, dtype=torch.bfloat16)

    _ensure_bias_predictor_runtime_placement(
        bias_predictor=bias_predictor,
        reference_tensor=reference_tensor,
    )

    assert bias_predictor.weight is predictor_weight
    assert bias_predictor.weight.dtype == reference_tensor.dtype
    assert bias_predictor.weight.grad.dtype == reference_tensor.dtype
    assert bias_predictor.weight.main_grad.dtype == reference_tensor.dtype


def test_predictive_losses_can_build_gradients_inside_enable_grad_scope():
    bias_predictor = torch.nn.Linear(4, 3, bias=False).to(dtype=torch.bfloat16)
    old_inputs = torch.randn(2, 4, dtype=torch.bfloat16)
    old_logits = torch.randn(2, 3, dtype=torch.bfloat16)
    current_logits = torch.randn(2, 3, dtype=torch.bfloat16)

    with torch.no_grad():
        with torch.enable_grad():
            predicted_delta_logits = bias_predictor(old_inputs.detach())
            predictive_loss = compute_predictive_loss(
                old_logits=old_logits.detach(),
                current_logits=current_logits.detach(),
                predicted_delta_logits=predicted_delta_logits,
                loss_type="kl-post",
            )
            synthetic_loss = build_synthetic_predictive_loss(
                bias_predictor=bias_predictor,
                input_tensor=old_inputs,
            )

    assert predictive_loss.requires_grad is True
    assert synthetic_loss.requires_grad is True
