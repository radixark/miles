from __future__ import annotations

from argparse import ArgumentParser, Namespace
from typing import Any

from sglang_router.launch_router import RouterArgs
from tests.fast.fixtures.args_fixtures import parser_defaults, resolve_parse_boundary_configs

from miles.utils.args.component_rollout import InferenceRuntimeMutState


def make_args(**overrides: Any) -> Namespace:
    """Args namespace covering every field touched by ``miles/ray/rollout/``.
    Adding a new field is fine; deleting one likely breaks tests."""
    parser: ArgumentParser = ArgumentParser()
    RouterArgs.add_cli_args(parser, use_router_prefix=True, exclude_host_port=True)
    router_defaults: dict[str, Any] = vars(parser.parse_args([]))
    defaults: dict[str, Any] = dict(
        # rollout core
        rollout_data_yaml_config=None,
        rollout_num_gpus=8,
        rollout_num_gpus_per_engine=1,
        eval_num_gpus=0,
        eval_num_gpus_per_engine=1,
        eval_uses_snapshots=False,
        num_gpus_per_node=8,
        rollout_batch_size=8,
        n_samples_per_prompt=4,
        n_samples_per_eval_prompt=4,
        rollout_max_response_len=512,
        rollout_temperature=1.0,
        rollout_top_p=1.0,
        rollout_top_k=-1,
        rollout_top_logprobs_num=0,
        rollout_sampling_logprobs_mode="selected",
        use_sampling_support_replay=False,
        over_sampling_batch_size=None,
        rollout_global_dataset=False,
        num_rollout=1,
        update_weights_interval=1,
        check_weight_update_equal=False,
        check_weight_update_skip_list=None,
        check_weight_update_selector="all",
        # batch / training
        global_batch_size=8,
        use_dynamic_global_batch_size=False,
        wandb_always_use_train_step=False,
        disable_rollout_trim_samples=False,
        balance_data=False,
        delay_split_train_data_by_dp=False,
        # object store
        object_store_backend="ray",
        worker_comm_backend="ray",
        mooncake_store_init_kwargs=None,
        mooncake_replica_num=1,
        # advantage / reward
        advantage_estimator="grpo",
        rewards_normalization=True,
        grpo_std_normalization=False,
        length_penalty_coeff=0.0,
        length_penalty_min_tokens=1,
        reward_key=None,
        log_reward_category=None,
        log_passrate=False,
        pin_rollout_manager_to_head=False,
        cluster_backend="ray",
        # placement / colocation
        debug_train_only=False,
        debug_rollout_only=False,
        debug_skip_weight_update=False,
        colocate=False,
        actor_num_nodes=1,
        actor_num_gpus_per_node=8,
        indep_dp=False,
        train_backend="megatron",
        kl_coef=0,
        use_kl_loss=False,
        use_opd=False,
        opd_type="megatron",
        train_env_vars={},
        dumper_source_patcher_config_train=None,
        offload_train=False,
        offload_train_target="cpu",
        offload_train_disk_dir="/tmp/offload",
        offload_train_disk_chunk_mb=64,
        critic_num_nodes=0,
        critic_num_gpus_per_node=0,
        use_critic=False,
        megatron_config=None,
        # sglang router
        sglang_router_ip=None,
        sglang_router_port=None,
        sglang_router_policy=None,
        sglang_router_request_timeout_secs=600,
        sglang_dp_size=1,
        sglang_pp_size=1,
        sglang_pp_layer_partition=None,
        sglang_ep_size=1,
        sglang_api_key=None,
        multi_lora_n_adapters=0,
        target_modules=None,
        sglang_speculative_algorithm=None,
        sglang_config=None,
        sglang_model_routers=None,
        prefill_num_servers=None,
        # routers / session server
        router="sglang",
        miles_router_timeout=None,
        miles_router_max_connections=None,
        miles_router_health_check_failure_threshold=3,
        sglang_server_concurrency=512,
        use_session_server=False,
        use_rollout_routing_replay=False,
        multimodal_token_space_processing=False,
        sglang_enable_prefill_weight_versions=False,
        session_server_ip=None,
        session_server_external_host=None,
        session_server_port=None,
        session_server_workers=1,
        run_uuid="0123456789abcdef",
        # deployment
        deploy_component="all",
        deploy_instance_id=None,
        init_expected_num_cells=None,
        trainer_controller_addrs=None,
        inference_controller_addr=None,
        # external rollout
        rollout_external=False,
        rollout_external_engine_addrs=None,
        rollout_external_router_pd=False,
        custom_inference_engine_provider_path="miles.ray.specs.inference.backend_inference_engine_provider",
        # offload / fault tolerance
        offload_rollout=False,
        use_fault_tolerance=False,
        ft_components=[],
        rollout_health_check_interval=30.0,
        rollout_health_check_timeout=30.0,
        rollout_health_check_first_wait=0.0,
        rollout_health_check_failure_threshold=1,
        # engine launch command
        seed=42,
        fp16=False,
        use_rollout_indexer_replay=False,
        rollout_indexer_topk_num_streams=None,
        replay_layer_splits=[],
        use_output_store=False,
        env_report_interval_seconds=3600.0,
        # checkpoint / data source
        hf_checkpoint="/fake/model",
        lora_rank=0,
        rollout_function_path="miles.rollout.sglang_rollout.generate_rollout",
        eval_function_path="miles.rollout.sglang_rollout.eval_generate_rollout",
        data_source_path="miles.data.dummy.DummyDataSource",
        custom_reward_post_process_path=None,
        custom_convert_samples_to_train_data_path=None,
        custom_rollout_log_function_path=None,
        custom_eval_rollout_log_function_path=None,
        # debug data
        save_debug_rollout_data=None,
        save_debug_trajectory_data=None,
        load_debug_rollout_data=None,
        load_debug_rollout_data_subsample=None,
        ci_inject_rollout_data_path=None,
        ci_inject_rollout_data_start_rollout_id=None,
        ci_inject_rollout_data_min_match_ratio=0.9,
        # event checkpointing (event_logger.restore/snapshot in RolloutExecutor)
        save_debug_event_data=None,
        enable_sample_ownership_checker=False,
        sample_ownership_grace_steps=10,
        load=None,
        requested_load=None,
        save=None,
        # CI
        ci_test=False,
        ci_tito_special_token_count_threshold=0.0,
        ci_assert_prefill_lag_max=None,
        # dumper (sglang debug dumper integration)
        dumper_enable=False,
    )
    defaults.update(router_defaults)
    defaults["inference_runtime_mut_state"] = InferenceRuntimeMutState()
    defaults.update(overrides)
    defaults.setdefault("starts_inference_engines", not defaults["debug_train_only"] or defaults["eval_num_gpus"] > 0)
    if defaults["debug_train_only"]:
        defaults["rollout_num_gpus"] = 0
    return resolve_parse_boundary_configs(Namespace(**{**parser_defaults(), **defaults}))
