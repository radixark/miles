"""The launcher's argv, parsed by Miles: RePrompt's GRPO settings plus a frozen diffusion generator behind Miles' router."""

import shlex
from types import SimpleNamespace

import pytest
from examples.experimental.prompt_enhancer.reward import PromptEnhancerLogConfig, PromptEnhancerRewardConfig
from examples.experimental.prompt_enhancer.run_qwen25_3b_prompt_enhancer import ScriptArgs, _train_args, prepare
from tests.fast.fixtures.args_fixtures import parse_megatron_test_config
from transformers import Qwen2Config

from miles.backends.sglang_utils.sglang_api_client import WorkerType
from miles.backends.sglang_utils.sglang_config import EngineKind


def _parse(tmp_path, monkeypatch, **script_args):
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    Qwen2Config(
        hidden_size=128,
        num_attention_heads=2,
        num_key_value_heads=2,
        num_hidden_layers=1,
        intermediate_size=256,
        rms_norm_eps=1e-5,
        tie_word_embeddings=True,
    ).save_pretrained(tmp_path / "Qwen2.5-3B-Instruct")
    (tmp_path / "Qwen2.5-3B-Instruct_torch_dist").mkdir(exist_ok=True)
    args = ScriptArgs(run_id="test", model_dir=str(tmp_path), **script_args)
    return parse_megatron_test_config("--ffn-hidden-size", "256", *shlex.split(_train_args(args)))


@pytest.fixture
def config(monkeypatch, tmp_path):
    return _parse(tmp_path, monkeypatch)


# The README's ImageReward command: scorer URLs and no --scorer.
_IMAGEREWARD = dict(scorer_urls="http://a:1,http://b:2")


def test_the_grpo_settings_are_reprompts(config):
    assert (config.rollout_batch_size, config.n_samples_per_prompt, config.global_batch_size) == (32, 4, 128)
    assert config.num_epoch == 3
    assert (config.rollout_temperature, config.rollout_max_response_len) == (0.9, 256)
    assert (config.rollout_top_p, config.rollout_top_k) == (1.0, -1)
    assert config.rollout_skip_special_tokens
    assert config.advantage_estimator == "grpo"
    assert (config.use_kl_loss, config.kl_loss_coef, config.kl_loss_type) == (True, 0.04, "k3")
    assert config.eps_clip == 0.2 and config.eps_clip_high == 0.2
    megatron = config.raw_megatron.base_args
    assert (megatron["lr"], megatron["min_lr"], megatron["lr_decay_style"]) == (2e-6, 0.0, "linear")
    assert (megatron["lr_warmup_iters"], megatron["weight_decay"], megatron["clip_grad"]) == (0, 0.0, 1.0)
    assert (megatron["adam_beta1"], megatron["adam_beta2"], megatron["adam_eps"]) == (0.9, 0.999, 1e-8)
    assert (megatron["hidden_dropout"], megatron["attention_dropout"]) == (0.0, 0.0)


def test_the_reward_and_log_hooks_get_their_settings(config):
    assert config.custom_rm_path.path == "examples.experimental.prompt_enhancer.reward.prompt_enhancer_rm"
    assert config.custom_rm_path.config == PromptEnhancerRewardConfig(
        pe_generator_model="generator",
        pe_width=512,
        pe_height=512,
        pe_num_inference_steps=9,
        pe_guidance_scale=0.0,
        pe_seed_per_sample=False,
        pe_scorer_pool="hpsv3",
        pe_image_weight=0.08,
    )
    assert not config.group_rm
    log_hook = config.custom_rollout_log_function_path
    assert log_hook.config == PromptEnhancerLogConfig(pe_sample_log="/root/shared_data/test/samples")


def test_the_generator_is_a_frozen_diffusion_model_after_the_policy(config, tmp_path):
    policy, generator = config.sglang.models
    assert policy.name == "policy" and policy.update_weights
    assert generator.name == "generator" and generator.update_weights is False
    (group,) = generator.server_groups
    assert (group.engine_kind, group.num_gpus_per_engine, group.worker_type) == (
        EngineKind.DIFFUSION,
        1,
        WorkerType.REGULAR,
    )
    assert group.model_path == f"{tmp_path}/Z-Image-Turbo"
    assert config.router == "miles"


def test_the_job_holds_the_trainer_the_policy_the_generator_engines_and_the_scorer(config):
    assert config.rollout_num_gpus == 6 and config.actor_num_gpus_per_node == 1 and not config.colocate
    assert config.num_gpus_per_node == 8
    assert config.reward_scaling.pools["hpsv3"].model_dump() == dict(num_gpus=1, gpu_offset=6)


def test_hpsv3_scores_on_a_reward_pool_by_default(config, tmp_path):
    (pool,) = config.reward_pools
    assert pool.scorer_path == "miles.rollout.rm_hub.image.hpsv3.HPSv3Scorer"
    assert pool.scorer_kwargs == dict(
        checkpoint_path=f"{tmp_path}/HPSv3/HPSv3.safetensors", base_model_path=f"{tmp_path}/Qwen2-VL-7B-Instruct"
    )
    assert (pool.name, pool.num_workers, pool.uses_gpu, pool.batch_size, pool.max_wait_ms) == (
        "hpsv3",
        1,
        True,
        16,
        50,
    )


def test_scorer_urls_alone_keep_the_imagereward_job(monkeypatch, tmp_path):
    config = _parse(tmp_path, monkeypatch, **_IMAGEREWARD)

    reward_config = config.custom_rm_path.config
    assert (reward_config.pe_scorer_pool, reward_config.pe_scorer_urls) == (None, "http://a:1,http://b:2")
    assert reward_config.pe_image_weight == 0.25
    assert config.reward_pools == [] and config.reward_scaling.num_gpus == 0
    assert (config.num_gpus_per_node, config.rollout_num_gpus) == (7, 6)


def test_imagereward_without_urls_uses_the_local_scorer_server():
    args = ScriptArgs(scorer="imagereward")

    assert (args.scorer_urls, args.scorer_gpus, args.num_gpus_per_node) == ("http://127.0.0.1:18200", 0, 7)


def test_scorer_urls_with_hpsv3_are_refused():
    with pytest.raises(AssertionError, match="ImageReward's scorers"):
        ScriptArgs(scorer="hpsv3", scorer_urls="http://a:1")


@pytest.mark.parametrize("scorer, scorer_gpus", [("hpsv3", 0), ("imagereward", 1), ("pickscore", 1)])
def test_a_scorer_without_its_gpus_is_refused(scorer, scorer_gpus):
    with pytest.raises(AssertionError, match="--scorer"):
        ScriptArgs(scorer=scorer, scorer_gpus=scorer_gpus, num_gpus_per_node=7 + scorer_gpus)


def test_the_parity_flags_reach_the_run(monkeypatch, tmp_path):
    config = _parse(tmp_path, monkeypatch, seed_per_sample=True, training_seed=43)

    assert config.custom_rm_path.config.pe_seed_per_sample
    assert config.rollout_seed == 43 and config.raw_megatron.base_args["seed"] == 43


def test_without_a_training_seed_miles_defaults_stay(config):
    assert config.rollout_seed == 42 and config.raw_megatron.base_args["seed"] == 1234


def test_a_split_that_does_not_fill_the_job_is_refused():
    with pytest.raises(AssertionError, match="must add up to"):
        ScriptArgs(generator_gpus=2, num_gpus_per_node=8)


def test_the_checkpoint_is_converted_on_the_trainers_gpus(monkeypatch):
    conversions, commands = [], []
    backend = SimpleNamespace(exec_command_cpu=commands.append, convert_checkpoint=lambda **kw: conversions.append(kw))
    monkeypatch.setattr(ScriptArgs, "create_backend", lambda self: backend)

    prepare(ScriptArgs(model_dir="/m"))

    (conversion,) = conversions
    assert conversion["num_gpus_per_node"] == 1
    assert (conversion["hf_checkpoint"], conversion["dir_dst"]) == ("/m/Qwen2.5-3B-Instruct", "/m")
    assert "hf download MizzenAI/HPSv3 --local-dir /m/HPSv3" in commands
    assert "hf download Qwen/Qwen2-VL-7B-Instruct --local-dir /m/Qwen2-VL-7B-Instruct" in commands
