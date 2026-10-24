from argparse import Namespace

import pytest
from tests.fast.backends.training_utils.backend_owner_fixtures import native_backend_owner_state

from miles.backends.training_utils.metrics import log_utils
from miles.utils.timer import Timer

__all__ = ["native_backend_owner_state"]


class TestBackendFlopsLogging:
    def test_performance_log_computes_model_flops_from_the_backend_config(
        self, native_backend_owner_state: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The real performance path reads model dimensions from backend configuration through FLOPs accounting."""
        timer = Timer()
        monkeypatch.setattr(timer, "timers", {"actor_train": 2.0})
        monkeypatch.setattr(timer, "seq_lens", [4], raising=False)
        monkeypatch.setattr(log_utils.dist, "get_world_size", lambda: 2)
        logs: list[dict] = []
        monkeypatch.setattr(log_utils.tracking, "log", lambda args, metrics, step_key: logs.append(metrics))
        args = Namespace(
            backend=Namespace(
                hidden_size=2,
                num_attention_heads=1,
                num_query_groups=1,
                vocab_size=8,
                ffn_hidden_size=4,
                num_experts=None,
                num_layers=1,
                q_lora_rank=None,
                kv_lora_rank=None,
                kv_channels=2,
                qk_pos_emb_head_dim=0,
                v_head_dim=0,
            ),
            mfu_peak_tflops=100.0,
            trainer_model_id=None,
            wandb_always_use_train_step=False,
        )

        log_utils.log_perf_data(rollout_id=1, args=args)

        [metrics] = logs
        assert metrics["perf/actor_train_tflops"] == pytest.approx(3.84e-10, rel=1e-12, abs=0)
        assert metrics["perf/actor_train_mfu"] == pytest.approx(3.84e-12, rel=1e-12, abs=0)
        assert metrics["perf/actor_train_tok_per_s"] == 2.0
