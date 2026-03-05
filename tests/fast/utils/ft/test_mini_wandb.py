from datetime import datetime, timedelta, timezone

from miles.utils.ft.controller.mini_wandb import MiniWandb


class TestMiniWandbLogAndQuery:
    def test_log_step_then_query_last_n(self) -> None:
        wandb = MiniWandb(active_run_id="run-1")
        wandb.log_step(run_id="run-1", rank=0, step=1, metrics={"loss": 2.5})
        wandb.log_step(run_id="run-1", rank=0, step=2, metrics={"loss": 2.3})
        wandb.log_step(run_id="run-1", rank=0, step=3, metrics={"loss": 2.1})

        result = wandb.query_last_n_steps(metric_name="loss", rank=0, last_n=2)
        assert len(result) == 2
        assert result[0] == (2, 2.3)
        assert result[1] == (3, 2.1)

    def test_run_id_mismatch_discards(self) -> None:
        wandb = MiniWandb(active_run_id="run-1")
        wandb.log_step(run_id="run-2", rank=0, step=1, metrics={"loss": 999.0})

        result = wandb.query_last_n_steps(metric_name="loss", rank=0, last_n=10)
        assert result == []

    def test_latest_returns_most_recent_value(self) -> None:
        wandb = MiniWandb(active_run_id="run-1")
        wandb.log_step(run_id="run-1", rank=0, step=1, metrics={"loss": 3.0})
        wandb.log_step(run_id="run-1", rank=0, step=2, metrics={"loss": 2.0})

        assert wandb.latest(metric_name="loss", rank=0) == 2.0

    def test_latest_missing_metric_returns_none(self) -> None:
        wandb = MiniWandb(active_run_id="run-1")
        wandb.log_step(run_id="run-1", rank=0, step=1, metrics={"loss": 1.0})

        assert wandb.latest(metric_name="grad_norm", rank=0) is None

    def test_latest_missing_rank_returns_none(self) -> None:
        wandb = MiniWandb(active_run_id="run-1")
        assert wandb.latest(metric_name="loss", rank=99) is None

    def test_clear_empties_all_data(self) -> None:
        wandb = MiniWandb(active_run_id="run-1")
        wandb.log_step(run_id="run-1", rank=0, step=1, metrics={"loss": 1.0})
        wandb.log_step(run_id="run-1", rank=1, step=1, metrics={"loss": 2.0})

        wandb.clear()

        assert wandb.query_last_n_steps(metric_name="loss", rank=0, last_n=10) == []
        assert wandb.latest(metric_name="loss", rank=1) is None


class TestMiniWandbTimeWindow:
    def test_query_time_window_returns_recent(self) -> None:
        wandb = MiniWandb(active_run_id="run-1")
        wandb.log_step(run_id="run-1", rank=0, step=1, metrics={"loss": 3.0})
        wandb.log_step(run_id="run-1", rank=0, step=2, metrics={"loss": 2.5})

        result = wandb.query_time_window(
            metric_name="loss", rank=0, window=timedelta(minutes=5)
        )
        assert len(result) == 2
        assert result[0][0] == 1  # step
        assert result[0][2] == 3.0  # value
        assert result[1][0] == 2
        assert result[1][2] == 2.5


class TestMiniWandbTimeWindowEdgeCases:
    def test_missing_rank_returns_empty(self) -> None:
        wandb = MiniWandb(active_run_id="run-1")
        result = wandb.query_time_window(
            metric_name="loss", rank=99, window=timedelta(minutes=5)
        )
        assert result == []

    def test_missing_metric_returns_empty(self) -> None:
        wandb = MiniWandb(active_run_id="run-1")
        wandb.log_step(run_id="run-1", rank=0, step=1, metrics={"loss": 1.0})

        result = wandb.query_time_window(
            metric_name="nonexistent", rank=0, window=timedelta(minutes=5)
        )
        assert result == []

    def test_multi_metric_per_step(self) -> None:
        wandb = MiniWandb(active_run_id="run-1")
        wandb.log_step(
            run_id="run-1", rank=0, step=1,
            metrics={"loss": 2.0, "grad_norm": 1.5, "lr": 0.001},
        )
        wandb.log_step(
            run_id="run-1", rank=0, step=2,
            metrics={"loss": 1.8, "grad_norm": 1.3, "lr": 0.001},
        )

        loss = wandb.query_last_n_steps(metric_name="loss", rank=0, last_n=10)
        grad = wandb.query_last_n_steps(metric_name="grad_norm", rank=0, last_n=10)
        lr = wandb.query_last_n_steps(metric_name="lr", rank=0, last_n=10)

        assert len(loss) == 2
        assert loss[0] == (1, 2.0)
        assert loss[1] == (2, 1.8)
        assert len(grad) == 2
        assert grad[0] == (1, 1.5)
        assert len(lr) == 2
        assert lr[0][1] == 0.001


class TestMiniWandbRingBuffer:
    def test_max_steps_evicts_oldest(self) -> None:
        wandb = MiniWandb(active_run_id="run-1", max_steps=3)
        for i in range(5):
            wandb.log_step(
                run_id="run-1", rank=0, step=i, metrics={"loss": float(i)}
            )

        result = wandb.query_last_n_steps(metric_name="loss", rank=0, last_n=10)
        assert len(result) == 3
        assert result[0][0] == 2  # oldest remaining step
        assert result[2][0] == 4  # newest step


class TestMiniWandbMaxAgeEviction:
    def test_max_age_evicts_old_records(self) -> None:
        wandb = MiniWandb(active_run_id="run-1", max_age=timedelta(seconds=1))

        old_time = datetime.now(timezone.utc) - timedelta(seconds=2)
        wandb.log_step(
            run_id="run-1", rank=0, step=1, metrics={"loss": 3.0},
            receive_time=old_time,
        )
        wandb.log_step(run_id="run-1", rank=0, step=2, metrics={"loss": 2.0})

        result = wandb.query_last_n_steps(metric_name="loss", rank=0, last_n=10)
        assert len(result) == 1
        assert result[0] == (2, 2.0)


class TestMiniWandbMultiRank:
    def test_multi_rank_data_independent(self) -> None:
        wandb = MiniWandb(active_run_id="run-1")
        wandb.log_step(run_id="run-1", rank=0, step=1, metrics={"loss": 1.0})
        wandb.log_step(run_id="run-1", rank=1, step=1, metrics={"loss": 99.0})

        assert wandb.latest(metric_name="loss", rank=0) == 1.0
        assert wandb.latest(metric_name="loss", rank=1) == 99.0

    def test_query_last_n_per_rank(self) -> None:
        wandb = MiniWandb(active_run_id="run-1")
        for i in range(3):
            wandb.log_step(
                run_id="run-1", rank=0, step=i, metrics={"loss": float(i)}
            )
            wandb.log_step(
                run_id="run-1", rank=1, step=i, metrics={"loss": float(i + 100)}
            )

        rank0 = wandb.query_last_n_steps(metric_name="loss", rank=0, last_n=2)
        rank1 = wandb.query_last_n_steps(metric_name="loss", rank=1, last_n=2)
        assert rank0[-1][1] == 2.0
        assert rank1[-1][1] == 102.0


class TestMiniWandbNoActiveRunId:
    def test_no_active_run_id_accepts_all(self) -> None:
        wandb = MiniWandb(active_run_id=None)
        wandb.log_step(run_id="any-run", rank=0, step=1, metrics={"loss": 1.0})

        assert wandb.latest(metric_name="loss", rank=0) == 1.0

    def test_set_active_run_id(self) -> None:
        wandb = MiniWandb(active_run_id=None)
        wandb.log_step(run_id="run-1", rank=0, step=1, metrics={"loss": 1.0})

        wandb.set_active_run_id("run-2")
        wandb.log_step(run_id="run-1", rank=0, step=2, metrics={"loss": 2.0})
        wandb.log_step(run_id="run-2", rank=0, step=3, metrics={"loss": 3.0})

        result = wandb.query_last_n_steps(metric_name="loss", rank=0, last_n=10)
        # step 1 was accepted (no active_run_id), step 2 discarded (wrong run_id), step 3 accepted
        assert len(result) == 2
        assert result[0] == (1, 1.0)
        assert result[1] == (3, 3.0)
