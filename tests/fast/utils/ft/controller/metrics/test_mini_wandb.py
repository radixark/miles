from datetime import datetime, timedelta, timezone

from miles.utils.ft.controller.metrics.mini_wandb import MiniWandb


class TestMiniWandbLogAndQuery:
    def test_log_step_then_query_last_n(self) -> None:
        wandb = MiniWandb(active_run_id="run-1")
        wandb.log_step(run_id="run-1", step=1, metrics={"loss": 2.5})
        wandb.log_step(run_id="run-1", step=2, metrics={"loss": 2.3})
        wandb.log_step(run_id="run-1", step=3, metrics={"loss": 2.1})

        result = wandb.query_last_n_steps(metric_name="loss", last_n=2)
        assert len(result) == 2
        assert result[0] == (2, 2.3)
        assert result[1] == (3, 2.1)

    def test_run_id_mismatch_discards(self) -> None:
        wandb = MiniWandb(active_run_id="run-1")
        wandb.log_step(run_id="run-2", step=1, metrics={"loss": 999.0})

        result = wandb.query_last_n_steps(metric_name="loss", last_n=10)
        assert result == []

    def test_latest_returns_most_recent_value(self) -> None:
        wandb = MiniWandb(active_run_id="run-1")
        wandb.log_step(run_id="run-1", step=1, metrics={"loss": 3.0})
        wandb.log_step(run_id="run-1", step=2, metrics={"loss": 2.0})

        assert wandb.latest(metric_name="loss") == 2.0

    def test_latest_missing_metric_returns_none(self) -> None:
        wandb = MiniWandb(active_run_id="run-1")
        wandb.log_step(run_id="run-1", step=1, metrics={"loss": 1.0})

        assert wandb.latest(metric_name="grad_norm") is None

    def test_latest_empty_returns_none(self) -> None:
        wandb = MiniWandb(active_run_id="run-1")
        assert wandb.latest(metric_name="loss") is None

    def test_clear_empties_all_data(self) -> None:
        wandb = MiniWandb(active_run_id="run-1")
        wandb.log_step(run_id="run-1", step=1, metrics={"loss": 1.0})

        wandb.clear()

        assert wandb.query_last_n_steps(metric_name="loss", last_n=10) == []
        assert wandb.latest(metric_name="loss") is None


class TestMiniWandbTimeWindow:
    def test_query_time_window_returns_recent(self) -> None:
        wandb = MiniWandb(active_run_id="run-1")
        wandb.log_step(run_id="run-1", step=1, metrics={"loss": 3.0})
        wandb.log_step(run_id="run-1", step=2, metrics={"loss": 2.5})

        result = wandb.query_time_window(
            metric_name="loss", window=timedelta(minutes=5)
        )
        assert len(result) == 2
        assert result[0][0] == 1  # step
        assert result[0][2] == 3.0  # value
        assert result[1][0] == 2
        assert result[1][2] == 2.5


class TestMiniWandbTimeWindowEdgeCases:
    def test_empty_returns_empty(self) -> None:
        wandb = MiniWandb(active_run_id="run-1")
        result = wandb.query_time_window(
            metric_name="loss", window=timedelta(minutes=5)
        )
        assert result == []

    def test_missing_metric_returns_empty(self) -> None:
        wandb = MiniWandb(active_run_id="run-1")
        wandb.log_step(run_id="run-1", step=1, metrics={"loss": 1.0})

        result = wandb.query_time_window(
            metric_name="nonexistent", window=timedelta(minutes=5)
        )
        assert result == []

    def test_multi_metric_per_step(self) -> None:
        wandb = MiniWandb(active_run_id="run-1")
        wandb.log_step(
            run_id="run-1", step=1,
            metrics={"loss": 2.0, "grad_norm": 1.5, "lr": 0.001},
        )
        wandb.log_step(
            run_id="run-1", step=2,
            metrics={"loss": 1.8, "grad_norm": 1.3, "lr": 0.001},
        )

        loss = wandb.query_last_n_steps(metric_name="loss", last_n=10)
        grad = wandb.query_last_n_steps(metric_name="grad_norm", last_n=10)
        lr = wandb.query_last_n_steps(metric_name="lr", last_n=10)

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
                run_id="run-1", step=i, metrics={"loss": float(i)}
            )

        result = wandb.query_last_n_steps(metric_name="loss", last_n=10)
        assert len(result) == 3
        assert result[0][0] == 2  # oldest remaining step
        assert result[2][0] == 4  # newest step


class TestMiniWandbMaxAgeEviction:
    def test_max_age_evicts_old_records(self) -> None:
        wandb = MiniWandb(active_run_id="run-1", max_age=timedelta(seconds=1))

        old_time = datetime.now(timezone.utc) - timedelta(seconds=2)
        wandb.log_step(
            run_id="run-1", step=1, metrics={"loss": 3.0},
            receive_time=old_time,
        )
        wandb.log_step(run_id="run-1", step=2, metrics={"loss": 2.0})

        result = wandb.query_last_n_steps(metric_name="loss", last_n=10)
        assert len(result) == 1
        assert result[0] == (2, 2.0)


class TestMiniWandbNoActiveRunId:
    def test_no_active_run_id_accepts_all(self) -> None:
        wandb = MiniWandb(active_run_id=None)
        wandb.log_step(run_id="any-run", step=1, metrics={"loss": 1.0})

        assert wandb.latest(metric_name="loss") == 1.0

    def test_set_active_run_id(self) -> None:
        wandb = MiniWandb(active_run_id=None)
        wandb.log_step(run_id="run-1", step=1, metrics={"loss": 1.0})

        wandb.set_active_run_id("run-2")
        wandb.log_step(run_id="run-1", step=2, metrics={"loss": 2.0})
        wandb.log_step(run_id="run-2", step=3, metrics={"loss": 3.0})

        result = wandb.query_last_n_steps(metric_name="loss", last_n=10)
        # step 1 was accepted (no active_run_id), step 2 discarded (wrong run_id), step 3 accepted
        assert len(result) == 2
        assert result[0] == (1, 1.0)
        assert result[1] == (3, 3.0)


class TestMiniWandbStepMonotonicity:
    def test_out_of_order_step_discarded(self) -> None:
        wandb = MiniWandb(active_run_id="run-1")
        wandb.log_step(run_id="run-1", step=5, metrics={"loss": 1.0})
        wandb.log_step(run_id="run-1", step=3, metrics={"loss": 999.0})
        wandb.log_step(run_id="run-1", step=10, metrics={"loss": 0.5})

        result = wandb.query_last_n_steps(metric_name="loss", last_n=10)
        assert len(result) == 2
        assert result[0] == (5, 1.0)
        assert result[1] == (10, 0.5)

    def test_duplicate_step_discarded(self) -> None:
        wandb = MiniWandb(active_run_id="run-1")
        wandb.log_step(run_id="run-1", step=1, metrics={"loss": 2.0})
        wandb.log_step(run_id="run-1", step=1, metrics={"loss": 3.0})

        result = wandb.query_last_n_steps(metric_name="loss", last_n=10)
        assert len(result) == 1
        assert result[0] == (1, 2.0)

    def test_clear_resets_last_step(self) -> None:
        wandb = MiniWandb(active_run_id="run-1")
        wandb.log_step(run_id="run-1", step=10, metrics={"loss": 1.0})
        wandb.clear()
        wandb.log_step(run_id="run-1", step=1, metrics={"loss": 2.0})

        result = wandb.query_last_n_steps(metric_name="loss", last_n=10)
        assert len(result) == 1
        assert result[0] == (1, 2.0)

    def test_set_active_run_id_resets_last_step(self) -> None:
        wandb = MiniWandb(active_run_id="run-1")
        wandb.log_step(run_id="run-1", step=10, metrics={"loss": 1.0})

        wandb.set_active_run_id("run-2")
        wandb.log_step(run_id="run-2", step=1, metrics={"loss": 2.0})

        result = wandb.query_last_n_steps(metric_name="loss", last_n=10)
        assert len(result) == 2
        assert result[1] == (1, 2.0)

    def test_set_same_run_id_does_not_reset(self) -> None:
        wandb = MiniWandb(active_run_id="run-1")
        wandb.log_step(run_id="run-1", step=10, metrics={"loss": 1.0})

        wandb.set_active_run_id("run-1")
        wandb.log_step(run_id="run-1", step=5, metrics={"loss": 999.0})

        result = wandb.query_last_n_steps(metric_name="loss", last_n=10)
        assert len(result) == 1
        assert result[0] == (10, 1.0)
