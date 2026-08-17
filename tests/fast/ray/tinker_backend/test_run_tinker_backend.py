from examples.tinker_backend import run_tinker_backend


def test_static_batch_mode_omits_packed_sequence_args(monkeypatch):
    captured = {}

    def capture_execute_train(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(run_tinker_backend.U, "execute_train", capture_execute_train)

    args = run_tinker_backend.ScriptArgs(use_dynamic_batch_size=False)
    run_tinker_backend._serve(args, service=True)

    train_args = captured["train_args"]
    assert "--use-dynamic-batch-size" not in train_args
    assert "--max-tokens-per-gpu" not in train_args


def test_dynamic_batch_mode_remains_the_default(monkeypatch):
    captured = {}

    def capture_execute_train(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(run_tinker_backend.U, "execute_train", capture_execute_train)

    args = run_tinker_backend.ScriptArgs()
    run_tinker_backend._serve(args, service=True)

    train_args = captured["train_args"]
    assert "--use-dynamic-batch-size" in train_args
    assert "--max-tokens-per-gpu 9216" in train_args
