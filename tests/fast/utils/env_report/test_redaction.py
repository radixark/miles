import pytest

from miles.utils.env_report.redaction import _redact, redact_argv, redact_env_vars


class TestRedactArgv:
    def test_hides_the_value_of_a_secret_flag(self) -> None:
        """A hashed wandb_key in args is pointless while the same key sits in argv verbatim."""
        argv = redact_argv(["train.py", "--wandb-key", "s3cret", "--reward-key", "reward"])
        assert "s3cret" not in argv
        assert argv[:2] == ["train.py", "--wandb-key"]
        assert argv[-2:] == ["--reward-key", "reward"]

    def test_hides_the_value_of_an_inline_secret_flag(self) -> None:
        argv = redact_argv(["train.py", "--wandb-key=s3cret"])
        assert "s3cret" not in argv[1]
        assert argv[1].startswith("--wandb-key=redacted-sha256:")

    def test_keeps_an_argv_without_secrets_unchanged(self) -> None:
        argv = ["train.py", "--reward-key", "reward", "--lr=1e-4"]
        assert redact_argv(argv) == argv

    def test_a_trailing_secret_flag_hides_nothing_that_follows(self) -> None:
        assert redact_argv(["train.py", "--wandb-key"]) == ["train.py", "--wandb-key"]

    def test_hides_every_value_of_a_multi_valued_secret_flag(self) -> None:
        """The router takes its control-plane keys as a list, so hiding only the first leaks the rest."""
        argv = redact_argv(["train.py", "--router-control-plane-api-keys", "k1:n:r:s1", "k2:n:r:s2", "--lr", "1"])

        assert "s1" not in " ".join(argv) and "s2" not in " ".join(argv)
        assert argv[-2:] == ["--lr", "1"]

    def test_hides_the_secrets_inside_an_environment_valued_flag(self) -> None:
        """--train-env-vars carries a whole environment, and a credential in it is not a secret arg by name."""
        argv = redact_argv(["train.py", "--train-env-vars", '{"WANDB_API_KEY": "hunter2", "NCCL_DEBUG": "INFO"}'])

        assert "hunter2" not in argv[2]
        assert "INFO" in argv[2]


class TestRedact:
    def test_same_secret_hashes_to_same_digest(self) -> None:
        """Skew auditing needs to compare secrets across processes without revealing them."""
        assert _redact("hunter2") == _redact("hunter2")
        assert _redact("hunter2") != _redact("hunter3")
        assert "hunter2" not in _redact("hunter2")


class TestRedactEnvVars:
    @pytest.mark.parametrize(
        "name",
        [
            "WANDB_API_KEY",
            "HF_TOKEN",
            "MY_SECRET",
            "DB_PASSWORD",
            "PG_PASSWD",
            "GCP_CREDENTIALS",
            "hf_token",
            "NEON_DATABASE_URL",
        ],
    )
    def test_redacts_a_secret_named_variable(self, name: str) -> None:
        """Every secret-ish name suffix is redacted, whatever its case."""
        assert "s3cret" not in redact_env_vars({name: "s3cret"})[name]

    @pytest.mark.parametrize(
        "name", ["TOKENIZERS_PARALLELISM", "KEYRING_PATH", "SSH_KEY_FILE", "CUDA_VISIBLE_DEVICES"]
    )
    def test_keeps_a_variable_that_merely_contains_a_secret_word(self, name: str) -> None:
        """Hashing every name containing 'key' would erase exactly the values an audit reads."""
        assert redact_env_vars({name: "plain"})[name] == "plain"

    def test_sorts_and_keeps_every_variable(self) -> None:
        """The report dumps all env vars, so redaction must never drop one."""
        redacted = redact_env_vars({"ZZZ": "z", "HF_TOKEN": "t", "RANK": "3"})
        assert list(redacted.keys()) == ["HF_TOKEN", "RANK", "ZZZ"]
        assert redacted["RANK"] == "3"
