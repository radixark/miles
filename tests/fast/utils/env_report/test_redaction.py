import pytest

from miles.utils.env_report.redaction import _redact, redact_argv, redact_env_vars, redact_server_info


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


class TestRedactServerInfo:
    def test_redacts_the_keys_repeated_inside_every_internal_state(self) -> None:
        """Each internal state is a whole ServerArgs, so the credentials appear once per scheduler."""
        redacted = redact_server_info(
            {"internal_states": [{"api_key": "a", "waiting_queue": 0}, {"ssl_keyfile_password": "c"}]}
        )

        assert redacted["internal_states"][0]["api_key"].startswith("redacted-sha256:")
        assert redacted["internal_states"][0]["waiting_queue"] == 0
        assert redacted["internal_states"][1]["ssl_keyfile_password"].startswith("redacted-sha256:")

    def test_redacts_an_internal_state_that_carries_no_environment(self) -> None:
        """An engine too old to answer with env_vars still repeats its ServerArgs in every state."""
        redacted = redact_server_info({"internal_states": [{"api_key": "a"}]})

        assert redacted["internal_states"][0]["api_key"].startswith("redacted-sha256:")

    def test_redacts_the_engine_keys_that_arrive_without_a_miles_prefix(self) -> None:
        """The engine reports its own ServerArgs, whose names lack the sglang_ prefix miles args carry."""
        redacted = redact_server_info(
            {"api_key": "a", "admin_api_key": "b", "ssl_keyfile_password": "c", "model_path": "/models/qwen"}
        )

        assert all(redacted[name].startswith("redacted-sha256:") for name in ("api_key", "admin_api_key"))
        assert redacted["ssl_keyfile_password"].startswith("redacted-sha256:")
        assert redacted["model_path"] == "/models/qwen"

    def test_redacts_the_environment_of_every_scheduler(self) -> None:
        """internal_states holds one entry per DP scheduler, and each carries a whole environment."""
        redacted = redact_server_info(
            {
                "internal_states": [
                    {"env_vars": {"HF_TOKEN": "t0", "RANK": "0"}},
                    {"env_vars": {"WANDB_API_KEY": "k1"}},
                ]
            }
        )

        assert redacted["internal_states"][0]["env_vars"]["HF_TOKEN"].startswith("redacted-sha256:")
        assert redacted["internal_states"][0]["env_vars"]["RANK"] == "0"
        assert redacted["internal_states"][1]["env_vars"]["WANDB_API_KEY"].startswith("redacted-sha256:")

    def test_keeps_an_internal_state_that_carries_no_environment(self) -> None:
        """An engine built before the env var gate existed reports no env_vars key at all."""
        server_info = {"internal_states": [{"waiting_queue": 0}], "version": "0.5.0"}
        assert redact_server_info(server_info) == server_info

    def test_leaves_the_caller_copy_untouched(self) -> None:
        """The raw response is the engine's answer, and redaction must not rewrite it under the caller."""
        server_info = {"api_key": "a", "internal_states": [{"env_vars": {"HF_TOKEN": "t"}}]}

        redact_server_info(server_info)

        assert server_info["api_key"] == "a"
        assert server_info["internal_states"][0]["env_vars"]["HF_TOKEN"] == "t"

    def test_ignores_an_internal_states_shape_it_does_not_know(self) -> None:
        """The engine's schema is not miles's to control, and an unexpected shape must not raise."""
        assert redact_server_info({"internal_states": None}) == {"internal_states": None}
        assert redact_server_info({"internal_states": ["opaque"]}) == {"internal_states": ["opaque"]}
