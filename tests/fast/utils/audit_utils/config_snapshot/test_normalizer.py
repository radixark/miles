import pytest

from miles.utils.audit_utils.config_snapshot.enabled import SNAPSHOT_TESTING_ENABLED

if not SNAPSHOT_TESTING_ENABLED:
    pytest.skip("Snapshot testing is disabled by op48-277", allow_module_level=True)

import os
from collections.abc import Callable

from tests.fast.fixtures.args_fixtures import parse_megatron_test_config

from miles.utils.args.trainer_utils import compute_trainer_config
from miles.utils.audit_utils.config_snapshot.converter import ConfigSnapshotConverter
from miles.utils.audit_utils.config_snapshot.generated_value_registry import (
    GENERATED_VALUES_ENV_VAR,
    generated_value_registry,
)
from miles.utils.audit_utils.config_snapshot.models import ConfigSnapshotGeneratedValue
from miles.utils.audit_utils.config_snapshot.normalizer import normalize_record
from miles.utils.audit_utils.process_identity import SimpleProcessIdentity
from miles.utils.env_report.redaction import redact_arg
from miles.utils.test_utils.snapshot import SNAPSHOT_RECORD_DIR_ENV_VAR, dump_snapshot


class TestGeneratedValueRegistration:
    def test_registration_without_a_snapshot_attempt_has_no_side_effect(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Ordinary launches do not carry snapshot normalization metadata."""
        monkeypatch.delenv(SNAPSHOT_RECORD_DIR_ENV_VAR, raising=False)
        monkeypatch.delenv(GENERATED_VALUES_ENV_VAR, raising=False)
        generated_value_registry.register(kind="run_id", value="generated")
        assert generated_value_registry.read() == []

    def test_repeated_values_are_idempotent_and_distinct_values_keep_distinct_tokens(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Generated identities stay distinct without token assignment depending on record order."""
        monkeypatch.setenv(SNAPSHOT_RECORD_DIR_ENV_VAR, "/snapshot/records")
        monkeypatch.delenv(GENERATED_VALUES_ENV_VAR, raising=False)
        generated_value_registry.register(kind="run_id", value="first")
        generated_value_registry.register(kind="run_id", value="first")
        generated_value_registry.register(kind="run_id", value="second")
        assert [(item.name, item.value) for item in generated_value_registry.read()] == [
            ("0000", "first"),
            ("0001", "second"),
        ]
        assert GENERATED_VALUES_ENV_VAR not in os.environ

    def test_environment_changes_do_not_replace_the_process_registry(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Registration survives environment changes and is exported only at launch boundaries."""
        monkeypatch.setenv(SNAPSHOT_RECORD_DIR_ENV_VAR, "/snapshot/records")
        generated_value_registry.register(kind="run_id", value="generated")
        monkeypatch.setenv(GENERATED_VALUES_ENV_VAR, "not a registry")
        assert [entry.value for entry in generated_value_registry.read()] == ["generated"]
        assert generated_value_registry.to_env()[GENERATED_VALUES_ENV_VAR] != "not a registry"
        copy = generated_value_registry.read()
        copy.clear()
        assert [entry.value for entry in generated_value_registry.read()] == ["generated"]


class TestGeneratedPathNormalization:
    @pytest.mark.parametrize("registered", [False, True])
    @pytest.mark.parametrize("run_id", ["first", "second"])
    def test_tinker_checkpoint_root_normalizes_only_registered_path_segments(
        self, make_record: Callable, registered: bool, run_id: str
    ) -> None:
        """Tinker run paths stabilize without hiding user roots, suffixes or custom fields."""
        path = f"/checkpoints/{run_id}/adapter"
        config = {
            "tinker_checkpoint_root": path,
            "backend": {"tinker_checkpoint_root": f"/checkpoints/{run_id}-other/adapter"},
            "custom": {"tinker_checkpoint_root": path},
        }
        record = make_record(config=config).model_copy(
            update={
                "generated_values": (
                    [ConfigSnapshotGeneratedValue(kind="run_id", name="0000", value=run_id)] if registered else []
                )
            }
        )

        expected = {**config, "tinker_checkpoint_root": "/checkpoints/$RUN_ID_0000/adapter" if registered else path}
        assert normalize_record(record)["args"] == expected
        assert record.config["args"] == config

    @pytest.mark.parametrize("registered", [False, True])
    @pytest.mark.parametrize("explicit_critic_save", [False, True])
    def test_derived_critic_identity_survives_trainer_backend_migration(
        self,
        make_record: Callable,
        monkeypatch: pytest.MonkeyPatch,
        registered: bool,
        explicit_critic_save: bool,
    ) -> None:
        """Only a critic path derived from a registered identity normalizes after moving into the backend."""
        monkeypatch.setenv(SNAPSHOT_RECORD_DIR_ENV_VAR, "/snapshot/records")
        if registered:
            generated_value_registry.register(kind="run_id", value="generated")
        path = "/checkpoints/generated_critic"
        flags = ["--critic-save", path] if explicit_critic_save else []
        args = parse_megatron_test_config(
            "--advantage-estimator", "ppo", "--save", "/checkpoints/generated/", "--save-interval", "1", *flags
        )
        trainer = compute_trainer_config(args, args.raw_megatron.trainers[1])
        record = make_record(config={"backend": {"save": trainer.backend.save}}).model_copy(
            update={"generated_values": generated_value_registry.read()}
        )

        expected = "/checkpoints/$RUN_ID_0000_critic" if registered and not explicit_critic_save else path
        assert normalize_record(record)["args"]["backend"]["save"] == expected
        assert record.config["args"]["backend"]["save"] == path
        generated_value_registry.register(kind="run_id", value="next-run")
        assert generated_value_registry.read()[-1].name == ("0001" if registered else "0000")

    @pytest.mark.parametrize("registered", [False, True])
    def test_trainer_overrides_normalize_only_registered_paths(self, make_record: Callable, registered: bool) -> None:
        """Trainer paths normalize registered identities without changing custom values or source records."""
        path = "/checkpoints/generated/critic"
        overrides = {"save": path, "load": path, "custom": {"save": path}}
        record = make_record(
            config={"raw_megatron": {"base_args": {}, "trainers": [{"overrides": overrides}]}}
        ).model_copy(
            update={
                "generated_values": (
                    [ConfigSnapshotGeneratedValue(kind="run_id", name="0000", value="generated")] if registered else []
                )
            }
        )

        actual = normalize_record(record)["args"]["raw_megatron"]["trainers"][0]["overrides"]
        expected_path = "/checkpoints/$RUN_ID_0000/critic" if registered else path
        assert actual == {"save": expected_path, "load": expected_path, "custom": {"save": path}}
        assert record.config["args"]["raw_megatron"]["trainers"][0]["overrides"] == {
            "save": path,
            "load": path,
            "custom": {"save": path},
        }

    @pytest.mark.parametrize("typed_backend", [False, True])
    def test_derived_critic_save_preserves_parent_and_role_across_generated_runs(
        self, make_record: Callable, typed_backend: bool
    ) -> None:
        """Flat and typed checkpoint payloads retain critic identity without generated run suffix drift."""
        for parent in ["/personal/checkpoints", "/other/checkpoints"]:
            results = []
            for run_id in ["260928-221656-095", "260929-221656-095"]:
                save = f"{parent}/{run_id}"
                config = {"critic_save": save + "_critic"}
                config.update({"backend": {"save": save + "/"}} if typed_backend else {"save": save + "/"})
                record = make_record(config=config).model_copy(
                    update={
                        "generated_values": [ConfigSnapshotGeneratedValue(kind="run_id", name="0000", value=run_id)]
                    }
                )
                normalized = normalize_record(record)["args"]
                assert normalized["critic_save"] == f"{parent}/$RUN_ID_0000_critic"
                assert record.config["args"]["critic_save"] == save + "_critic"
                results.append(normalized)
            assert results[0] == results[1]

    @pytest.mark.parametrize(
        "save,critic_save,registered",
        [
            ("/checkpoints/generated", "/checkpoints/generated_critic", False),
            (None, "/checkpoints/generated_critic", True),
            ("/other/generated", "/checkpoints/generated_critic", True),
            ("/checkpoints/generated-other", "/checkpoints/generated-other_critic", True),
            ("/checkpoints/generated", "/checkpoints/generated_critic_extra", True),
        ],
    )
    def test_critic_suffix_requires_registered_identity_and_exact_save_derivation(
        self, make_record: Callable, save: str | None, critic_save: str, registered: bool
    ) -> None:
        """Unregistered or independently configured critic paths do not borrow a generated identity."""
        record = make_record(config={"save": save, "critic_save": critic_save}).model_copy(
            update={
                "generated_values": (
                    [ConfigSnapshotGeneratedValue(kind="run_id", name="0000", value="generated")] if registered else []
                )
            }
        )
        assert normalize_record(record)["args"]["critic_save"] == critic_save

    def test_critic_suffix_does_not_generalize_to_other_fields_or_nested_objects(self, make_record: Callable) -> None:
        """The derived critic rule leaves unrelated suffix paths and custom configuration untouched."""
        record = make_record(
            config={
                "save": "/checkpoints/generated_critic",
                "load": "/checkpoints/generated_critic",
                "custom": {"save": "/checkpoints/generated", "critic_save": "/checkpoints/generated_critic"},
            }
        ).model_copy(
            update={"generated_values": [ConfigSnapshotGeneratedValue(kind="run_id", name="0000", value="generated")]}
        )
        assert normalize_record(record) == record.config

    @pytest.mark.parametrize(
        "field",
        ["eval_hf_dir", "dumper_dir", "dumper_source_patcher_config_train", "dumper_source_patcher_config_inference"],
    )
    def test_registered_directories_preserve_modes_and_path_semantics(self, make_record: Callable, field: str) -> None:
        """Registered roots stabilize while mode, parent and descendant paths remain observable."""
        results = {}
        for mode, parent, suffix, child in [
            ("fleet", "/dev/shm", "first", ""),
            ("fleet", "/dev/shm", "second", ""),
            ("external", "/dev/shm", "first", ""),
            ("fleet", "/tmp", "first", ""),
            ("fleet", "/dev/shm", "first", "/step_1"),
            ("fleet", "/dev/shm", "first", "/step_2"),
        ]:
            root = f"{parent}/miles_eval_{mode}_{suffix}"
            record = make_record(config={field: root + child}).model_copy(
                update={
                    "generated_values": [
                        ConfigSnapshotGeneratedValue(
                            kind="temporary_directory", name=f"fully_async_eval_{mode}", value=root
                        )
                    ]
                }
            )
            actual = normalize_record(record)["args"][field]
            assert actual == f"{parent}/$TEMPORARY_DIRECTORY_fully_async_eval_{mode}{child}"
            results[(mode, parent, suffix, child)] = actual
        assert results[("fleet", "/dev/shm", "first", "")] == results[("fleet", "/dev/shm", "second", "")]
        assert len(set(results.values())) == 5

    @pytest.mark.parametrize(
        "field",
        ["eval_hf_dir", "dumper_dir", "dumper_source_patcher_config_train", "dumper_source_patcher_config_inference"],
    )
    @pytest.mark.parametrize("registered", [False, True])
    def test_registered_path_fields_without_matching_provenance_remain_literal(
        self, make_record: Callable, registered: bool, field: str
    ) -> None:
        """Similar prefixes and unrelated fields never borrow directory provenance."""
        root = "/dev/shm/miles_eval_fleet_random"
        record = make_record(
            config={
                field: root + ("-sibling" if registered else ""),
                "custom_path": root,
                "custom_config": {field: root},
            }
        ).model_copy(
            update={
                "generated_values": (
                    [
                        ConfigSnapshotGeneratedValue(
                            kind="temporary_directory", name="fully_async_eval_fleet", value=root
                        )
                    ]
                    if registered
                    else []
                )
            }
        )
        assert normalize_record(record) == record.config

    def test_raw_megatron_precision_paths_use_registered_temporary_directories(self, make_record: Callable) -> None:
        """Raw Megatron precision paths stabilize across generated directories without dropping config fields."""
        snapshots = []
        for directory in ["/tmp/miles-dsv4-bshd-thd-u1ae27up", "/tmp/miles-dsv4-bshd-thd-other"]:
            record = make_record(
                config={
                    "raw_megatron": {
                        "trainers": [
                            {
                                "trainer_id": "actor",
                                "model_id": None,
                                "role": "actor",
                                "actor_index": 0,
                                "overrides": {},
                            }
                        ],
                        "base_args": {
                            "te_precision_config_file": f"{directory}/te_precision.yaml",
                            "tensor_model_parallel_size": 4,
                        },
                    }
                }
            )
            record = record.model_copy(
                update={
                    "context": record.context.model_copy(
                        update={"source": SimpleProcessIdentity(component="rollout_executor"), "capture_id": "rollout"}
                    ),
                    "generated_values": [
                        ConfigSnapshotGeneratedValue(kind="temporary_directory", name="dsv4_parity", value=directory)
                    ],
                }
            )
            normalized = normalize_record(record)
            assert normalized["args"]["raw_megatron"] == {
                "trainers": record.config["args"]["raw_megatron"]["trainers"],
                "base_args": {
                    "te_precision_config_file": "/tmp/$TEMPORARY_DIRECTORY_dsv4_parity/te_precision.yaml",
                    "tensor_model_parallel_size": 4,
                },
            }
            assert record.config["args"]["raw_megatron"]["base_args"]["te_precision_config_file"] == (
                f"{directory}/te_precision.yaml"
            )
            raw_snapshot = dump_snapshot(ConfigSnapshotConverter.convert([record]))
            assert directory not in raw_snapshot
            trainer = make_record(
                config={
                    "backend": {
                        "backend_name": "megatron",
                        "rank": 0,
                        "te_precision_config_file": f"{directory}/te_precision.yaml",
                    }
                }
            )
            snapshot = dump_snapshot(ConfigSnapshotConverter.convert([record, trainer]))
            assert directory not in snapshot
            assert snapshot.count("$TEMPORARY_DIRECTORY_dsv4_parity/te_precision.yaml") == 2
            snapshots.append(snapshot)
        assert snapshots[0] == snapshots[1]
        assert "$TEMPORARY_DIRECTORY_dsv4_parity/te_precision.yaml" in snapshots[0]

    @pytest.mark.parametrize("registered", [False, True])
    def test_raw_megatron_unregistered_and_unrelated_paths_remain_literal(
        self, make_record: Callable, registered: bool
    ) -> None:
        """Only registered directory boundaries in known raw Megatron path fields may change."""
        directory = "/tmp/miles-dsv4-bshd-thd-u1ae27up"
        path = f"{directory}/te_precision.yaml"
        record = make_record(
            config={
                "raw_megatron": {
                    "base_args": {
                        "te_precision_config_file": path if not registered else f"{directory}-other/te_precision.yaml",
                        "custom_path": path,
                        "custom_config": {"te_precision_config_file": path},
                    },
                    "te_precision_config_file": path,
                },
                "base_args": {"te_precision_config_file": path},
                "custom_config": {"raw_megatron": {"base_args": {"te_precision_config_file": path}}},
            }
        ).model_copy(
            update={
                "generated_values": (
                    [ConfigSnapshotGeneratedValue(kind="temporary_directory", name="dsv4_parity", value=directory)]
                    if registered
                    else []
                )
            }
        )
        assert normalize_record(record) == record.config

    def test_raw_megatron_precision_filename_changes_remain_visible(self, make_record: Callable) -> None:
        """Registered directory normalization preserves precision file selection differences."""
        directory = "/tmp/miles-dsv4-bshd-thd-u1ae27up"
        records = [
            make_record(
                config={"raw_megatron": {"base_args": {"te_precision_config_file": f"{directory}/{name}"}}}
            ).model_copy(
                update={
                    "generated_values": [
                        ConfigSnapshotGeneratedValue(kind="temporary_directory", name="dsv4_parity", value=directory)
                    ]
                }
            )
            for name in ["te_precision.yaml", "other_precision.yaml"]
        ]
        assert normalize_record(records[0]) != normalize_record(records[1])

    def test_tracking_ids_normalize_across_processes_without_provenance(self, make_record: Callable) -> None:
        """Tracking IDs normalize across processes without generated-value registration."""
        results = []
        for value in ["automatic-one", "automatic-two"]:
            source = make_record(config={"wandb_run_id": value})
            source = source.model_copy(
                update={
                    "context": source.context.model_copy(
                        update={"source": SimpleProcessIdentity(component="main"), "capture_id": "primary"}
                    ),
                }
            )
            consumer = make_record(config={"wandb_run_id": value})
            snapshot = dump_snapshot(ConfigSnapshotConverter.convert([consumer, source]))
            assert value not in snapshot
            assert snapshot == dump_snapshot(ConfigSnapshotConverter.convert([source, consumer]))
            results.append(snapshot)
        assert results[0] == results[1]

    @pytest.mark.parametrize("run_id", [None, "explicit", "automatic"])
    def test_tracking_ids_normalize_only_when_present(self, make_record: Callable, run_id: str | None) -> None:
        """Tracking identity is hidden while disabled tracking and unrelated values stay observable."""
        record = make_record(config={"wandb_run_id": run_id, "wandb_group": "explicit"})
        assert normalize_record(record) == {
            "args": {"wandb_run_id": None if run_id is None else "$WANDB_RUN_ID_0000", "wandb_group": "explicit"}
        }

    def test_only_registered_values_in_explicit_fields_are_normalized(self, make_record: Callable) -> None:
        """Random paths stabilize while static roots, suffixes and unrelated strings remain visible."""
        outputs = []
        for run_id, temp, commit in [
            ("260928-123456-001", "abc", "sha-a_3739"),
            ("260929-112233-002", "xyz", "sha-b_3739"),
        ]:
            record = make_record(
                config={
                    "save": f"/data/{run_id}/checkpoints",
                    "backend": {"load": f"/data/{run_id}/checkpoints"},
                    "dump_details": f"/tmp/{temp}/bshd/dump_details",
                    "wandb_group": f"prefix_{run_id}_{commit}",
                    "custom_path": "/data/260928-123456-001/checkpoints",
                    "hf_checkpoint": "/data/260928-123456-001/model",
                }
            ).model_copy(
                update={
                    "generated_values": [
                        ConfigSnapshotGeneratedValue(kind="run_id", name="0000", value=run_id),
                        ConfigSnapshotGeneratedValue(kind="temporary_directory", name="parity", value=f"/tmp/{temp}"),
                        ConfigSnapshotGeneratedValue(kind="ci_commit_name", name="github", value=commit),
                    ]
                }
            )
            outputs.append(normalize_record(record))
        assert outputs[0] == outputs[1]
        assert outputs[0]["args"] == {
            "save": "/data/$RUN_ID_0000/checkpoints",
            "backend": {"load": "/data/$RUN_ID_0000/checkpoints"},
            "dump_details": "/tmp/$TEMPORARY_DIRECTORY_parity/bshd/dump_details",
            "wandb_group": "prefix_$RUN_ID_0000_$CI_COMMIT_NAME_github",
            "custom_path": "/data/260928-123456-001/checkpoints",
            "hf_checkpoint": "/data/260928-123456-001/model",
        }

    @pytest.mark.parametrize(
        "path", ["/other/id/checkpoints", "/data/id/other", "/data/id-extra/checkpoints", "/data/other/checkpoints"]
    )
    def test_path_root_suffix_and_unregistered_values_remain_distinguishable(
        self, make_record: Callable, path: str
    ) -> None:
        """Normalization cannot hide a different output root, file, or unregistered identity."""
        metadata = [ConfigSnapshotGeneratedValue(kind="run_id", name="0000", value="id")]
        expected = make_record(config={"save": "/data/id/checkpoints"}).model_copy(
            update={"generated_values": metadata}
        )
        actual = make_record(config={"save": path}).model_copy(update={"generated_values": metadata})
        assert normalize_record(expected) != normalize_record(actual)

    def test_swapping_two_generated_references_remains_observable(self, make_record: Callable) -> None:
        """Two generated run IDs never collapse to one anonymous path token."""
        metadata = [
            ConfigSnapshotGeneratedValue(kind="run_id", name=str(i), value=value)
            for i, value in enumerate(["first", "second"])
        ]
        records = [
            make_record(config={"save": f"/data/{value}/checkpoints"}).model_copy(
                update={"generated_values": metadata}
            )
            for value in ["first", "second"]
        ]
        assert normalize_record(records[0]) != normalize_record(records[1])

    def test_rank_local_generated_identity_conflicts_cannot_be_hidden(self, make_record: Callable) -> None:
        """Equivalent ranks cannot silently normalize different values under the same identity."""
        records = [
            make_record(rank=rank, config={"save": f"/data/{value}/checkpoints"}).model_copy(
                update={"generated_values": [ConfigSnapshotGeneratedValue(kind="run_id", name="0000", value=value)]}
            )
            for rank, value in enumerate(["first", "second"])
        ]
        with pytest.raises(ValueError, match="Conflicting generated"):
            ConfigSnapshotConverter.convert(records)

    def test_registrations_do_not_depend_on_raw_record_order(self, make_record: Callable) -> None:
        """Generated metadata produces identical snapshots when storage order changes."""
        records = [
            make_record(run=run, config={"save": f"/data/{value}/checkpoints"}).model_copy(
                update={"generated_values": [ConfigSnapshotGeneratedValue(kind="run_id", name="0000", value=value)]}
            )
            for run, value in enumerate(["first", "second"])
        ]
        assert dump_snapshot(ConfigSnapshotConverter.convert(records)) == dump_snapshot(
            ConfigSnapshotConverter.convert(list(reversed(records)))
        )


class TestWandbCredentialProvenance:
    @pytest.mark.parametrize("registered", [False, True])
    @pytest.mark.parametrize("key", [None, "environment-key", "explicit-key"])
    def test_only_matching_registered_credentials_normalize(
        self, make_record: Callable, registered: bool, key: str | None
    ) -> None:
        """Credential provenance cannot hide disabled tracking or an explicit different key."""
        digest = redact_arg("wandb_key", key)
        record = make_record(config={"wandb_key": digest, "unrelated": digest}).model_copy(
            update={
                "generated_values": (
                    [
                        ConfigSnapshotGeneratedValue(
                            kind="wandb_key", name="environment", value=redact_arg("wandb_key", "environment-key")
                        )
                    ]
                    if registered
                    else []
                )
            }
        )

        assert normalize_record(record) == {
            "args": {
                "wandb_key": "$WANDB_KEY_environment" if registered and key == "environment-key" else digest,
                "unrelated": digest,
            }
        }

    def test_environment_rotation_preserves_the_snapshot(self, make_record: Callable) -> None:
        """Different environment credentials have one stable token when each has provenance."""
        snapshots = []
        for key in ("first-environment-key", "second-environment-key"):
            digest = redact_arg("wandb_key", key)
            record = make_record(config={"wandb_key": digest}).model_copy(
                update={
                    "generated_values": [
                        ConfigSnapshotGeneratedValue(kind="wandb_key", name="environment", value=digest)
                    ]
                }
            )
            snapshots.append(normalize_record(record))
        assert snapshots == [{"args": {"wandb_key": "$WANDB_KEY_environment"}}] * 2


class TestOtherBackendGeneratedPaths:
    @pytest.mark.parametrize("backend", ["raw_fsdp", "raw_torchtitan"])
    def test_registered_load_path_is_normalized_only_in_the_backend_container(
        self, make_record: Callable, backend: str
    ) -> None:
        """FSDP and TorchTitan load paths normalize registered directories while preserving unrelated values."""
        directory = "/tmp/generated-checkpoint"
        fields = {"load": directory + "/weights", "model_name": directory, "save": "/user/checkpoints"}
        record = make_record(config={backend: fields, "custom": {backend: fields}}).model_copy(
            update={
                "generated_values": [
                    ConfigSnapshotGeneratedValue(kind="temporary_directory", name="checkpoint", value=directory)
                ]
            }
        )

        normalized = normalize_record(record)["args"]

        assert normalized[backend] == {
            "load": "/tmp/$TEMPORARY_DIRECTORY_checkpoint/weights",
            "model_name": directory,
            "save": "/user/checkpoints",
        }
        assert normalized["custom"][backend] == fields
        assert record.config["args"][backend] == fields


class TestStandaloneLocalRankNormalization:
    @pytest.mark.parametrize("nested", [False, True])
    def test_single_node_ranks_merge_without_changing_raw_or_custom_fields(
        self, make_record: Callable, nested: bool
    ) -> None:
        """Standalone ranks merge only after validating their local device identity."""
        records = []
        for rank in [0, 7]:
            fields = {"rank": rank, "local_rank": rank, "custom": {"local_rank": 9}}
            record = make_record(rank=rank, config={"backend": fields} if nested else fields)
            record = record.model_copy(
                update={"context": record.context.model_copy(update={"deploy_component": "standalone"})}
            )
            records.append(record)

        case = ConfigSnapshotConverter.convert(records)

        process = next(iter(case.processes.values()))
        assert process.ranks == [0, 7]
        expected = {"rank": "$RANK", "local_rank": "$RANK", "custom": {"local_rank": 9}}
        assert process.base == {"args": {"backend": expected} if nested else expected}
        raw = [record.config["args"] for record in records]
        assert [(fields["backend"] if nested else fields)["local_rank"] for fields in raw] == [0, 7]

    @pytest.mark.parametrize("nested", [False, True])
    @pytest.mark.parametrize("local_rank", [None, -1, True, 1.5, "1", 0, 2])
    def test_invalid_or_mismatched_local_rank_is_rejected(
        self, make_record: Callable, local_rank: object, nested: bool
    ) -> None:
        """Malformed or incorrect device placement cannot disappear during rank deduplication."""
        fields = {"rank": 1, "local_rank": local_rank}
        record = make_record(rank=1, config={"backend": fields} if nested else fields)
        record = record.model_copy(
            update={"context": record.context.model_copy(update={"deploy_component": "standalone"})}
        )

        with pytest.raises(AssertionError, match="Unexpected standalone local rank"):
            normalize_record(record)

    @pytest.mark.parametrize("nested", [False, True])
    def test_other_deployments_keep_local_rank_literal(self, make_record: Callable, nested: bool) -> None:
        """The single-node standalone rule does not change other deployment layouts."""
        fields = {"local_rank": 1}
        record = make_record(rank=7, config={"backend": fields} if nested else fields)
        assert normalize_record(record) == record.config


class TestObservedRunIds:
    def test_unused_registration_does_not_shift_the_used_run_id(self, make_record: Callable) -> None:
        """Unused CLI defaults cannot change the snapshot of the selected run ID."""
        record = make_record(config={"tinker_checkpoint_root": "/checkpoints/selected"})
        original = record.model_copy(
            update={
                "generated_values": [
                    ConfigSnapshotGeneratedValue(kind="run_id", name="0000", value="selected"),
                ]
            }
        )
        shifted = record.model_copy(
            update={
                "generated_values": [
                    ConfigSnapshotGeneratedValue(kind="run_id", name="0000", value="unused"),
                    ConfigSnapshotGeneratedValue(kind="run_id", name="0001", value="selected"),
                ]
            }
        )
        assert dump_snapshot(ConfigSnapshotConverter.convert([original])) == dump_snapshot(
            ConfigSnapshotConverter.convert([shifted])
        )

    def test_swapping_used_run_ids_between_paths_remains_visible(self, make_record: Callable) -> None:
        """Canonical numbering preserves distinct references when both generated IDs are used."""
        metadata = [
            ConfigSnapshotGeneratedValue(kind="run_id", name=f"{i:04d}", value=value)
            for i, value in enumerate(["unused", "first", "second"])
        ]
        records = [
            make_record(config={"save": f"/checkpoints/{a}", "load": f"/checkpoints/{b}"}).model_copy(
                update={"generated_values": metadata}
            )
            for a, b in [("first", "second"), ("second", "first")]
        ]
        assert dump_snapshot(ConfigSnapshotConverter.convert([records[0]])) != dump_snapshot(
            ConfigSnapshotConverter.convert([records[1]])
        )
