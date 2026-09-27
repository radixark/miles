import numpy as np
import pytest
import safetensors.numpy

from miles.utils.disk_delta import checkpoint_tensor_layout, make_tensor_reader, validate_nvme_delta_paths


def test_tensor_reader_validates_declared_layout(tmp_path):
    expected = np.arange(6, dtype=np.float16).reshape(2, 3)
    safetensors.numpy.save_file({"weight": expected}, tmp_path / "model.safetensors")
    read = make_tensor_reader(str(tmp_path))

    actual = read("weight", expected_dtype="F16", expected_shape=(2, 3))
    np.testing.assert_array_equal(actual, expected.view(np.uint8).reshape(-1))

    with pytest.raises(ValueError, match="dtype=F16, shape=\\(2, 3\\)"):
        read("weight", expected_dtype="BF16", expected_shape=(2, 3))
    with pytest.raises(ValueError, match="dtype=F16, shape=\\(2, 3\\)"):
        read("weight", expected_dtype="F16", expected_shape=(3, 2))

    assert checkpoint_tensor_layout(str(tmp_path), "weight") == ("F16", (2, 3))


@pytest.mark.parametrize("field", ["publication_dir", "receiver_dir"])
@pytest.mark.parametrize("relation", ["same", "parent", "child"])
def test_nvme_baseline_directories_cannot_overlap_sync_trees(tmp_path, field, relation):
    baseline = tmp_path / "baseline"
    other = {"same": baseline, "parent": tmp_path, "child": baseline / "nested"}[relation]
    directories = {"publication_dir": None, "receiver_dir": None, field: str(other)}
    with pytest.raises(ValueError, match="must not overlap"):
        validate_nvme_delta_paths(str(baseline), **directories)
    assert not baseline.exists()


def test_nvme_path_separation_resolves_symlinks_and_relative_components(tmp_path, monkeypatch):
    publication = tmp_path / "published"
    publication.mkdir()
    (tmp_path / "alias").symlink_to(publication, target_is_directory=True)
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValueError, match="must not overlap --update-weight-disk-dir"):
        validate_nvme_delta_paths("alias/nested/../baseline", publication_dir="published", receiver_dir=None)


def test_nvme_path_separation_accepts_siblings_with_shared_prefixes(tmp_path):
    validate_nvme_delta_paths(
        str(tmp_path / "baseline"),
        publication_dir=str(tmp_path / "baseline-published"),
        receiver_dir=str(tmp_path / "baseline-receiver"),
    )
