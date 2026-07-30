import pytest
from pydantic import ValidationError

from miles.ray.tinker.protocol import (
    ModelInput,
    ModelInputChunk,
    SamplingParams,
    TensorData,
    TinkerError,
    dense_tensor_data,
    encoded_tokens,
)


def test_dense_tensor_data_decodes_csr():
    value = TensorData(
        data=[3.0, 4.0],
        dtype="float32",
        shape=[2, 3],
        sparse_crow_indices=[0, 1, 2],
        sparse_col_indices=[2, 0],
    )

    data, shape = dense_tensor_data(value)

    assert data == [0.0, 0.0, 3.0, 4.0, 0.0, 0.0]
    assert shape == [2, 3]


@pytest.mark.parametrize(
    "value",
    [
        TensorData(
            data=[3.0],
            dtype="float32",
            shape=[2, 3],
            sparse_crow_indices=[0, 1],
            sparse_col_indices=[2],
        ),
        TensorData(
            data=[3.0],
            dtype="float32",
            shape=[2, 3],
            sparse_crow_indices=[0, 1, 0],
            sparse_col_indices=[2],
        ),
        TensorData(
            data=[3.0],
            dtype="float32",
            shape=[2, 3],
            sparse_crow_indices=[0, 1, 1],
            sparse_col_indices=[3],
        ),
    ],
)
def test_dense_tensor_data_rejects_malformed_csr(value):
    with pytest.raises(TinkerError):
        dense_tensor_data(value)


def test_encoded_tokens_concatenates_chunks():
    value = ModelInput(
        chunks=[
            ModelInputChunk(type="encoded_text", tokens=[1, 2]),
            ModelInputChunk(type="encoded_text", tokens=[3]),
        ]
    )

    assert encoded_tokens(value) == [1, 2, 3]


def test_non_text_chunk_returns_typed_user_error():
    value = ModelInput(chunks=[ModelInputChunk(type="image", tokens=None, data="ignored")])

    with pytest.raises(TinkerError) as exc_info:
        encoded_tokens(value)

    assert exc_info.value.category == "user"


def test_sampling_params_reject_unknown_fields():
    with pytest.raises(ValidationError, match="extra_forbidden"):
        SamplingParams(temperature=0.5, unsupported_penalty=1.0)
