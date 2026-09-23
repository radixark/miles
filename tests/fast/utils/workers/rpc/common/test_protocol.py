import pytest
from pydantic import ValidationError

from miles.utils.workers.rpc.common.protocol import CallStatusResponse, HealthResponse, SubmitResponse


@pytest.mark.parametrize("response_model", [SubmitResponse, CallStatusResponse, HealthResponse])
def test_response_models_reject_unknown_status_values(
    response_model: type[SubmitResponse] | type[CallStatusResponse] | type[HealthResponse],
) -> None:
    """Every response model rejects status values outside its protocol literal."""
    with pytest.raises(ValidationError):
        response_model.model_validate({"status": "unknown"})
