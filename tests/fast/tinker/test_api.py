from fastapi.testclient import TestClient

from miles.tinker.api import create_app


class RecordingBackend:
    def __init__(self):
        self.calls = []

    async def create_model(self, model_id, lora_config, model_role):
        self.calls.append(("create", model_id, lora_config, model_role))
        return {"model_id": model_id, "base_model": "Qwen/Qwen3-0.6B", "lora_config": lora_config}

    async def forward_backward(self, model_id, batch):
        self.calls.append(("forward_backward", model_id, batch))
        return {"loss_fn_output_type": "importance_sampling", "loss_fn_outputs": [], "metrics": {"loss": 1.0}}

    async def optim_step(self, model_id, adam_params):
        self.calls.append(("optim_step", model_id, adam_params))
        return {"metrics": {"learning_rate": adam_params["learning_rate"]}}

    async def save_sampler(self, model_id, checkpoint_id):
        self.calls.append(("save_sampler", model_id, checkpoint_id))
        return {"path": f"tinker://{model_id}/sampler_weights/{checkpoint_id}"}

    async def sample(self, model_id, request):
        self.calls.append(("sample", model_id, request))
        return {"sequences": [{"stop_reason": "length", "tokens": [1, 2], "logprobs": [-0.1, -0.2]}]}

    async def delete_model(self, model_id):
        self.calls.append(("delete", model_id))


def test_tinker_forward_backward_and_step_are_distinct_calls():
    backend = RecordingBackend()
    client = TestClient(create_app(backend, "Qwen/Qwen3-0.6B"))
    session = client.post("/api/v1/create_session", json={}).json()["session_id"]
    created = client.post(
        "/api/v1/create_model",
        json={"session_id": session, "base_model": "Qwen/Qwen3-0.6B", "lora_config": {"rank": 16}},
    ).json()
    model_id = created["model_id"]

    fb = client.post(
        "/api/v1/forward_backward",
        json={"model_id": model_id, "forward_backward_input": {"data": [], "loss_fn": "importance_sampling"}},
    ).json()
    assert client.post("/api/v1/retrieve_future", json={"request_id": fb["future_id"]}).json()["metrics"] == {
        "loss": 1.0
    }
    assert [call[0] for call in backend.calls] == ["create", "forward_backward"]

    step = client.post(
        "/api/v1/optim_step",
        json={"model_id": model_id, "adam_params": {"learning_rate": 4e-5}},
    ).json()
    assert client.post("/api/v1/retrieve_future", json={"request_id": step["future_id"]}).json()["metrics"] == {
        "learning_rate": 4e-5
    }
    assert [call[0] for call in backend.calls] == ["create", "forward_backward", "optim_step"]


def test_rejects_rank_above_preallocated_capacity():
    client = TestClient(create_app(RecordingBackend(), "Qwen/Qwen3-0.6B", max_lora_rank=16))
    session = client.post("/api/v1/create_session", json={}).json()["session_id"]
    response = client.post(
        "/api/v1/create_model",
        json={"session_id": session, "base_model": "Qwen/Qwen3-0.6B", "lora_config": {"rank": 32}},
    )
    assert response.status_code == 400


def test_sampler_snapshot_routes_to_the_same_adapter():
    backend = RecordingBackend()
    client = TestClient(create_app(backend, "Qwen/Qwen3-0.6B"))
    session = client.post("/api/v1/create_session", json={}).json()["session_id"]
    model_id = client.post(
        "/api/v1/create_model",
        json={"session_id": session, "base_model": "Qwen/Qwen3-0.6B", "lora_config": {"rank": 16}},
    ).json()["model_id"]
    saved = client.post(
        "/api/v1/save_weights_for_sampler",
        json={"model_id": model_id, "sampling_session_seq_id": 0, "seq_id": 1},
    ).json()
    snapshot = client.post("/api/v1/retrieve_future", json={"request_id": saved["future_id"]}).json()
    sampled = client.post(
        "/api/v1/asample",
        json={
            "sampling_session_id": snapshot["sampling_session_id"],
            "seq_id": 2,
            "prompt": {"chunks": [{"type": "encoded_text", "tokens": [1]}]},
            "sampling_params": {"max_tokens": 2},
        },
    ).json()
    output = client.post("/api/v1/retrieve_future", json={"request_id": sampled["future_id"]}).json()
    assert output["sequences"][0]["tokens"] == [1, 2]
    assert backend.calls[-1][0:2] == ("sample", model_id)
