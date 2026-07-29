async def _submit(raw, method: str, call_id: str, query: dict, **extra):
    return await raw.post(f"/v1/{method}", json={"call_id": call_id, "query": query, **extra})


class TestMethodLookup:
    async def test_unknown_method_404(self, raw, tag):
        """Submitting to a method the worker does not define is a 404."""
        response = await _submit(raw, "no_such_method", tag, {})
        assert response.status_code == 404
        assert "no_such_method" in response.json()["detail"]

    async def test_dunder_not_exposed(self, raw, tag):
        """Dunder attributes are not reachable."""
        assert (await _submit(raw, "__init__", tag, {})).status_code == 404

    async def test_health_is_not_a_submit_target(self, raw, tag):
        """POST /v1/health falls through to method lookup and 404s."""
        assert (await _submit(raw, "health", tag, {})).status_code == 404

    async def test_calls_path_is_not_a_submit_target(self, raw):
        """POST /v1/calls/... is method lookup, not a call query."""
        assert (await raw.post("/v1/calls", json={"call_id": "x", "query": {}})).status_code == 404


class TestEnvelopeValidation:
    async def test_malformed_json_400(self, raw):
        """A body that is not JSON is a client error, normalized to 400."""
        response = await raw.post("/v1/demo_async", content=b"{not json", headers={"content-type": "application/json"})
        assert response.status_code == 400

    async def test_missing_call_id_400(self, raw):
        """An envelope without call_id is rejected."""
        assert (await raw.post("/v1/demo_async", json={"query": {"value": {}}})).status_code == 400

    async def test_missing_query_400(self, raw, tag):
        """An envelope without query is rejected."""
        assert (await raw.post("/v1/demo_async", json={"call_id": tag})).status_code == 400

    async def test_extra_envelope_field_400(self, raw, tag):
        """Unknown envelope fields are rejected rather than ignored."""
        response = await raw.post("/v1/demo_async", json={"call_id": tag, "query": {"value": {}}, "bogus": 1})
        assert response.status_code == 400

    async def test_query_not_an_object_400(self, raw, tag):
        """A non-object query is rejected."""
        assert (await raw.post("/v1/demo_async", json={"call_id": tag, "query": [1, 2]})).status_code == 400


class TestQueryValidation:
    async def test_unknown_kwarg_400(self, raw, tag):
        """An argument the method does not declare is rejected."""
        response = await _submit(raw, "demo_async", tag, {"value": {}, "extra": 3})
        assert response.status_code == 400 and "extra" in response.json()["detail"]

    async def test_missing_required_kwarg_400(self, raw, tag):
        """A missing required argument is rejected."""
        response = await _submit(raw, "demo_async", tag, {})
        assert response.status_code == 400 and "value" in response.json()["detail"]


class TestCallLookup:
    async def test_unknown_call_id_404(self, raw):
        """Polling an unknown call id is a 404."""
        response = await raw.get("/v1/calls/deadbeef", params={"timeout": 0.0})
        assert response.status_code == 404 and "deadbeef" in response.json()["detail"]
