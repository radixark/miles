import httpx


class TestManualProtocol:
    async def test_submit_then_poll_by_hand(self, raw):
        """The documented submit + poll pair works without the client."""
        submit = await raw.post("/v1/demo_async", json={"call_id": "manual-1", "query": {"value": {"a": 2}}})
        assert submit.status_code == 200
        assert submit.json() == {"status": "submitted"}

        for _ in range(50):
            poll = await raw.get("/v1/calls/manual-1", params={"timeout": 1.0})
            assert poll.status_code == 200
            if poll.json()["status"] != "pending":
                break
        assert poll.json() == {"status": "success", "result": {"a": 2}, "error": None}

    async def test_finished_call_can_be_polled_repeatedly(self, raw):
        """A finished outcome stays retrievable for later polls."""
        await raw.post("/v1/demo_async", json={"call_id": "manual-2", "query": {"value": {"b": 1}}})
        for _ in range(50):
            body = (await raw.get("/v1/calls/manual-2", params={"timeout": 5.0})).json()
            if body["status"] != "pending":
                break
        assert body == {"status": "success", "result": {"b": 1}, "error": None}

    async def test_second_client_sees_the_outcome(self, raw, server):
        """Call state belongs to the server, not to the connection that submitted it."""
        await raw.post("/v1/demo_async", json={"call_id": "manual-3", "query": {"value": {"c": 8}}})
        async with httpx.AsyncClient(base_url=server.url, timeout=30.0, trust_env=False) as other:
            for _ in range(50):
                body = (await other.get("/v1/calls/manual-3", params={"timeout": 5.0})).json()
                if body["status"] != "pending":
                    break
        assert body["result"] == {"c": 8}
