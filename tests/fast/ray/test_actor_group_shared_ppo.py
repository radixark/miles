class _RemoteTrain:
    def __init__(self, rank, calls):
        self.rank = rank
        self.calls = calls

    def remote(self, rollout_id, rollout_data_ref, **kwargs):
        self.calls.append((self.rank, rollout_id, rollout_data_ref, kwargs))

        async def result():
            return {"rank": self.rank}

        return result()


class _Handle:
    def __init__(self, rank, calls):
        self.train = _RemoteTrain(rank, calls)


class _AsyncCall:
    def __init__(self, name, calls, result=None):
        self.name = name
        self.calls = calls
        self.result = result

    async def __call__(self, *args, **kwargs):
        self.calls.append((self.name, args, kwargs))
        return self.result


async def test_train_routes_each_critic_payload_to_matching_actor_rank():
    from miles.ray.actor_group import RayTrainGroup

    calls = []
    group = object.__new__(RayTrainGroup)
    group._actor_handles = [_Handle(0, calls), _Handle(1, calls)]
    payloads = [{"values": ["v0"]}, {"values": ["v1"]}]

    result = await group.train(5, {"data_ref": "rollout"}, external_data=payloads)

    assert result == [{"rank": 0}, {"rank": 1}]
    assert calls == [
        (0, 5, "rollout", {"witness_info": None, "attempt": 0, "external_data": payloads[0]}),
        (1, 5, "rollout", {"witness_info": None, "attempt": 0, "external_data": payloads[1]}),
    ]


async def test_train_broadcasts_without_lifecycle_options():
    from miles.ray.actor_group import RayTrainGroup

    calls = []
    group = object.__new__(RayTrainGroup)
    group._actor_handles = [_Handle(0, calls), _Handle(1, calls)]

    await group.train(7, {"data_ref": "rollout"})

    assert calls == [
        (0, 7, "rollout", {"witness_info": None, "attempt": 0}),
        (1, 7, "rollout", {"witness_info": None, "attempt": 0}),
    ]


async def test_train_rejects_wrong_number_of_rank_payloads():
    import pytest

    from miles.ray.actor_group import RayTrainGroup

    group = object.__new__(RayTrainGroup)
    group._actor_handles = [_Handle(0, []), _Handle(1, [])]

    with pytest.raises(ValueError, match="one payload per train worker"):
        await group.train(5, {"data_ref": "rollout"}, external_data=[{"values": []}])


async def test_train_only_ft_does_not_recover_rollout_engines():
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    from miles.ray.actor_group import RayTrainGroup

    calls = []
    info = SimpleNamespace(snapshot_cell_id_to_hashes={})
    group = object.__new__(RayTrainGroup)
    group.args = SimpleNamespace(
        debug_train_only=False,
        debug_rollout_only=False,
        use_fault_tolerance=True,
        ft_components=["train"],
    )
    group._inference_controller = SimpleNamespace(
        recover_updatable_engines=_AsyncCall("recover", calls),
        start_update_weights=_AsyncCall("start", calls, result=info),
        end_update_weights=_AsyncCall("end", calls),
    )
    group._broadcast = AsyncMock()

    await group.update_weights(rollout_id=1)

    assert [name for name, _, _ in calls] == ["start", "end"]
    group._broadcast.assert_awaited_once_with("update_weights", info=info)
