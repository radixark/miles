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


async def test_train_rejects_wrong_number_of_rank_payloads():
    import pytest

    from miles.ray.actor_group import RayTrainGroup

    group = object.__new__(RayTrainGroup)
    group._actor_handles = [_Handle(0, []), _Handle(1, [])]

    with pytest.raises(ValueError, match="one payload per train worker"):
        await group.train(5, {"data_ref": "rollout"}, external_data=[{"values": []}])
