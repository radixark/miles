from unittest.mock import MagicMock

from ray.actor import ActorHandle

from miles.ray.rollout.server_engine import ServerEngine


def test_generation_id_tracks_one_actor_allocation() -> None:
    engine = ServerEngine()

    engine.mark_allocated_uninitialized(MagicMock(spec=ActorHandle))
    first_generation_id = engine.generation_id
    engine.mark_alive()

    assert engine.generation_id == first_generation_id

    engine.mark_stopped()
    engine.mark_allocated_uninitialized(MagicMock(spec=ActorHandle))

    assert engine.generation_id != first_generation_id
