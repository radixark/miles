"""Adapter shards are keyed by (tp, pp, ep, etp): EP ranks hold different local experts,
ETP ranks hold different slices of them, and the realized coordinates are not the
tp x pp x ep cross product when ETP < TP."""

from miles.backends.megatron_utils.lora_checkpoint import all_megatron_checkpoints_exist, megatron_shard_name


def _names(coords, ep_size):
    return {megatron_shard_name(*coord, ep_size=ep_size) for coord in coords}


def test_shard_name_omits_ep_suffix_without_expert_parallelism():
    # Checkpoints written before expert adapters existed must stay loadable.
    assert megatron_shard_name(0, 0, 0, 0, ep_size=1) == "adapter_megatron_tp0_pp0.pt"
    assert megatron_shard_name(1, 2, 0, 0, ep_size=1) == "adapter_megatron_tp1_pp2.pt"


def test_shard_name_is_unique_per_expert_parallel_rank():
    names = {megatron_shard_name(0, 0, ep, 0, ep_size=4) for ep in range(4)}
    assert len(names) == 4
    assert megatron_shard_name(0, 0, 2, 0, ep_size=4) == "adapter_megatron_tp0_pp0_ep2.pt"


def test_shard_name_omits_etp_suffix_when_etp_does_not_exceed_tp():
    # ETP <= TP: the etp rank is a function of the tp rank, so names stay unchanged.
    assert megatron_shard_name(1, 0, 0, 0, ep_size=1, etp_size=1, tp_size=2) == "adapter_megatron_tp1_pp0.pt"


def test_shard_name_is_unique_per_expert_tensor_parallel_rank():
    # ETP > TP: two ranks share a tp rank but hold different expert slices.
    names = {megatron_shard_name(0, 0, 0, etp, ep_size=2, etp_size=2, tp_size=1) for etp in range(2)}
    assert names == {"adapter_megatron_tp0_pp0_ep0_etp0.pt", "adapter_megatron_tp0_pp0_ep0_etp1.pt"}


def test_completeness_check_requires_every_realized_shard(tmp_path):
    coords = [(0, 0, 0, 0), (0, 0, 1, 0), (0, 0, 2, 0)]
    for coord in coords[:2]:
        (tmp_path / megatron_shard_name(*coord, ep_size=3)).touch()

    assert not all_megatron_checkpoints_exist(tmp_path, _names(coords, 3))

    (tmp_path / megatron_shard_name(*coords[2], ep_size=3)).touch()
    assert all_megatron_checkpoints_exist(tmp_path, _names(coords, 3))


def test_completeness_ignores_unrealized_coordinates(tmp_path):
    # TP=2, EP=2, ETP=1: only (0,0,0) and (1,0,1) exist; a cross-product check
    # would demand four shards and never resume.
    coords = [(0, 0, 0, 0), (1, 0, 1, 0)]
    for coord in coords:
        (tmp_path / megatron_shard_name(*coord, ep_size=2)).touch()

    assert all_megatron_checkpoints_exist(tmp_path, _names(coords, 2))


def test_completeness_check_with_single_shard(tmp_path):
    (tmp_path / "adapter_megatron_tp0_pp0.pt").touch()
    assert all_megatron_checkpoints_exist(tmp_path, _names([(0, 0, 0, 0)], 1))
