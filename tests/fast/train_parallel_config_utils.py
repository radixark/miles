from miles.utils.dp_schedule import TrainParallelConfig


def make_train_parallel_config(
    *,
    dp_size: int = 1,
    cp_size: int = 1,
    vpp_size: int | None = 1,
    microbatch_group_size_per_vp_stage: int | None = None,
    independent_dp: bool = False,
    supports_precomputed_schedule: bool = False,
) -> TrainParallelConfig:
    return TrainParallelConfig(
        dp_size=dp_size,
        cp_size=cp_size,
        vpp_size=vpp_size,
        microbatch_group_size_per_vp_stage=microbatch_group_size_per_vp_stage,
        independent_dp=independent_dp,
        supports_precomputed_schedule=supports_precomputed_schedule,
    )
