from argparse import Namespace

from tests.fast.backends.sglang_utils.conftest import make_engine_args

from miles.backends.sglang_utils.sglang_engine import _compute_server_args


class TestResolvedServerArgs:
    def test_parallel_and_generic_sdk_values_are_read_from_resolved_config_without_flat_fields(self) -> None:
        """Engine SDK kwargs retain resolved parallel dimensions and passthrough fields after CLI ownership moves."""
        original = make_engine_args()
        original.sglang = original.sglang.model_copy(
            update={
                "base_args": original.sglang.base_args
                | {
                    "dp_size": 2,
                    "pp_size": 3,
                    "ep_size": 4,
                    "api_key": "resolved",
                    "disable_cuda_graph": True,
                }
            }
        )
        args = Namespace(**{name: value for name, value in vars(original).items() if not name.startswith("sglang_")})

        result = _compute_server_args(
            args,
            node_rank=0,
            dist_init_addr="127.0.0.1:9000",
            nccl_port=9001,
            host="127.0.0.1",
            port=9002,
            disaggregation_bootstrap_port=None,
            base_gpu_id=0,
            engine_info_bootstrap_port=9003,
            sglang_overrides={},
            num_gpus_per_engine=24,
            gated_launch_port=9004,
            random_seed=17,
        )

        assert (result["dp_size"], result["pp_size"], result["ep_size"]) == (2, 3, 4)
        assert result["api_key"] == "resolved"
        assert result["disable_cuda_graph"] is True
        assert result["random_seed"] == 17
