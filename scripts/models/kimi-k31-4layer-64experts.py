from model_args_utils import load_sibling_model_args


def model_args() -> str:
    # Kimi-K3.1 is K3 plus a DSA indexer on every MLA layer
    return load_sibling_model_args(__file__, "kimi-k3", nlayers=4, num_experts=64).replace(
        "miles_plugins.models.kimi_k3 get_kimi_k3_spec", "miles_plugins.models.kimi_k31 get_kimi_k31_spec"
    )
