from model_args_utils import load_sibling_model_args


def model_args() -> str:
    return load_sibling_model_args(__file__, "qwen3.8-2.4T-A95B", nlayers=4, mtp_num_layers=None)
