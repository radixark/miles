from model_args_utils import load_sibling_model_args


def model_args() -> str:
    # 4-layer partial from `tools/convert_mimo_v2_to_bf16.py --layers 0,1,5,6`:
    # global+dense, SWA+MoE, global+MoE, SWA+MoE (every decoder variant, PP=2 splits it in half).
    return load_sibling_model_args(__file__, "mimo-v2.6-flash", nlayers=4)
