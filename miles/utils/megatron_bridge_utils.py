from contextlib import contextmanager
from megatron.core.utils import unwrap_model


@contextmanager
def patch_megatron_model(model):
    model_config = unwrap_model(model)[0].config
    assert not hasattr(model_config, "share_embeddings_and_output_weights")
    setattr(model_config, "share_embeddings_and_output_weights", TODO)

    try:
        yield
    finally:
        delattr(model_config, "share_embeddings_and_output_weights")
