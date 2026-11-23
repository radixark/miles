"""The TOP operations GLM-5's model code calls, injected as ``config.top_glm_ops`` by prepare_glm_config.

``miles_plugins/models/glm5/glm5.py`` runs its native path when the config carries no ops object and calls these
otherwise, so the model file imports nothing from the TOP plugin.
"""


class TopGLMOps:
    def __init__(self, *, fp8: bool):
        self.fp8 = fp8

    @staticmethod
    def rope_init_kwargs(config):
        return {"use_cpu_initialization": config.use_cpu_initialization}

    @property
    def sparse_mla(self):
        """The sparse MLA autograd function for this checkpoint's KV cache dtype."""
        if self.fp8:
            from miles_plugins.top.glm_fp8_forward import FP8SparseMLA

            return FP8SparseMLA
        from miles_plugins.top.glm_sparse_mla import TopSparseMLA

        return TopSparseMLA

    @staticmethod
    def local_spec_provider():
        from miles_plugins.top.glm_model import GLMLocalSpecProvider

        return GLMLocalSpecProvider()

    @staticmethod
    def match_linear_dtype(x, linear):
        from miles_plugins.top.glm_model import match_linear_dtype

        return match_linear_dtype(x, linear)

    def select_indices(self, index_q, index_k, weights, starts, ends, topk, *, topk_backend, replay_enabled):
        from miles_plugins.top.glm_indexer import select_indices

        if topk_backend != "flashinfer" or replay_enabled:
            raise NotImplementedError("[top] GLM index selection requires FlashInfer without replay")
        return select_indices(index_q, index_k, weights, starts, ends, topk, fp8=self.fp8)

    @staticmethod
    def select_block_rows(num_keys, max_rows):
        from miles_plugins.top.glm_indexer import select_block_rows

        return select_block_rows(num_keys, max_rows)

    @staticmethod
    def rotary(q, cu_seqlens, base, interleaved, *, gathered):
        from megatron.core import parallel_state

        from miles_plugins.top.glm_rope import rotary

        cp_size = parallel_state.get_context_parallel_world_size()
        if gathered or cp_size == 1:
            # GLM's attention gathers exactly its keys (MLA and indexer); serving rotates those in the kernel's key slot.
            return rotary(q, cu_seqlens, base, interleaved, key=gathered)
        # Local rows under all-gather CP: rotate at their global positions, as the native path does.
        # Rotary is row-wise, so this rank's rows equal its rows at CP1; only they are rotated and kept for backward.
        cp_rank = parallel_state.get_context_parallel_rank()
        return rotary(q, cu_seqlens, base, interleaved, key=False, cp_size=cp_size, cp_rank=cp_rank)

    @staticmethod
    def head_weights(hidden_states, weight):
        """Indexer head weights; the 1/sqrt(index_head_dim) factor is applied inside serving's selection."""
        from miles_plugins.top.glm_indexer import HeadWeights

        return HeadWeights.apply(hidden_states, weight)
