import torch
import torch.nn.functional as F
from megatron.core.models.gpt.gpt_model import GPTModel

from miles_plugins.models.muse_spark.ops.norms import MuseSpark1xRMSNorm
from miles_plugins.models.muse_spark.spec import get_muse_spark_1x_spec


class MuseSpark1xGPTModel(GPTModel):
    def __init__(self, config, *, pg_collection, pre_process=True, post_process=True, vp_stage=None):
        if pg_collection is None:
            raise ValueError("Spark model requires explicit process groups")
        if vp_stage is not None:
            raise ValueError("Spark model does not support virtual pipeline stages")
        if pre_process != (pg_collection.pp.rank() == 0) or post_process != (
            pg_collection.pp.rank() == pg_collection.pp.size() - 1
        ):
            raise ValueError("Spark embedding/output stages must match the pipeline rank")
        spark = config.spark
        super().__init__(
            config,
            get_muse_spark_1x_spec(config, pp_rank=pg_collection.pp.rank()),
            spark.vocab_size,
            spark.max_position_embeddings,
            pre_process=pre_process,
            post_process=post_process,
            parallel_output=pg_collection.tp.size() > 1,
            share_embeddings_and_output_weights=False,
            position_embedding_type="rope" if spark.use_rope else "none",
            rotary_base=spark.rope_theta,
            pg_collection=pg_collection,
        )
        self.embedding_norm = MuseSpark1xRMSNorm(spark.hidden_size, spark.rms_norm_eps, has_weight=False)

    def _preprocess(self, input_ids, position_ids, decoder_input=None, **kwargs):
        uses_token_embeddings = decoder_input is None
        preproc_output = list(super()._preprocess(input_ids, position_ids, decoder_input, **kwargs))
        hidden_states = preproc_output[0]
        spark = self.config.spark
        if hidden_states is not None and uses_token_embeddings and spark.normalize_tok_embeddings:
            hidden_states = self.embedding_norm(hidden_states.to(self.config.params_dtype))
            hidden_states = hidden_states * spark.embedding_multiplier
        if hidden_states is not None:
            preproc_output[0] = hidden_states.to(
                torch.float32 if spark.residual_dtype == "fp32" else self.config.params_dtype
            ).contiguous()
        if spark.use_rope and position_ids is not None and kwargs.get("packed_seq_params") is None:
            positions = position_ids[0]
            preproc_output[1] = self.rotary_pos_emb.get_emb(int(positions.max()) + 1)[positions]
        return tuple(preproc_output)

    def _compute_language_model_output(
        self,
        *,
        hidden_states,
        output_layer,
        output_weight,
        labels,
        compute_language_model_loss,
        runtime_gather_output,
        **kwargs,
    ):
        spark = self.config.spark
        if spark.use_fp32_for_output:
            # Round inputs to the checkpoint dtype before the FP32 projection.
            projection_weight = output_weight if output_weight is not None else output_layer.weight
            logits = F.linear(hidden_states.to(projection_weight.dtype).float(), projection_weight.float())
        else:
            logits, _ = output_layer(
                hidden_states,
                weight=output_weight,
                runtime_gather_output=False if labels is not None else runtime_gather_output,
            )
        logits = (logits * spark.metap_output_multiplier).float()
        if spark.output_soft_cap_temp is not None:
            soft_cap = spark.output_soft_cap_temp
            logits = soft_cap * torch.tanh(logits / soft_cap)
        if labels is not None:
            return compute_language_model_loss(labels, logits)
        return logits.transpose(0, 1).contiguous()

    def forward(
        self,
        input_ids,
        position_ids=None,
        attention_mask=None,
        decoder_input=None,
        labels=None,
        # Miles passes this hint; Spark's logits transform always requires FP32.
        fp32_output=True,
        **kwargs,
    ):
        if any(
            kwargs.get(name) is not None
            for name in (
                "inference_context",
                "inference_params",
                "sequence_len_offset",
                "output_processor",
            )
        ):
            raise ValueError("Spark model supports full sequences without a KV cache")
        kwargs.pop("output_processor", None)
        # Miles' loss and log-probability functions consume one vocabulary shard.
        if kwargs.get("runtime_gather_output") is None:
            kwargs["runtime_gather_output"] = False
        # THD positions come from cu_seqlens.
        if position_ids is not None and kwargs.get("packed_seq_params") is None:
            if position_ids.shape != input_ids.shape or position_ids.ndim != 2:
                raise ValueError("position_ids and input_ids must have matching [batch, sequence] shape")
            if self.config.spark.use_rope and not torch.equal(position_ids, position_ids[:1].expand_as(position_ids)):
                raise ValueError("Spark fused RoPE requires a shared position sequence across the batch")
            if torch.any(position_ids < 0) or torch.any(position_ids >= self.max_sequence_length):
                raise ValueError("position_ids are outside max_position_embeddings")
        return super().forward(
            input_ids,
            position_ids,
            attention_mask,
            decoder_input,
            labels,
            output_processor=self._compute_language_model_output,
            **kwargs,
        )
