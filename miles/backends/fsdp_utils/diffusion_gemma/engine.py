"""Integrate diffusion SFT with the existing FSDP optimizer/checkpoint lifecycle."""

import hashlib

import torch
import torch.distributed as dist

from miles.backends.fsdp_utils.adaptations.precision import precision_forward_context
from miles.backends.fsdp_utils.diffusion_gemma.training import loss_sums, normalize_loss, prepare_batch

_DATA_KEYS = ["tokens", "response_lengths", "loss_masks", "multimodal_train_inputs"]


def _batch_seed(*, seed: int, step: int, microbatch: int, rank: int) -> int:
    # Step is restored from the native FSDP checkpoint. No rank-local generator
    # state needs to be recovered from rank zero's RNG snapshot.
    key = f"diffusion-sft:{seed}:{step}:{microbatch}:{rank}".encode()
    return int.from_bytes(hashlib.sha256(key).digest()[:8], "little") % (2**63 - 1)


def train(actor, *, rollout_id: int, rollout_data: dict) -> None:
    # Runtime imports keep standalone CPU objective/model validation independent
    # of the Ray/SGLang worker environment.
    from miles.backends.training_utils.data.rollout import get_data_iterator
    from miles.backends.training_utils.metrics.checks import check_grad_norm
    from miles.backends.training_utils.metrics.log_utils import log_train_step
    from miles.backends.training_utils.parallel import get_parallel_state

    args = actor.args
    state = get_parallel_state()
    iterators, schedule = get_data_iterator(args, actor.model_parts, rollout_data)
    if not schedule or any(count < 1 for count in schedule):
        raise ValueError("DiffusionGemma SFT requires a nonempty fixed microbatch schedule")
    for step_id, count in enumerate(schedule):
        rows = [iterators[0].get_next(_DATA_KEYS) for _ in range(count)]
        metrics, diff_loss, ar_loss = _optimizer_step(
            actor,
            rows=rows,
            dp_size=state.effective_dp.size,
            rank=state.effective_dp.rank,
            group=state.effective_dp.group,
        )
        if args.ci_test:
            check_grad_norm(
                args=args,
                grad_norm=metrics.grad_norm,
                rollout_id=rollout_id,
                step_id=step_id,
                role="actor",
                rank=state.intra_dp_cp.rank,
            )
        log_train_step(
            args=args,
            loss_dict={
                "loss": diff_loss + args.diffusion_encoder_loss_weight * ar_loss,
                "diffusion_loss": diff_loss,
                "encoder_ar_loss": ar_loss,
            },
            grad_norm=metrics.grad_norm,
            rollout_id=rollout_id,
            step_id=step_id,
            num_steps_per_rollout=len(schedule),
            role="actor",
            extra_metrics=metrics.extra_metrics,
            should_log=state.is_metrics_rank,
        )
    actor.prof.step(rollout_id=rollout_id)
    actor._after_rollout(rollout_id, rollout_data)


def _optimizer_step(actor, *, rows: list[dict], dp_size: int, rank: int, group):
    counts = _global_counts(rows, canvas_length=actor.hf_config.canvas_length, group=group)
    actor._zero_grad()
    sums = torch.zeros(2, device=rows[0]["tokens"][0].device, dtype=torch.float64)
    for microbatch, row in enumerate(rows):
        batch = _prepare(actor, row=row, microbatch=microbatch, rank=rank)
        with precision_forward_context(actor.precision_policy):
            encoder_logits, decoder_logits = actor.model(**batch.model_kwargs())
        local_sums = loss_sums(encoder_logits=encoder_logits, decoder_logits=decoder_logits, batch=batch)
        loss = normalize_loss(
            sums=local_sums,
            token_counts=counts,
            dp_size=dp_size,
            encoder_loss_weight=actor.args.diffusion_encoder_loss_weight,
        )
        loss.backward()
        sums += torch.stack([value.detach().double() for value in local_sums])
        del loss, local_sums, encoder_logits, decoder_logits, batch
    metrics = actor._apply_step()
    actor.global_step += 1
    actor.micro_step += len(rows)
    dist.all_reduce(sums, group=group)
    diff_loss, ar_loss = (sums / counts).tolist()
    return metrics, diff_loss, ar_loss


def _global_counts(rows, *, canvas_length: int, group) -> torch.Tensor:
    diffusion, encoder = 0, 0
    for row in rows:
        if row.get("multimodal_train_inputs") and any(value is not None for value in row["multimodal_train_inputs"]):
            raise ValueError("DiffusionGemma offline SFT currently accepts text-only samples")
        for tokens, response_length in zip(row["tokens"], row["response_lengths"], strict=True):
            diffusion += canvas_length
            encoder += tokens.numel() + (-response_length) % canvas_length - 1
    counts = torch.tensor([diffusion, encoder], device=rows[0]["tokens"][0].device, dtype=torch.long)
    dist.all_reduce(counts, group=group)
    if bool((counts <= 0).any()):
        raise ValueError("DiffusionGemma requires positive diffusion and encoder token counts")
    return counts


def _prepare(actor, *, row: dict, microbatch: int, rank: int):
    config = actor.hf_config
    text = config.text_config
    eos = getattr(getattr(actor, "tokenizer", None), "eos_token_id", None)
    if eos is None:
        eos = getattr(config, "eos_token_id", None)
    if eos is None:
        eos = text.eos_token_id
    if isinstance(eos, (list, tuple)):
        eos = eos[0]
    if eos is None:
        raise ValueError("DiffusionGemma requires an EOS token for terminal canvas fill")
    return prepare_batch(
        tokens=row["tokens"],
        response_lengths=row["response_lengths"],
        loss_masks=row["loss_masks"],
        canvas_length=config.canvas_length,
        sliding_window=text.sliding_window,
        eos_token_id=eos,
        pad_token_id=text.pad_token_id or 0,
        vocab_size=text.vocab_size,
        noise_epsilon=actor.args.diffusion_noise_epsilon,
        self_conditioning_probability=actor.args.diffusion_self_conditioning_probability,
        seed=_batch_seed(seed=actor.args.seed, step=actor.global_step, microbatch=microbatch, rank=rank),
    )
