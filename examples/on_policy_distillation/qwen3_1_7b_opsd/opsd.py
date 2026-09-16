"""Example-local teacher scoring, weighted targets, and clipped forward KL."""

import math

import torch
from math_verify import parse, verify

from miles.backends.training_utils.loss_hub.logit_processors import get_responses
from miles.ray.rollout.train_data_conversion import convert_samples_to_train_data
from miles.utils.http_utils import post
from miles.utils.processing_utils import load_tokenizer


def _validate(args):
    if args.train_backend != "megatron" or args.object_store_backend != "ray":
        raise ValueError("This example requires Megatron and the Ray object store.")
    if args.tensor_model_parallel_size != 1 or args.context_parallel_size != 1 or args.true_on_policy_mode:
        raise ValueError("This example requires TP=CP=1 and true_on_policy_mode disabled.")
    if args.use_dynamic_global_batch_size or args.use_opd or args.compute_advantages_and_returns:
        raise ValueError("Use a fixed global batch and disable core OPD and advantage computation.")
    if not isinstance(args.opsd_top_k, int) or args.opsd_top_k < 1:
        raise ValueError("opsd_top_k must be positive.")
    if not math.isfinite(args.opsd_kl_coef) or args.opsd_kl_coef < 0:
        raise ValueError("opsd_kl_coef must be finite and nonnegative.")
    if args.opsd_kl_clip is not None and (not math.isfinite(args.opsd_kl_clip) or args.opsd_kl_clip <= 0):
        raise ValueError("opsd_kl_clip must be positive and finite, or null.")


def _teacher_targets(response, response_ids, top_k):
    length = len(response_ids)
    info = response["meta_info"]
    scored = info["input_token_logprobs"][-length:] if length else []
    if [int(entry[1]) for entry in scored] != response_ids:
        raise ValueError("Teacher/student response token mismatch.")
    rows = info["input_top_logprobs"][-length:] if length else []
    if len(rows) != length:
        raise ValueError("Missing teacher top-k positions.")
    ids, weights = [], []
    for row in rows:
        entries = [entry for entry in (row or []) if entry is not None][:top_k]
        if not entries:
            raise ValueError("Empty teacher support.")
        token_ids = [int(entry[1]) for entry in entries]
        logprobs = [float(entry[0]) for entry in entries]
        if len(set(token_ids)) != len(token_ids) or min(token_ids) < 0:
            raise ValueError("Invalid teacher token IDs.")
        if any(not math.isfinite(value) or value > 0 for value in logprobs):
            raise ValueError("Invalid teacher log probabilities.")
        ids.extend(token_ids + [0] * (top_k - len(entries)))
        weights.extend([math.exp(value) for value in logprobs] + [0.0] * (top_k - len(entries)))
    return {"target_tokens": ids, "loss_weights": weights}


def _extract_boxed(text):
    start = text.rfind("\\boxed{")
    if start < 0:
        return None
    depth = 0
    for i in range(start + 6, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[start + 7 : i].strip()
    return None


def _is_correct(response, label):
    boxed = _extract_boxed(response)
    if boxed is None:
        return 0.0
    gold = parse(f"${label}$", fallback_mode="no_fallback", parsing_timeout=None)
    predicted = parse(f"${boxed}$", fallback_mode="no_fallback", parsing_timeout=None)
    return float(verify(gold, predicted, timeout_seconds=None))


async def reward_func(args, sample, **kwargs):
    if sample.metadata.get("opsd_eval"):
        return _is_correct(sample.response or "", str(sample.label))
    _validate(args)
    tokenizer = load_tokenizer(args.hf_checkpoint, chat_template_path=args.chat_template_path)
    prompt_ids = tokenizer.encode(sample.metadata["teacher_prompt"], add_special_tokens=False)
    response_ids = sample.tokens[len(sample.tokens) - sample.response_length :]
    response = await post(
        args.rm_url,
        {
            "input_ids": list(prompt_ids) + list(response_ids),
            "sampling_params": {"temperature": 0, "max_new_tokens": 0, "skip_special_tokens": False},
            "return_logprob": True,
            "top_logprobs_num": args.opsd_top_k,
            "logprob_start_len": len(prompt_ids) - 1,
        },
    )
    sample.train_metadata = _teacher_targets(response, list(response_ids), args.opsd_top_k)
    return 0.0


def convert_samples(args, samples):
    _validate(args)
    data = convert_samples_to_train_data(args, samples, {}, None, None)
    # Existing target fields carry flattened [response, top-k] IDs and probabilities.
    for key in ("target_tokens", "loss_weights"):
        data[key] = [sample.train_metadata[key] for sample in samples]
    data.pop("metadata", None)
    return data


def loss_function(args, batch, logits, sum_of_sample_mean):
    _validate(args)
    responses = get_responses(
        logits,
        args=args,
        unconcat_tokens=batch["unconcat_tokens"],
        total_lengths=batch["total_lengths"],
        response_lengths=batch["response_lengths"],
        max_seq_lens=batch.get("max_seq_lens"),
    )
    losses, coverage, clipped = [], [], []
    for (student_logits, _), targets, weights in zip(
        responses, batch["target_tokens"], batch["loss_weights"], strict=True
    ):
        shape = (student_logits.shape[0], args.opsd_top_k)
        ids = torch.as_tensor(targets, device=logits.device, dtype=torch.long).reshape(shape)
        teacher = torch.as_tensor(weights, device=logits.device, dtype=torch.float32).detach().reshape(shape)
        student_logp = student_logits.float().log_softmax(-1).gather(1, ids)
        terms = torch.special.xlogy(teacher, teacher) - teacher * student_logp
        # Clip vocabulary contributions before summing, not the per-token KL.
        clip = args.opsd_kl_clip
        losses.append((terms if clip is None else terms.clamp(max=clip)).sum(-1))
        coverage.append(teacher.sum(-1))
        clipped.append(torch.zeros_like(teacher[:, 0]) if clip is None else (terms > clip).float().mean(-1))
    kl = sum_of_sample_mean(torch.cat(losses))
    loss = args.opsd_kl_coef * kl
    return loss, {
        "loss": loss.detach(),
        "opd_forward_kl": kl.detach(),
        "opd_teacher_coverage": sum_of_sample_mean(torch.cat(coverage)).detach(),
        "opd_kl_clipfrac": sum_of_sample_mean(torch.cat(clipped)).detach(),
    }
