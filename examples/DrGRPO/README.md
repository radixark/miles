# Dr.GRPO Constant pg-loss Normalization

By default, miles divides each sample's policy-gradient loss by its number of effective tokens (`loss_mask.sum()`). The Dr.GRPO paper ([arXiv:2503.20783](https://arxiv.org/abs/2503.20783), Eq. 2) identifies this per-sample, length-dependent normalization as a source of length bias and instead divides by a **constant** equal to the model's max context length. The same rule is used by DeepSWE ([blog](https://www.together.ai/blog/deepswe), `max_context_length` ~ 40k).

## Usage

Set the constant divisor with `--pg-loss-divisor` (typically the model's max context length):

```bash
GRPO_ARGS=(
   --advantage-estimator grpo
   --pg-loss-divisor 40960
   # ... other arguments
)
```

Notes:

- Only `pg_loss` is affected; other metrics (`pg_clipfrac`, `ppo_kl`, `entropy_loss`, etc.) keep the default per-sample-mean reduction.
- The divisor must be a positive number; startup fails loud on a non-positive value, so a misconfiguration can never silently rescale the gradient.
- Context parallelism is supported: the divisor is a shared constant on every CP rank, so Megatron's gradient sum-allreduce reproduces the `cp_size == 1` result with no extra communication.
- With `--calculate-per-token-loss` the divisor has no effect (Megatron then normalizes by token count itself).
