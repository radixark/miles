# Packed SFT Rollout

Miles can train pre-packed SFT blocks by using a custom data source and rollout
function. This path is useful when samples are pre-tokenized offline into long
blocks, for example near-128K DeepSeek-V4 SFT blocks.

Each JSONL row is one training sample:

```json
{"tokens":[0,128803,42,43],"loss_mask":[0,0,1,1],"metadata":{"source":"example"}}
```

The required fields are:

- `tokens`: full token ids for one packed block.
- `loss_mask`: either a full-length 0/1 mask aligned to `tokens`, or a tail mask
  when `response_length` is also provided. Full-length masks may be sparse, so
  one packed block can contain multiple conversations and multiple assistant
  spans.
- `response_length`: optional. When omitted with a full-length mask, Miles keeps
  token 0 as the prefix token and uses `tokens[1:]` as the response window.

The first trainable token cannot be token position 0. Miles shifts the response
mask into next-token prediction positions during training, so every packed block
needs at least one prefix/context token before the first trainable token.

Use:

```bash
--data-source-path miles.rollout.packed_sft_rollout.PackedSFTDataSource \
--rollout-function-path miles.rollout.packed_sft_rollout.generate_rollout \
--prompt-data /path/to/packed.jsonl \
--input-key ignored \
--rollout-batch-size 64 \
--global-batch-size 64 \
--loss-type sft_loss \
--calculate-per-token-loss \
--disable-compute-advantages-and-returns \
--debug-train-only
```

For DeepSeek-V4 BSHD long-context training, this is compatible with fixed
`--micro-batch-size 1`, `--qkv-format bshd`, CP, and `--allgather-cp`.

This is offline packing, not native multi-sample THD sequence packing. Tokens in
one JSONL row form one continuous autoregressive context, so add separators in
the offline packer if document boundaries matter.

To build a packed JSONL from an OpenAI-style SFT JSONL:

```bash
python scripts/tools/pack_sft_jsonl.py \
  --input /path/to/train.shuffle.max131073.jsonl \
  --output /path/to/train.shuffle.max131073.packed128k.jsonl \
  --hf-checkpoint /path/to/DeepSeek-V4-Flash-bf16 \
  --loss-mask-type deepseek_v4 \
  --block-size 131072
```

The packer emits `response_length = len(tokens) - 1` and `loss_mask[1:]`, so
the first token in each packed block is context only. This matches Miles'
next-token loss alignment while still allowing sparse loss over multiple
assistant spans inside the block.
