---
title: Search-R1 (Tool Use)
description: Train a model to issue search queries, integrate observations, and answer multi-turn QA.
---

# Search-R1 — Tool-augmented multi-turn RL

**What you'll learn:** how to wire up a tool (web search) into a Miles training loop —
custom multi-turn rollout, observation interleaving, reward function, and TIS to keep
training stable when train ≠ inference.

This is a Miles-friendly reproduction of the original
[Search-R1](https://github.com/PeterGriffinJin/Search-R1).

## Prerequisites

* `radixark/miles:latest` container.
* Either a serper.dev API key (Google search backend) or ~135 GB free disk for the
  local Wikipedia retriever (see [appendix](#appendix-local-wikipedia-retriever)).
* You completed [Customization](../user-guide/customization.md) — this example uses a
  custom rollout function and reward.

## Files

```text
examples/search-r1/
├── generate_with_search.py       # custom rollout (multi-turn loop)
├── google_search_server.py       # serper.dev wrapper
├── local_search_server.py        # FastAPI server in front of FAISS index
├── local_dense_retriever/        # E5-base index/corpus downloader
├── qa_em_format.py               # exact-match reward
└── run_qwen2.5_3B.sh             # launch script
```

## Quick start

### 1. Set up environment

```bash
cd /root && git clone https://github.com/radixark/miles.git
cd miles && pip install -e . --no-deps && pip install chardet
```

### 2. Prepare data

```bash
git clone https://github.com/PeterGriffinJin/Search-R1.git
cd Search-R1 && pip install -e . --no-deps && pip install tensordict

WORK_DIR=/root/Search-R1
LOCAL_DIR=$WORK_DIR/data/nq_hotpotqa_train
python $WORK_DIR/scripts/data_process/qa_search_train_merge.py \
    --local_dir $LOCAL_DIR \
    --data_sources nq,hotpotqa
```

### 3. Convert the model

```bash
hf download Qwen/Qwen2.5-3B --local-dir /root/Qwen2.5-3B
cd /root/miles
source scripts/models/qwen2.5-3B.sh
PYTHONPATH=/root/Megatron-LM python tools/convert_hf_to_torch_dist.py \
   ${MODEL_ARGS[@]} \
   --hf-checkpoint /root/Qwen2.5-3B \
   --save           /root/Qwen2.5-3B_torch_dist
```

### 4. Run

```bash
bash examples/search-r1/run_qwen2.5_3B.sh
```

## Configuration

Open `generate_with_search.py` and edit `SEARCH_R1_CONFIGS`:

```python
SEARCH_R1_CONFIGS = {
    "max_turns": 2,
    "topk": 3,
    "search_concurrency": 256,
    "search_backend": "local",     # or "google"

    "local": {
        "search_url": "http://127.0.0.1:8000/retrieve",
        "proxy": None,
    },

    "google": {
        "api_key": "your_serper_key",
        "snippet_only": True,
        "proxy": None,
    },

    "return_logprob": True,        # required for TIS
    "format_score": 0.2,
}
```

## Walkthrough — multi-turn rollout

The custom rollout lives in `generate_with_search.py:generate`. The loop is
straightforward but every step matters:

```python
async def generate(args, sample: Sample, sampling_params) -> Sample:
    prompt = build_prompt(sample)
    full_response, loss_masks, tokens = "", [], []

    for turn in range(SEARCH_R1_CONFIGS["max_turns"]):
        # 1. Model generates an action
        out = await call_sglang(prompt + full_response, sampling_params)
        toks = tokenize(out.text)
        full_response += out.text
        tokens += toks
        loss_masks += [1] * len(toks)        # model tokens count toward loss

        # 2. Parse action
        action, content = parse_action(out.text)

        # 3. Run the tool
        if action == "search":
            result = await search_backend(content, topk=SEARCH_R1_CONFIGS["topk"])
            obs_text = render_observation(result)
            obs_toks = tokenize(obs_text)
            full_response += obs_text
            tokens += obs_toks
            loss_masks += [0] * len(obs_toks)   # observation tokens MASKED OUT
        elif action == "answer":
            break

    sample.response   = full_response
    sample.tokens     = tokens
    sample.loss_mask  = loss_masks
    sample.metadata["turns_used"] = turn + 1
    return sample
```

### The two crucial details

1. **Loss masking.** Tool/observation tokens get `loss_mask=0`. Without this, the model
   learns to *predict the search results*, which is both wrong and wildly unhelpful.
2. **Tokenization alignment.** The model must see and the trainer must score the
   *exact same tokens*. Pre-tokenizing vs. re-tokenizing at training time can drift —
   that's where the [chat template verifier](../user-guide/agentic-chat-template.md)
   matters.

## Walkthrough — reward

```python
async def reward_func(args, sample: Sample, **kwargs) -> float:
    answer = extract_final_answer(sample.response)
    label  = sample.label
    em     = exact_match(answer, label)
    fmt    = SEARCH_R1_CONFIGS["format_score"] if has_valid_format(sample.response) else 0
    return em + fmt
```

`format_score=0.2` gives partial credit for the correct `<answer>...` shape even if the
content is wrong — keeps gradient flowing during early exploration.

## Enabling TIS

The trajectory mixes model tokens (we want gradients) with tool tokens (we don't).
Without correction, the implicit policy ratio in the GRPO objective is *off-policy* —
the search results came from a stochastic environment, not the model.

**Truncated Importance Sampling (TIS)** corrects for this. To enable:

1. Set `"return_logprob": True` in `SEARCH_R1_CONFIGS`.
2. Uncomment the TIS flags in `run_qwen2.5_3B.sh`:

```bash
GRPO_ARGS+=( --use-tis )
CUSTOM_ARGS+=(
   --custom-config-path examples/train_infer_mismatch_helper/mis.yaml
   --custom-tis-function-path examples.train_infer_mismatch_helper.mis.compute_mis_weights_with_cp
)
```

When `return_logprob=True`, response post-processing is automatically disabled to keep
token / logp alignment.

## What to watch

```text
search_r1/turns_per_sample          ~1.5 (depends on max_turns)
search_r1/search_calls_per_sample    ~1.0
reward/exact_match                   trending up
reward/format                        ~0.18 (steady — most outputs are well-formed)
loss_mask/observation_fraction       0.4 – 0.7 (lots of obs tokens, all masked)
tis/effective_sample_size            > 0.7 × batch_size
```

If `tis/effective_sample_size` collapses below 0.5, your inference distribution has
drifted too far. Lower `--lr` or shorten `max_turns`.

## Tuning knobs

| Knob | Effect |
|---|---|
| `max_turns` | More turns = more retrieval, more drift |
| `topk` | More retrieved snippets = longer context |
| `search_concurrency` | Cap on simultaneous tool calls (mind your QPS limit) |
| `format_score` | Partial credit for correct shape — higher = faster early shaping |

## Troubleshooting

| Problem | Fix |
|---|---|
| "Ray process stuck" | `rm -rf /root/.cache`, then `rm -rf /root/.*` if still stuck |
| Retriever 502 errors | `lsof -i :8000` — make sure your local server is alive |
| Conda activation collisions | Deactivate the `retriever` env before launching training |
| EM stays at 0 | Check the answer extractor — most often a regex mismatch |
| Loss masks shifted by one token | Tokenizer added a leading space; align with `add_special_tokens=False` |

## Variations

* **Use Google instead of local.** Set `"search_backend": "google"` and add an API key.
* **Different tool.** Replace `search_backend` with anything else — calculator, code
  exec, internal API. The pattern is identical.
* **Group RM.** With multiple trajectories per prompt (GRPO), enable `--group-rm` so
  rewards are computed in a batch.
* **Longer chains.** Bump `max_turns` to 8+ for deep-reasoning tasks. Watch
  `loss_mask/observation_fraction` — if it dominates, the model is barely training.

## Appendix — local Wikipedia retriever

Heavy but completely offline. ~135 GB total disk and a separate conda env to avoid
conflicting with Miles.

### One-time setup

```bash
# 1. Conda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh
bash ~/miniconda.sh -b -p $HOME/miniconda3
source ~/miniconda3/etc/profile.d/conda.sh

conda create -n retriever python=3.10 -y && conda activate retriever
conda install pytorch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 \
              pytorch-cuda=12.1 -c pytorch -c nvidia -y
pip install transformers datasets pyserini huggingface_hub uvicorn fastapi
conda install faiss-gpu=1.8.0 -c pytorch -c nvidia -y

# 2. Index + corpus (~135 GB)
save_path=/root/Index
python /root/miles/examples/search-r1/local_dense_retriever/download.py \
    --save_path $save_path
cat $save_path/part_* > $save_path/e5_Flat.index
gzip -d $save_path/wiki-18.jsonl.gz
```

### Run the server

```bash
conda activate retriever
python /root/miles/examples/search-r1/local_dense_retriever/retrieval_server.py \
    --index_path /root/Index/e5_Flat.index \
    --corpus_path /root/Index/wiki-18.jsonl \
    --topk 3 \
    --retriever_name e5 \
    --retriever_model intfloat/e5-base-v2 \
    --faiss_gpu
```

5–7 GB of GPU memory per GPU. First startup is slow (model + index load); subsequent
restarts are 1–2 minutes.

### Then launch training

```bash
conda deactivate              # don't train inside the retriever env!
cd /root/miles
bash examples/search-r1/run_qwen2.5_3B.sh
```
