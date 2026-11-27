# Prompt enhancement with a frozen SGLang Diffusion generator

An LLM policy rewrites a user's short image prompt. A frozen SGLang Diffusion engine, launched by Miles as a second model in the run, renders the rewrite. HPSv3, on a reward pool in the run, scores the image against the user's original prompt (or ImageReward, RePrompt's own reward, for parity), and GRPO trains the policy.

The settings are RePrompt's RL stage ([paper](https://arxiv.org/abs/2505.17540), [code](https://github.com/microsoft/DKI_LLM/tree/main/RePrompt)): Qwen2.5-3B-Instruct, 1,108 prompts for 3 epochs, 32 prompts × 4 rewrites per step, lr 2e-6, k3 KL 0.04, temperature 0.9, 256 new tokens. The generator is Z-Image-Turbo at 512×512, 9 steps, no CFG; RePrompt's FLUX.1-dev is gated on Hugging Face.

## Files

| File | What it does |
|---|---|
| `make_data.py` | Turns a prompt file into Miles prompt data: RePrompt's system prompt and assistant prefix, rendered with the policy's chat template; the user's prompt goes to `metadata`. |
| `reward.py` | `prompt_enhancer_rm`: extracts the first `<answer>`, renders it with the run's generator, scores the image against the original prompt, and adds RePrompt's format and length terms. `log_rollout_data` logs the terms and render and score latencies. |
| `scorer_server.py` | ImageReward over HTTP, in its own venv: receives PNG bytes and a prompt, answers a score. The parity path; HPSv3 runs on a reward pool instead (`miles.rollout.rm_hub.image.hpsv3`). |
| `run_qwen25_3b_prompt_enhancer.py` | The launcher. |
| `geneval.py` | GenEval images for a checkpoint: rewrites GenEval's prompts and renders them through diffusion engines in GenEval's folder layout. |

## The reward

`--pe-image-weight * image score + format + length`, the format and length terms as RePrompt computes them:

- **Image:** the text of the first `<answer>...</answer>` on one line (or `a photo`), rendered at `--pe-width` × `--pe-height` by the model `--pe-generator-model` names, scored against the user's original prompt, never the rewrite. The score comes from the reward pool `--pe-scorer-pool` names (HPSv3, weight 0.08 in the launcher) or from `scorer_server.py` at `--pe-scorer-urls` (ImageReward, RePrompt's weight 0.25, which is its `0.5 * IR / 2`). HPSv3's weight keeps the image term's spread at ImageReward's: on Z-Image-Turbo's images HPSv3 spreads 3.1 times wider.
- **Format:** 1 when the completion is one line of `<think>` reasoning, a newline and one `<answer>`; else -1.
- **Length:** 1 for an answer of 16 to 76 words, -1 outside that, 0 with no answer.

It works per sample, because Miles' eval rejects `--group-rm`, and needs `--rollout-skip-special-tokens`. When the engine fails or times out on every retry, the pool cannot score the image, or the scorer server fails three times, the reward is `None` and the group is dropped. A request the engine refuses (a 4xx, or an image of another size than asked) stops the run. Z-Image-Turbo rounds sizes to a multiple of 16, so keep `--pe-width` and `--pe-height` multiples of 16.

**Seeds.** By default a group's rewrites share one image seed, derived from the original prompt and the group's draw (`Sample.group_index`), so a group's images differ only by the rewrite and the seed changes each time a prompt is drawn. `--pe-seed-per-sample` draws one per sample instead, which matches RePrompt's own run.

## Run

GPU layout on one 8-GPU node: the job holds 1 trainer, 1 policy engine, 5 generator engines (`--generator-gpus`) and 1 HPSv3 worker (`--scorer-gpus`), declared with `--reward-pool-config` after the rollout GPUs. Download the models, build the data, convert the checkpoint and train:

```bash
python examples/experimental/prompt_enhancer/run_qwen25_3b_prompt_enhancer.py
```

The generator is declared in `--sglang-config` as a second model with `update_weights: false` and a server group of `engine_kind: diffusion`, after the policy, and the run uses `--router miles`, the router with an image route.

### ImageReward, for parity with RePrompt

The scorer runs outside the job on the eighth GPU; `--scorer-urls` alone selects it (`--scorer imagereward`), and the job keeps 7 GPUs.

1. A venv for the scorer (ImageReward needs transformers 4.x):

   ```bash
   uv venv /root/venv-ir --python 3.11
   uv pip install --python /root/venv-ir/bin/python torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
   uv pip install --python /root/venv-ir/bin/python transformers==4.51.3 image-reward==1.5 datasets==3.5.0 pyarrow==19.0.1 "numpy<2" pillow \
     fairscale timm==0.6.13 ftfy regex "git+https://github.com/openai/CLIP.git" "setuptools<70"
   CUDA_VISIBLE_DEVICES=7 /root/venv-ir/bin/python examples/experimental/prompt_enhancer/scorer_server.py --port 18200
   ```

2. Download the models, build the data, convert the checkpoint and train:

   ```bash
   CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 python examples/experimental/prompt_enhancer/run_qwen25_3b_prompt_enhancer.py \
     --scorer-urls http://127.0.0.1:18200 --seed-per-sample
   ```

### Nine epochs of each

Each configuration was trained for nine epochs (306 steps) with two training seeds, Z-Image-Turbo at 512×512 and 9 steps, and compared with the untrained policy on GenEval's 553 prompts × 8 images (see below):

- **HPSv3, the default:** ImageReward, which this run never trains on, rises +0.18 (95% CI [+0.14, +0.22]). GenEval rises +0.015 pooled, but its prompt-level 95% CI, [−0.006, +0.036], includes zero and the two seeds give −0.008 and +0.038, so a GenEval gain is not established; none further is detected after three epochs.
- **ImageReward, for parity:** GenEval +0.039 (95% CI [+0.018, +0.061]), both seeds positive.

The two differ in reward, image-seed mode and image weight at once, so this does not say which of those matters.

## Evaluate on GenEval

Start one engine per GPU with SGLang's own CLI, render each arm, then score each arm with GenEval's detector in GenEval's own environment ([GenEval](https://github.com/djghosh13/geneval), set up as its README says; `GENEVAL` below is its checkout and `MASK2FORMER` its detector config):

```bash
# SGLang derives an engine's other ports from --port, so neighbours 1 apart collide: space them and set the rest.
for g in 0 1 2 3 4 5 6 7; do
  CUDA_VISIBLE_DEVICES=$g sglang serve --model-path /root/models/Z-Image-Turbo --port $((30000 + 100 * g)) \
    --master-port $((31000 + 100 * g)) --scheduler-port $((32000 + 100 * g)) &
done
for g in 0 1 2 3 4 5 6 7; do
  until curl -sf localhost:$((30000 + 100 * g))/health > /dev/null; do sleep 10; done
done
URLS=$(seq -s, -f "http://127.0.0.1:%g" 30000 100 30700)
METADATA=$GENEVAL/prompts/evaluation_metadata.jsonl
CKPT=/root/shared_data/$RUN_ID/hf/iter_99  # the launcher's --save-hf export after 100 steps
CUDA_VISIBLE_DEVICES=0 python -m examples.experimental.prompt_enhancer.geneval --policy /root/models/Qwen2.5-3B-Instruct \
  --metadata $METADATA --image-urls $URLS --out geneval/base --images-per-prompt 8
CUDA_VISIBLE_DEVICES=0 python -m examples.experimental.prompt_enhancer.geneval --policy $CKPT \
  --metadata $METADATA --image-urls $URLS --out geneval/trained --images-per-prompt 8
for arm in base trained; do
  (cd $GENEVAL && python evaluation/evaluate_images.py ../geneval/$arm/images \
     --outfile ../geneval/$arm/results.jsonl --model-path <detector weights dir> --model-config $MASK2FORMER \
   && python evaluation/summary_scores.py ../geneval/$arm/results.jsonl)
done
```

An output directory belongs to one set of inputs (policy, prompts, seed offset, image count, sampling): rerunning with the same ones resumes, and any other is refused, so give each arm and each seed offset its own `--out`.

The rewriter runs as RePrompt's GenEval script runs it: bf16, greedy, 512 new tokens. Image `k` of prompt `i` has seed `i * 1000 + k + --seed-offset` in every arm, so images pair across arms by prompt and seed, which an exact McNemar test on image-level correctness needs. A GenEval difference between two runs of this recipe needs at least 8 images per prompt and 2 training seeds per side to read: one seed moves GenEval by about 0.015 here. To pool two trained seeds, pair each with its own base cohort at a distinct `--seed-offset`, so no base image is counted twice.
