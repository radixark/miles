import asyncio
import copy
import logging
from typing import Any

from tqdm import tqdm

from miles.rollout.base_types import RolloutFnConstructorInput, RolloutFnEvalInput, RolloutFnEvalOutput
from miles.rollout.inference_rollout.inference_rollout_common import (
    GenerateState,
    compute_sampling_params,
    generate_and_rm,
)
from miles.utils.data import Dataset
from miles.utils.eval_config import EvalDatasetConfig
from miles.utils.misc import as_completed_async
from miles.utils.processing_utils import load_processor, load_tokenizer
from miles.utils.types import Sample

logger = logging.getLogger(__name__)


async def eval_rollout_single_dataset(
    state: GenerateState,
    dataset_cfg: EvalDatasetConfig,
    prompt_dataset_cache: dict[Any, Dataset],
) -> dict[str, dict[str, list[Any]]]:
    args = state.args
    assert not args.group_rm, "Group RM is not supported for eval rollout"

    cache_key = dataset_cfg.cache_key + (args.hf_checkpoint, args.apply_chat_template)
    if cache_key not in prompt_dataset_cache:
        tokenizer = load_tokenizer(args.hf_checkpoint, trust_remote_code=True)
        processor = load_processor(args.hf_checkpoint, trust_remote_code=True)
        prompt_dataset_cache[cache_key] = Dataset(
            path=dataset_cfg.path,
            tokenizer=tokenizer,
            processor=processor,
            max_length=args.eval_max_prompt_len,
            prompt_key=dataset_cfg.input_key,
            label_key=dataset_cfg.label_key,
            multimodal_keys=args.multimodal_keys,
            metadata_key=dataset_cfg.metadata_key,
            tool_key=dataset_cfg.tool_key,
            apply_chat_template=args.apply_chat_template,
            apply_chat_template_kwargs=args.apply_chat_template_kwargs,
        )
    dataset = prompt_dataset_cache[cache_key]

    base_sampling_params = compute_sampling_params(
        args,
        temperature=dataset_cfg.temperature,
        top_p=dataset_cfg.top_p,
        top_k=dataset_cfg.top_k,
        max_new_tokens=dataset_cfg.max_response_len,
    )

    tasks = []
    # do multiple samples for eval prompts
    sample_index = 0
    for _i, prompt_sample in enumerate(dataset.samples):
        for j in range(dataset_cfg.n_samples_per_eval_prompt):
            # use the same prompt for multiple samples
            sample = copy.deepcopy(prompt_sample)
            sample.index = sample_index
            sample_index += 1
            sample.metadata = dataset_cfg.inject_metadata(getattr(sample, "metadata", None))
            sampling_params = base_sampling_params
            if getattr(args, "sglang_enable_deterministic_inference", False):
                sampling_params = base_sampling_params.copy()
                sampling_params["sampling_seed"] = args.rollout_seed + j
            tasks.append(
                asyncio.create_task(
                    generate_and_rm(
                        state,
                        sample,
                        sampling_params=sampling_params,
                        evaluation=True,
                    )
                )
            )

    data = []
    do_print = True
    pbar = tqdm(total=len(tasks), desc=f"Eval {dataset_cfg.name}", disable=not do_print)
    async for sample in as_completed_async(tasks):
        if do_print:
            # TODO improve this after enhancing samples' type
            s = (sample[0] if len(sample) > 0 else None) if isinstance(sample, list) else sample
            if s is not None:
                logger.info(
                    "eval_rollout_single_dataset example data: "
                    f"{[str(s.prompt) + s.response]} "
                    f"reward={s.reward}"
                )
            do_print = False
        if isinstance(sample, list):
            data.extend(sample)
        else:
            data.append(sample)
        pbar.update(1)
    pbar.close()

    data.sort(key=lambda sample: sample.index)

    reward_key = args.eval_reward_key or args.reward_key
    return {
        dataset_cfg.name: {
            "rewards": [sample.reward if not reward_key else sample.reward[reward_key] for sample in data],
            "truncated": [sample.status == Sample.Status.TRUNCATED for sample in data],
            "samples": data,
        }
    }


class SimpleEvalRolloutFn:
    def __init__(self, input: RolloutFnConstructorInput):
        self.prompt_dataset_cache = {}
        self.state = GenerateState(input.args)

    async def __call__(self, input: RolloutFnEvalInput) -> RolloutFnEvalOutput:
        assert not self.state.args.group_rm, "Group RM is not supported for eval rollout"

        coros = []
        for dataset_cfg in getattr(self.state.args, "eval_datasets", []) or []:
            coros.append(eval_rollout_single_dataset(self.state, dataset_cfg, self.prompt_dataset_cache))
        results_list = await asyncio.gather(*coros)
        results = {k: v for r in results_list for k, v in r.items()}
        return RolloutFnEvalOutput(data=results)
