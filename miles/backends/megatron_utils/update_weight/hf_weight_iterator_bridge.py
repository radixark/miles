import dataclasses
import torch 

from miles.utils import megatron_bridge_utils
from miles.utils.iter_utils import chunk_named_params_by_size

from ..megatron_to_hf import postprocess_hf_param
from ..misc_utils import strip_param_name_prefix
from .hf_weight_iterator_base import HfWeightIteratorBase

##############################
###########lora###############
##############################
def _normalize_base_weight_name(param_name: str) -> str:
    """Remove the 'base_layer' suffix emitted when merge_adapter_weights=False."""
    if param_name.endswith("base_layer.weight"):
        return param_name[: -len("base_layer.weight")] + "weight"
    return param_name
##############################
##############################
##############################


class HfWeightIteratorBridge(HfWeightIteratorBase):

    def __init__(self, *args, **kwargs):
        
        super().__init__(*args, **kwargs)

        # ##############################
        # ###########lora###############
        # ##############################
        # self.is_lora = is_lora # already get from HfWeightIteratorBase 
        # self._base_synced = _base_synced # already get from HfWeightIteratorBase 
        # ##############################
        # ##############################
        # ##############################

        from megatron.bridge import AutoBridge

        import miles_plugins.megatron_bridge  # noqa: F401

        self._bridge = AutoBridge.from_hf_pretrained(self.args.hf_checkpoint, trust_remote_code=True)

    def get_hf_weight_chunks(self, megatron_local_weights):
        # TODO support quantization (e.g. modify megatron-bridge to provide megatron param name)
        renamed_megatron_local_weights = {strip_param_name_prefix(k): v for k, v in megatron_local_weights.items()}
        with megatron_bridge_utils.patch_megatron_model(self.model):
            ##############################
            ###########lora###############
            ##############################
            # ## This is the origin way - weight sync will process - base model + lora weights  
            # ## to-do (yusheng): Optimize: use the method in  `self.is_lora` but need to deal with CUDA issue (weight not on the same device) - might need to be delt with in megatron-core 

            # conversion_tasks = self._bridge.get_conversion_tasks(self.model)
            # conversion_tasks = _process_conversion_tasks(conversion_tasks, renamed_megatron_local_weights)

            # named_weights = self._bridge.export_hf_weights(self.model, cpu=False, conversion_tasks=conversion_tasks)

            # # for hf_param_name, weight, megatron_param_name in named_weights:
            # #     print(hf_param_name)
            # # exit()
            
            # named_weights = (
            #     (
            #         hf_param_name,
            #         postprocess_hf_param(
            #             args=self.args,
            #             megatron_param_name=megatron_param_name,
            #             hf_param_name=hf_param_name,
            #             param=weight,
            #         ),
            #     )
            #     for hf_param_name, weight, megatron_param_name in named_weights
            # )

            # yield from chunk_named_params_by_size(named_weights, chunk_size=self.args.update_weight_buffer_size)

            ####
            
            # Only sync base model on first call (or if not LoRA-only mode)
            # if not self.is_lora or not self._base_synced:
            if not self.is_lora:
                # only pass base model 
                conversion_tasks = self._bridge.get_conversion_tasks(self.model)
                conversion_tasks = _process_conversion_tasks(conversion_tasks, renamed_megatron_local_weights)
                named_weights = self._bridge.export_hf_weights(
                    self.model, 
                    cpu=False, 
                    conversion_tasks=conversion_tasks,
                    merge_adapter_weights=not self.is_lora, # Do not return merged (base.weight + lora.weight). 
                )

                # for hf_param_name, weight, megatron_param_name in named_weights:
                #     print(hf_param_name) 
                
                named_weights = (
                    (
                        ##############################
                        ###########lora###############
                        ##############################
                        hf_param_name,
                        # _normalize_base_weight_name(hf_param_name),
                        ##############################
                        ##############################
                        ##############################
                        postprocess_hf_param(
                            args=self.args,
                            megatron_param_name=megatron_param_name,
                            hf_param_name=hf_param_name,
                            param=weight,
                        ),
                    )
                    for hf_param_name, weight, megatron_param_name in named_weights
                )
                yield from chunk_named_params_by_size(named_weights, chunk_size=self.args.update_weight_buffer_size)
                if self.is_lora:
                    self._base_synced = True
                    # torch.cuda.synchronize()
            ##############################
            ##############################
            ##############################



            ##############################
            ###########lora###############
            ##############################
            if self.is_lora:
                lora_weights = self._bridge.export_adapter_weights(
                    self.model,
                    cpu=False,
                    show_progress=False
                )

                lora_weights = (
                    (
                        hf_param_name,
                        postprocess_hf_param(
                            args=self.args,
                            megatron_param_name=megatron_param_name,
                            hf_param_name=hf_param_name,
                            param=weight,
                        ),
                    )
                    for hf_param_name, weight, megatron_param_name in lora_weights
                )

                yield from chunk_named_params_by_size(lora_weights, chunk_size=self.args.update_weight_buffer_size)
            ##############################
            ##############################
            ##############################


def _process_conversion_tasks(vanilla_conversion_tasks, new_weight_dict):
    def _handle_one(task):
        if task.param_weight is None:
            return task

        weight_dict_key = f"vp_stages.{task.vp_stage}.{task.param_name}"
        assert (
            weight_dict_key in new_weight_dict
        ), f"{weight_dict_key=} not in new_weight_dict ({task.vp_stage=}, {task.param_name=}, {list(new_weight_dict)=})"

        new_param_weight = new_weight_dict[weight_dict_key]
        new_param_weight = new_param_weight.cuda()
        return dataclasses.replace(task, param_weight=new_param_weight)

    return _MapWithLen(_handle_one, vanilla_conversion_tasks)


class _MapWithLen:
    def __init__(self, fn, xs):
        self.fn = fn
        self.xs = xs

    def __len__(self):
        return len(self.xs)

    def __iter__(self):
        for x in self.xs:
            yield self.fn(x)
