"""Translate gateway datums to trainer batches and sampling requests to SGLang."""

import asyncio
from dataclasses import asdict
from pathlib import Path
from typing import TextIO

import httpx

from miles.backends.training_utils.checkpoint.adapter.commit import TENANT_FILENAME
from miles.backends.training_utils.checkpoint.adapter.commit import OwnershipError as AdapterOwnershipError
from miles.backends.training_utils.checkpoint.adapter.commit import (
    UncommittedAdapterDir,
    commit_adapter_dir,
    delete_adapter_dir,
    delete_retired_adapter_dirs,
    delete_unreferenced_adapter_dirs,
    format_sglang_lora_name,
    lock_checkpoint_root,
    resolve_adapter_dir,
)
from miles.backends.training_utils.checkpoint.adapter.parametrization import (
    PARAMETRIZATION_FILENAME,
    read_parametrization,
)
from miles.ray.rollout.train_data_conversion import ROLLOUT_DATA_VALUE_SPEC
from miles.tinker.core.types import OwnershipError, UserInputError
from miles.tinker.core.utils import PRE_COMMIT_META_FILENAME
from miles.utils import object_store
from miles.utils.http_utils import post
from miles.utils.lora.hf_lora_targets import GDN_IN_PROJ_COMPONENTS_BY_MODULE, HfLoraTargets
from miles.utils.lora.utils import matches_lora_target
from tinker.types.topk_logprobs import MASK_LOGPROB

# internal datum key -> trainer batch key
DATUM_TO_BATCH_KEYS = {"weights": "loss_weights", "advantages": "advantages", "sampling_logprobs": "rollout_log_probs"}


def _pad_to_dp_multiple(slot_datums: list, dp_size: int) -> list:
    """Equalize DP shares with zero-mask datums; their outputs are dropped after the loss pass."""
    remainder = len(slot_datums) % dp_size
    if remainder == 0:
        return slot_datums
    slot, datum = slot_datums[-1]
    filler = dict(datum) | {"padding": True}
    return slot_datums + [(slot, filler)] * (dp_size - remainder)


def _build_train_data(slot_datums: list) -> dict:
    """Miles response_lengths select label positions here, including prompt targets."""
    datums = [datum for _, datum in slot_datums]
    train_data = {
        "tokens": [datum["tokens"] for datum in datums],
        "target_tokens": [datum["target_tokens"] for datum in datums],
        "loss_masks": [[0 if datum.get("padding") else 1] * datum["target_len"] for datum in datums],
        "response_lengths": [datum["target_len"] for datum in datums],
        "total_lengths": [len(datum["tokens"]) for datum in datums],
        "sample_indices": list(range(len(datums))),
        "adapter_slots": [slot for slot, _ in slot_datums],
        "dynamic_global_batch_size": len(datums),
    }
    for datum_key, batch_key in DATUM_TO_BATCH_KEYS.items():
        if datum_key in datums[0]:
            train_data[batch_key] = [datum[datum_key] for datum in datums]
    return train_data


def commit_adapter(root: str, name: str, written: UncommittedAdapterDir, overwrite: bool) -> None:
    """The gateway is the committer of every adapter dir under its checkpoint root."""
    commit_adapter_dir(Path(root), name, written, overwrite=overwrite)


def delete_retired_adapters(root: str) -> None:
    delete_retired_adapter_dirs(Path(root))


def take_checkpoint_root(checkpoint_root: str) -> TextIO:
    """Lock the root for this gateway, then recover every adapter root under it (``<model_id>/<kind>``)."""
    lock = lock_checkpoint_root(Path(checkpoint_root))
    for adapter_root in Path(checkpoint_root).glob("*/*"):
        if adapter_root.is_dir():
            delete_unreferenced_adapter_dirs(adapter_root)
    return lock


def _is_pre_commit_dir(root: str, name: str) -> bool:
    checkpoint_dir = Path(root) / name
    # by content, not by link: an overwrite interrupted after adopting a pre-commit dir leaves <name> linked to it
    return (checkpoint_dir / PRE_COMMIT_META_FILENAME).is_file() and not (
        checkpoint_dir / PARAMETRIZATION_FILENAME
    ).is_file()


def resolve_adapter(root: str, name: str, tenant: str, model_id: str) -> dict | None:
    """A committed adapter's physical dir, sglang name and parametrization; None for a dir written before commits."""
    if _is_pre_commit_dir(root, name):
        return None
    try:
        committed = resolve_adapter_dir(Path(root), name, tenant=tenant)
        if committed is None:
            return None
        parametrization = read_parametrization(committed.path)
    except AdapterOwnershipError as error:
        raise OwnershipError(f"adapter {name!r} does not belong to this tenant") from error
    # ValueError covers undecodable and unparseable files; KeyError and TypeError a parsed file of the wrong schema
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise UserInputError(f"cannot read adapter {name!r}: {error}") from error
    return {
        "path": str(committed.path),
        "lora_name": format_sglang_lora_name(model_id, name, committed.commit_id),
        "parametrization": asdict(parametrization),
    }


def resolve_checkpoint_files(root: str, name: str, tenant: str | None) -> dict | None:
    """A committed checkpoint's physical dir, commit id and the files a client may download (not the
    gateway-private tenant.json); None when it does not exist or was written before commits. ``tenant`` None skips
    the ownership check."""
    if _is_pre_commit_dir(root, name):
        return None
    try:
        committed = resolve_adapter_dir(Path(root), name, tenant=tenant)
    except AdapterOwnershipError as error:
        raise OwnershipError(f"checkpoint {name!r} does not belong to this tenant") from error
    if committed is None:
        return None
    files = sorted(p.name for p in committed.path.iterdir() if p.is_file() and p.name != TENANT_FILENAME)
    return {"path": str(committed.path), "commit_id": committed.commit_id, "files": files}


def delete_adapter(root: str, name: str, tenant: str | None) -> None:
    try:
        delete_adapter_dir(Path(root), name, tenant=tenant)
    except AdapterOwnershipError as error:
        raise OwnershipError(f"checkpoint {name!r} does not belong to this tenant") from error


# a fused HF module's adapter is recorded under its components' leaves
_FUSED_LEAF_BY_COMPONENT = {
    component: leaf for leaf, components in GDN_IN_PROJ_COMPONENTS_BY_MODULE.items() for component in components
} | {"gate_proj": "gate_up_proj", "up_proj": "gate_up_proj"}


def classify_lora_groups(target_modules, layout: HfLoraTargets) -> dict[str, bool]:
    """The train_* flags of an adapter from the HF targets it records."""
    groups = {
        "train_attn": layout.attention + layout.whole_fused_attention,
        "train_mlp": layout.mlp,
        "train_unembed": layout.unembed,
    }
    # a checkpoint records routed experts stacked (experts.down_proj) where the layout names them per expert
    groups = {
        flag: (*globs, *(g.replace(".experts.*.", ".experts.") for g in globs)) for flag, globs in groups.items()
    }
    flags = dict.fromkeys(groups, False)
    for module in target_modules:
        parent, _, leaf = module.rpartition(".")
        names = (
            (module, f"{parent}.{_FUSED_LEAF_BY_COMPONENT[leaf]}") if leaf in _FUSED_LEAF_BY_COMPONENT else (module,)
        )
        hits = [flag for flag, globs in groups.items() if any(matches_lora_target(n, g) for n in names for g in globs)]
        if not hits:
            # a checkpoint another gateway wrote may come from another model
            raise UserInputError(f"adapter target {module!r} is in no LoRA group of this model")
        flags.update(dict.fromkeys(hits, True))
    return flags


class MilesBackend:
    def __init__(self, trainer, router_url: str, lora_layout: HfLoraTargets, dp_size: int = 1) -> None:
        self.trainer = trainer
        self.router_url = router_url
        self.lora_layout = lora_layout
        self.dp_size = dp_size

    def lora_groups(self, target_modules) -> dict[str, bool]:
        return classify_lora_groups(target_modules, self.lora_layout)

    async def trainer_dead(self) -> bool:
        return await self.trainer.has_errored_cell()

    async def load_slot(self, slot: int, rank: int, alpha: float, source: dict | None = None) -> dict | None:
        return _slot_failure(await self.trainer.load_slot(slot, rank, alpha, source=source))

    async def unload_slot(self, slot: int) -> dict | None:
        return _slot_failure(await self.trainer.unload_slot(slot))

    async def forward_backward(
        self, batch_id: int, slot_datums: list, loss_fn: str, loss_fn_config: dict
    ) -> list[dict] | dict:
        return await self._execute_batch("forward_backward", batch_id, slot_datums, loss_fn, loss_fn_config)

    async def forward_only(
        self, batch_id: int, slot_datums: list, loss_fn: str, loss_fn_config: dict
    ) -> list[dict] | dict:
        return await self._execute_batch("forward_only", batch_id, slot_datums, loss_fn, loss_fn_config)

    async def _execute_batch(
        self, method: str, batch_id: int, slot_datums: list, loss_fn: str, loss_fn_config: dict
    ) -> list[dict] | dict:
        train_data = _build_train_data(_pad_to_dp_multiple(slot_datums, self.dp_size))
        train_data["loss_fn"] = loss_fn
        train_data["loss_fn_config"] = loss_fn_config
        worker_results = await self._call_trainer(method, batch_id, train_data)
        by_index: dict[int, dict] = {}
        for worker_result in worker_results:
            if "error" in worker_result:
                return worker_result
            for datum_output in worker_result["per_datum"]:
                index = int(datum_output["sample_index"])
                if index not in by_index:
                    by_index[index] = {
                        "loss": float(datum_output["loss"]),
                        "logprobs": datum_output["logprobs"].tolist(),
                    }
        return [by_index[index] for index in range(len(slot_datums))]

    async def _call_trainer(self, method: str, batch_id: int, train_data: dict) -> list:
        store = object_store.get_instance()
        data_ref = store.put(value=train_data, value_spec=ROLLOUT_DATA_VALUE_SPEC)
        try:
            return await getattr(self.trainer, method)(
                batch_id=batch_id, data_ref=data_ref
            )  # config-access-exempt: attribute selected at runtime from method
        finally:
            store.remove(data_ref)

    async def optim_step(self, adam_params_by_slot: dict[int, dict]) -> dict[int, dict]:
        worker_results = await self.trainer.optim_step(adam_params_by_slot=adam_params_by_slot)
        return worker_results[0]

    async def save_slot(
        self, slot: int, root: str, name: str, tenant_digest: str, parametrization_fields: dict
    ) -> UncommittedAdapterDir:
        """The uncommitted state every rank agreed on; a failed save raises."""
        worker_results = await self.trainer.save_slot(
            slot=slot,
            root=root,
            name=name,
            tenant_digest=tenant_digest,
            parametrization_fields=parametrization_fields,
        )
        return worker_results[0]

    async def export_slot(
        self, slot: int, root: str, name: str, tenant_digest: str, parametrization_fields: dict
    ) -> UncommittedAdapterDir:
        """The uncommitted export every rank agreed on; a worker failure raises out of the trainer call."""
        results = await self.trainer.export_slot(
            slot=slot,
            root=root,
            name=name,
            tenant_digest=tenant_digest,
            parametrization_fields=parametrization_fields,
        )
        return results[0]

    async def export_external(
        self,
        slot: int,
        root: str,
        name: str,
        tenant_digest: str,
        parametrization_fields: dict,
        age_encryption_recipients: list[str],
    ) -> UncommittedAdapterDir:
        worker_results = await self.trainer.export_external_slot(
            slot=slot,
            root=root,
            name=name,
            tenant_digest=tenant_digest,
            parametrization_fields=parametrization_fields,
            age_encryption_recipients=age_encryption_recipients,
        )
        return worker_results[0]

    commit_adapter = staticmethod(commit_adapter)
    delete_retired_adapters = staticmethod(delete_retired_adapters)
    resolve_adapter = staticmethod(resolve_adapter)
    resolve_checkpoint_files = staticmethod(resolve_checkpoint_files)
    delete_adapter = staticmethod(delete_adapter)

    # -------- sampling --------

    async def sample(self, payload: dict, lora_name: str | None, lora_path: str | None = None) -> dict:
        request = self._generate_request(payload, lora_name, lora_path)
        try:
            responses = await asyncio.gather(
                *[
                    post(f"{self.router_url}/generate", _with_sample_seed(request, index))
                    for index in range(payload["num_samples"])
                ]
            )
        except httpx.HTTPStatusError as error:
            if 400 <= error.response.status_code < 500 and error.response.status_code not in (408, 429):
                raise UserInputError(error.response.text) from error
            return {"error": f"{error}: {error.response.text}"}
        except httpx.HTTPError as error:
            return {"error": str(error)}
        sequences = [_to_sequence(response) for response in responses]
        for sequence in sequences:
            if "error" in sequence:
                return sequence
        result = {"sequences": sequences}
        if payload["prompt_logprobs"]:
            result["prompt_logprobs"] = _prompt_logprobs(responses[0])
        if payload["topk_prompt_logprobs"]:
            result["topk_prompt_logprobs"] = _topk_prompt_logprobs(responses[0], payload["topk_prompt_logprobs"])
        return result

    def _generate_request(self, payload: dict, lora_name: str | None, lora_path: str | None = None) -> dict:
        # every SDK SamplingParams field, which core has already validated
        params = payload["sampling_params"]
        sampling_params = {
            "max_new_tokens": params["max_tokens"],
            "temperature": params["temperature"],
            "top_p": params["top_p"],
            "top_k": params["top_k"],
        }
        if params["seed"] is not None:
            sampling_params["sampling_seed"] = params["seed"]
        stop = params["stop"]
        if stop is not None:
            if stop == []:
                # tinker defines stop=[] as disabling every stop token, EOS included
                sampling_params["ignore_eos"] = True
            elif isinstance(stop, list) and isinstance(stop[0], int):
                sampling_params["stop_token_ids"] = stop
            else:
                sampling_params["stop"] = stop
        request = {"input_ids": payload["prompt_tokens"], "sampling_params": sampling_params, "return_logprob": True}
        if payload["prompt_logprobs"] or payload["topk_prompt_logprobs"]:
            request["logprob_start_len"] = 0
        if payload["topk_prompt_logprobs"]:
            request["top_logprobs_num"] = payload["topk_prompt_logprobs"]
        if lora_name is not None:
            request["lora_path"] = lora_name
            if lora_path is not None:
                # request-carried backfill source: the engine refills an evicted version itself
                request["lora_backfill_paths"] = {lora_name: lora_path}
        return request


def _slot_failure(worker_results: list) -> dict | None:
    return next((result for result in worker_results if result is not None), None)


def _with_sample_seed(request: dict, index: int) -> dict:
    """Give each sample a distinct seed when the caller pins the request seed."""
    request = dict(request)
    params = dict(request["sampling_params"])
    if (seed := params.get("sampling_seed")) is not None:
        params["sampling_seed"] = seed + index
    request["sampling_params"] = params
    return request


def _prompt_logprobs(response: dict) -> list[float]:
    entries = response["meta_info"]["input_token_logprobs"]
    return [float("nan") if entry[0] is None else float(entry[0]) for entry in entries]


def _topk_prompt_logprobs(response: dict, k: int) -> dict:
    token_ids, logprobs = [], []
    for position in response["meta_info"]["input_top_logprobs"]:
        candidates = position or []
        candidate_token_ids = [entry[1] for entry in candidates][:k]
        candidate_logprobs = [float(entry[0]) for entry in candidates][:k]
        token_ids.append(candidate_token_ids + [0] * (k - len(candidate_token_ids)))
        logprobs.append(candidate_logprobs + [MASK_LOGPROB] * (k - len(candidate_logprobs)))
    return {"token_ids": token_ids, "logprobs": logprobs}


def _to_sequence(response: dict) -> dict:
    output_token_logprobs = response["meta_info"]["output_token_logprobs"]
    finish = response["meta_info"]["finish_reason"]["type"]
    if finish == "abort":
        # a truncated sequence must fail the request, not pass as a completed sample
        return {"error": "the engine aborted this sample; resubmit the request"}
    return {
        "tokens": [entry[1] for entry in output_token_logprobs],
        "logprobs": [entry[0] for entry in output_token_logprobs],
        "stop_reason": "length" if finish == "length" else "stop",
    }
