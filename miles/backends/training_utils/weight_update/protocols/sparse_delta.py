"""Disk-delta with the 'sparse-xor' encoding: diffed on the trainer's GPU, applied in place on the engines.

Same published layout and hooks as disk-delta. The format is in miles/utils/sparse_delta.py.
"""

import json
import os
import shutil
from concurrent import futures

import safetensors.torch
import torch
import torch.distributed as dist
from sglang.srt import utils as sglang_utils

from miles.backends.training_utils.weight_update import session
from miles.backends.training_utils.weight_update.protocols import delta
from miles.utils import async_utils, disk_delta, distributed_utils, sparse_delta
from miles.utils.args import custom_view


class UpdateWeightFromSparseDelta(delta.UpdateWeightFromDiskDelta):
    """Keeps one extra copy of each sender rank's HF weights on GPU."""

    # The in-place apply needs a weight-update session.
    use_weight_update_session = True

    def __init__(self, args) -> None:
        super().__init__(args)
        self._old_bits: dict[str, torch.Tensor] = {}
        self._selector = "all"
        self._engine_gpu_counts: list[int] = []

    def connect(self, rollout_engines, engine_gpu_counts, *args, selector: str, **kwargs) -> None:
        super().connect(rollout_engines, engine_gpu_counts, *args, selector=selector, **kwargs)
        self._selector = selector
        self._engine_gpu_counts = list(engine_gpu_counts)

    def _capture_baseline(self, iter_buckets) -> None:
        """Seed the GPU copy from hf_checkpoint and clear old versions."""
        if dist.get_rank() == 0:
            shutil.rmtree(self.delta_dir, ignore_errors=True)
            os.makedirs(self.delta_dir, exist_ok=True)
            if self._post_write_hook is not None:
                fn_args = custom_view.compute_custom_function_config(
                    self.args, self.args.custom_update_weight_post_write_path
                )
                self._post_write_hook(fn_args, self.delta_dir, list(self.rollout_engines))
        dist.barrier(group=distributed_utils.get_gloo_group())
        read_hf = disk_delta.make_tensor_reader(self.args.hf_checkpoint)
        for bucket in iter_buckets(materialize=self.is_sender):
            if not self.is_sender:
                continue
            for name, tensor in bucket:
                baseline = read_hf(
                    name,
                    expected_dtype=delta._safetensors_dtype(tensor.dtype),
                    expected_shape=tuple(tensor.shape),
                )
                self._old_bits[name] = sparse_delta.get_bits(
                    torch.tensor(baseline, device=tensor.device).view(tensor.dtype)
                )

    def _begin_encode(self) -> None:
        self._version_dir = os.path.join(self.delta_dir, f"weight_v{self._snapshot_file_version:06d}")
        if self.is_sender:
            os.makedirs(self._version_dir, exist_ok=True)
        self._delta = {}
        # Per tensor: dtype, shape and checksum.
        self._checksums = {}
        self.changed_bytes = self.total_bytes = 0
        # Compress on host threads while the GPU diffs the next tensor.
        self._pool = futures.ThreadPoolExecutor(disk_delta.NUM_WORKERS)
        self._compressing: dict[str, tuple[futures.Future, dict]] = {}
        self._full: dict[str, torch.Tensor] = {}

    def send_bucket(self, bucket: list[tuple[str, torch.Tensor]]) -> None:
        for name, tensor in bucket:
            new_bits = sparse_delta.get_bits(tensor.detach().contiguous()).reshape(-1)
            encoded = sparse_delta.encode_delta(new_bits, self._old_bits[name]).cpu().numpy()
            self.total_bytes += new_bits.numel() * tensor.element_size()
            if self.args.update_weight_delta_verify:
                self._full[name] = tensor.detach().cpu()
            if not encoded.size:
                continue
            self.changed_bytes += encoded.size // (4 + tensor.element_size()) * tensor.element_size()
            spec = {"dtype": str(tensor.dtype).removeprefix("torch."), "shape": list(tensor.shape)}
            self._compressing[name] = (self._pool.submit(sparse_delta.compress_delta, encoded), spec)

    def after_base_weights(self) -> None:
        """Collect the compressed deltas."""
        for name, (compressed, spec) in self._compressing.items():
            self._delta[name] = compressed.result()
            checksum = disk_delta.checksum(self.checksum_algorithm, self._delta[name])
            self._checksums[name] = json.dumps(spec | {"checksum": checksum})
        self._pool.shutdown()
        if self._full:
            name = sparse_delta.FULL_FILES.replace("*", f"{dist.get_rank():05d}")
            safetensors.torch.save_file(self._full, os.path.join(self._version_dir, name))

    def _reload_engines(self, *, weight_version: int) -> None:
        group = distributed_utils.get_gloo_group()
        sources: list[str | None] = [None] * dist.get_world_size(group=group)
        dist.all_gather_object(sources, self._publish_version(), group=group)
        if dist.get_rank() == 0:
            request = sglang_utils.MultiprocessingSerializer.serialize(
                sparse_delta.build_request(
                    source_dir=self.args.update_weight_disk_dir,
                    version=self._snapshot_file_version,
                    pre_read_hook=self.args.sglang.base_args["custom_pull_weights_pre_read_hook"],
                    sources=sorted({source for source in sources if source}),
                    verify=self.args.update_weight_delta_verify,
                ),
                output_str=True,
            )
            results = async_utils.wait_futures(
                [
                    async_utils.submit(
                        client.update_weights_from_tensor(
                            # SGLang indexes the list by TP rank.
                            serialized_named_tensors=[request] * gpu_count,
                            load_format=sparse_delta.LOADER,
                            selector=self._selector,
                        )
                    )
                    for client, gpu_count in zip(self.rollout_engines, self._engine_gpu_counts, strict=True)
                ]
            )
            session.check_weight_sync_results(results, is_lora=False)
        dist.barrier(group=group)
