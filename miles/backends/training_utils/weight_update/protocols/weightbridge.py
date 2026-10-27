"""WeightBridge weight transfer: WeightBridge owns the whole per-update control flow.

Each SGLang engine started with ``--enable-weightbridge-receiver`` spawns a WeightBridge
coordinator and builds a receiver on every TP rank at startup, then polls it from its
scheduler loop. The trainer connects once through the coordinators; after that an update
is just ``send_weights()`` on every trainer rank. The receivers notice the trainer's data
flags, pause (drain and clear requests), load, flush the cache, bump the weight version and
resume on their own, so an update sends no Ray or HTTP request to the engines. The trainer's weight
version rides the update as WeightBridge metadata, so engines label it exactly as the broadcast path does.

Each trainer rank builds a ``SenderAdapter`` whose LoadSpec WeightBridge infers by probing
Megatron-Bridge's HF->Megatron import over the live Megatron parameters.
"""

import logging
import time
from argparse import Namespace
from collections.abc import Mapping, Sequence
from urllib.parse import urlsplit

import torch
import torch.distributed as dist

from miles.backends.sglang_utils.sglang_api_client import SGLangApiClient
from miles.backends.training_utils.parallel import ParallelState
from miles.backends.training_utils.weight_update.hf_weight_iterator import WeightUpdatePlacement
from miles.backends.training_utils.weight_update.protocol import WeightTransferProtocol
from miles.backends.training_utils.weight_update.rollout_cell_updater import create_rollout_cell_updaters
from miles.utils.distributed_utils import get_gloo_group
from miles.utils.misc import get_free_port

logger = logging.getLogger(__name__)


class UpdateWeightWeightBridge(WeightTransferProtocol):
    streams_buckets = False
    use_weight_update_session = False

    def __init__(self, args: Namespace) -> None:
        super().__init__(args)
        self.group_name = "weightbridge"
        self._model: Sequence[torch.nn.Module] | None = None
        self._adapter = None
        self._connected_urls: tuple[str, ...] | None = None
        self._epoch = 0
        self._logged_off_gpu = False

    def bind_trainer_model(self, model: Sequence[torch.nn.Module]) -> None:
        self._model = model

    def connect(
        self,
        rollout_engines: Sequence[SGLangApiClient],
        engine_gpu_counts: Sequence[int] | None,
        engine_gpu_offsets: Sequence[int] | None,
        engine_cell_ids: Sequence[str],
        parallel_state: ParallelState,
        placement: WeightUpdatePlacement,
        selector: str,
    ) -> None:
        """Build this rank's sender and run WeightBridge's one-time connect through the coordinators."""
        self.rollout_engines = rollout_engines
        self.cell_updaters_of_cell_id = create_rollout_cell_updaters(self.args, rollout_engines, engine_cell_ids)
        # Every trainer rank owns a shard of the weights, so every rank sends.
        self.is_sender = True
        receiver_urls = tuple(_coordinator_url(engine.server_url) for engine in rollout_engines)
        if self._connected_urls is not None:
            if receiver_urls != self._connected_urls:
                raise NotImplementedError(
                    "WeightBridge reconnect to a changed rollout engine set is not supported yet"
                )
            return

        start = time.monotonic()
        self._adapter = self._build_sender_adapter(list(receiver_urls))
        built = time.monotonic()
        dist.barrier(group=get_gloo_group())
        # The engines' scheduler loops are already polling, which is what drives their side of connect.
        self._adapter.connect()
        dist.barrier(group=get_gloo_group())
        self._connected_urls = receiver_urls
        logger.info(
            f"[weightbridge] connected {dist.get_world_size()} senders to {len(receiver_urls)} engines: "
            f"sender setup {built - start:.1f}s, connect {time.monotonic() - built:.1f}s"
        )

    def send_bucket(self, bucket: list[tuple[str, torch.Tensor]]) -> None:
        raise AssertionError("WeightBridge reads live parameters; the updater must not stream buckets to it")

    def send_live_weights(self, weight_version: int) -> None:
        # EWTT endpoint: wall clock right before send_weights() on every rank.
        logger.info(
            f"[weightbridge] send_start epoch={self._epoch} trainer_weight_version={weight_version} t_ns={time.time_ns()}"
        )
        # The engines label this update with the trainer's version, as the broadcast path's session does.
        event = self._adapter.send_weights(metadata={"weight_version": weight_version})
        if event is not None:
            event.synchronize()
        if not self.args.weightbridge_async_delivery:
            # Sync semantics: the next rollout must run on these weights, so wait until every receiver loaded them.
            self._adapter.wait_send_complete()
        self._epoch += 1
        self._adapter.flush_profile_outputs()

    def _build_sender_adapter(self, receiver_urls: list[str]):
        from wbridge.backend.sender import SenderArgs
        from wbridge.frontend.adapters import AdapterContext, SenderAdapter
        from wbridge.utils.distributed import get_local_ip
        from wbridge.utils.specgen import hf_weights_from_checkpoint

        assert self._model is not None, "bind_trainer_model() must run before connect()"
        rendezvous = [get_local_ip(), get_free_port(start_port=20000)] if dist.get_rank() == 0 else [None, None]
        dist.broadcast_object_list(rendezvous, src=0, group=get_gloo_group())
        master_addr, master_port = rendezvous

        hf_weights, hf_shapes = hf_weights_from_checkpoint(self.args.hf_checkpoint)
        ctx = AdapterContext(
            hf_weights=hf_weights,
            hf_shapes=hf_shapes,
            wksd_factory=self._worker_state_dict,
            load_weights=self._megatron_load_weights(),
            rank=dist.get_rank(),
        )
        sender_args = SenderArgs(
            world_size=dist.get_world_size(),
            receiver_urls=receiver_urls,
            master_addr=master_addr,
            master_port=master_port,
            protocol=self.args.weightbridge_protocol,
        )
        # Probing overwrites every parameter and restores it from the HF checkpoint; keep the
        # trainer's live values, which differ from the checkpoint after any training or resume.
        with torch.no_grad(), _preserve_tensors(self._worker_state_dict()):
            return SenderAdapter(ctx, sender_args)

    def _worker_state_dict(self) -> dict[str, torch.Tensor]:
        params = {
            f"{chunk_index}.{name}": param.data
            for chunk_index, chunk in enumerate(self._model)
            for name, param in chunk.named_parameters()
        }
        # miles' model-companion counters are CPU "parameters", not model weights; WeightBridge needs CUDA tensors.
        off_gpu = {name: str(tensor.device) for name, tensor in params.items() if not tensor.is_cuda}
        if off_gpu and not self._logged_off_gpu:
            self._logged_off_gpu = True
            logger.warning(
                f"[weightbridge] {len(off_gpu)} trainer parameters are not on CUDA and are not sent: {off_gpu}"
            )
        return {name: tensor for name, tensor in params.items() if tensor.is_cuda}

    def _megatron_load_weights(self):
        """Megatron-Bridge's HF->Megatron import, adapted to read WeightBridge's tensor factories."""
        # Local imports: Megatron-Bridge needs a GPU driver, so the module must import without it on CPU hosts.
        from megatron.bridge import AutoBridge

        from miles.utils import megatron_bridge_utils

        bridge = AutoBridge.from_hf_pretrained(self.args.hf_checkpoint, trust_remote_code=True)
        model = self._model

        def load_weights(hf_weights):
            pretrained = _FactoryBackedPretrained(bridge.hf_pretrained, hf_weights)
            with megatron_bridge_utils.patch_megatron_model(model):
                bridge._model_bridge.load_weights_hf_to_megatron(pretrained, list(model))

        return load_weights


def _coordinator_url(server_url: str) -> str:
    from wbridge.backend.control_channel import coordinator_tcp_port

    parts = urlsplit(server_url)
    return f"tcp://{parts.hostname}:{coordinator_tcp_port(parts.port)}"


class _FactoryBackedState(Mapping):
    """HF state whose tensors come from WeightBridge factories; ``source`` keeps the checkpoint key order."""

    def __init__(self, hf_weights, source) -> None:
        self._hf_weights = hf_weights
        self.source = source

    def __getitem__(self, name: str) -> torch.Tensor:
        return self._hf_weights[name]()

    def __iter__(self):
        return iter(self._hf_weights)

    def __len__(self) -> int:
        return len(self._hf_weights)


class _FactoryBackedPretrained:
    def __init__(self, pretrained, hf_weights) -> None:
        self._pretrained = pretrained
        self.state = _FactoryBackedState(hf_weights, pretrained.state.source)

    def __getattr__(self, name: str):
        return getattr(self._pretrained, name)  # config-access-exempt: delegates to the wrapped HF pretrained


class _preserve_tensors:
    def __init__(self, tensors: Mapping[str, torch.Tensor]) -> None:
        self._tensors = tensors
        self._saved: dict[str, torch.Tensor] = {}

    def __enter__(self):
        self._saved = {name: tensor.detach().to("cpu", copy=True) for name, tensor in self._tensors.items()}
        return self

    def __exit__(self, *exc_info):
        for name, tensor in self._tensors.items():
            tensor.copy_(self._saved[name])
        torch.cuda.synchronize()
        self._saved.clear()
        return False
