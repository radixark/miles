from array import array
from collections.abc import Sequence
from dataclasses import InitVar, dataclass, field

import torch


@dataclass(frozen=True, eq=False)
class RolloutSamplingMask:
    """One sample's sampling mask: for each response position, the token ids
    the rollout sampler could emit.

    Stored in CSR form so it stays two flat integer arrays end to end:
    ``ids`` is ``[total_support_size]`` (all supports concatenated), ``offsets``
    is ``[num_response_tokens + 1]``, and token ``t``'s support is
    ``ids[offsets[t] : offsets[t + 1]]``. That is the shape object-store
    transport needs, so no per-token nesting is rebuilt on the trainer side.
    """

    ids: InitVar[Sequence[int] | torch.Tensor]
    offsets: InitVar[Sequence[int] | torch.Tensor]
    _ids: torch.Tensor = field(init=False, repr=False)
    _offsets: torch.Tensor = field(init=False, repr=False)

    def __post_init__(self, ids: Sequence[int] | torch.Tensor, offsets: Sequence[int] | torch.Tensor):
        owned_ids = _to_owned_cpu_integer_tensor(ids, dtype=torch.int32)
        owned_offsets = _to_owned_cpu_integer_tensor(offsets, dtype=torch.long)
        if owned_offsets.numel() == 0 or owned_offsets[0] != 0 or owned_offsets[-1] != owned_ids.numel():
            raise ValueError("sampling-mask offsets must start at zero and end at the flattened id count")
        if torch.any(owned_offsets[1:] <= owned_offsets[:-1]):
            raise ValueError(
                "sampling-mask offsets must be strictly increasing: "
                "every response token needs a non-empty sampling mask"
            )
        object.__setattr__(self, "_ids", owned_ids)
        object.__setattr__(self, "_offsets", owned_offsets)

    @classmethod
    def from_mask_list(cls, mask_list: Sequence[Sequence[int]]) -> "RolloutSamplingMask":
        """Build from one mask (the allowed token ids) per response token.

        Args:
            mask_list: ragged ``[num_response_tokens][mask_size_t]``;
                ``mask_list[t]`` lists the token ids the sampler could emit at
                response position ``t``. SGLang's ``output_token_sampling_mask``
                arrives in this shape.
        """
        ids = []
        offsets = [0]
        for mask in mask_list:
            ids.extend(mask)
            offsets.append(len(ids))
        return cls(ids=ids, offsets=offsets)

    def __len__(self) -> int:
        return self._offsets.numel() - 1

    def _select_masks(self, token_indices: Sequence[int] | torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Flattened masks for the given response positions.

        Args:
            token_indices: ``[num_selected]`` response positions to read.

        Returns:
            ``(ids, lengths)`` where ``ids`` is ``[sum(lengths)]``, the selected
            masks concatenated in ``token_indices`` order, and ``lengths`` is
            ``[num_selected]``, each position's mask size.

        The returned ids may share the mask's private storage and are only for
        immediate read-only use by the scoring path.
        """
        if isinstance(token_indices, range) and token_indices.step == 1:
            if len(token_indices) == 0:
                return self._ids.new_empty(0), self._offsets.new_empty(0)
            if token_indices.start < 0 or token_indices.stop > len(self):
                raise ValueError(f"response indices must be in [0, {len(self)})")
            start, stop = token_indices.start, token_indices.stop
            lengths = self._offsets[start + 1 : stop + 1] - self._offsets[start:stop]
            return self._ids[self._offsets[start] : self._offsets[stop]], lengths

        indices = _to_cpu_integer_tensor(token_indices).to(torch.long)
        if torch.any(indices < 0) or torch.any(indices >= len(self)):
            raise ValueError(f"response indices must be in [0, {len(self)})")
        lengths = self._offsets[indices + 1] - self._offsets[indices]
        if indices.numel() == 0:
            return self._ids.new_empty(0), lengths
        run_starts = [0]
        run_starts.extend((torch.nonzero(indices[1:] != indices[:-1] + 1).flatten() + 1).tolist())
        run_starts.append(indices.numel())
        parts = [
            self._ids[self._offsets[indices[start]] : self._offsets[indices[end - 1] + 1]]
            for start, end in zip(run_starts[:-1], run_starts[1:], strict=True)
        ]
        return (parts[0] if len(parts) == 1 else torch.cat(parts)), lengths


def _to_owned_cpu_integer_tensor(
    values: Sequence[int] | torch.Tensor,
    *,
    dtype: torch.dtype,
) -> torch.Tensor:
    if isinstance(values, torch.Tensor):
        _validate_integer_tensor(values)
        return values.to(device="cpu", dtype=dtype, copy=True)

    if dtype == torch.int32 and len(values) > 0:
        storage = array("i", values)
        # frombuffer keeps this private backing array alive without copying it.
        return torch.frombuffer(storage, dtype=torch.int32)

    return torch.tensor(values, dtype=dtype, device="cpu")


def _to_cpu_integer_tensor(values: Sequence[int] | torch.Tensor) -> torch.Tensor:
    if isinstance(values, torch.Tensor):
        tensor = values.detach().cpu()
    elif len(values) == 0:
        tensor = torch.empty(0, dtype=torch.long, device="cpu")
    elif isinstance(values, range):
        tensor = torch.arange(values.start, values.stop, values.step, device="cpu")
    else:
        tensor = torch.as_tensor(values, device="cpu")
    _validate_integer_tensor(tensor)
    return tensor


def _validate_integer_tensor(tensor: torch.Tensor) -> None:
    if tensor.ndim != 1 or tensor.dtype == torch.bool or torch.is_floating_point(tensor) or torch.is_complex(tensor):
        raise ValueError("sampling-mask ids, offsets, and response indices must be one-dimensional integers")
