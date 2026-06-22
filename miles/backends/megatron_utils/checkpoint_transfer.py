import torch


class _TensorViewCodec:
    """Encode tensors as (unique_storages, view_metas) and decode back.

    Many input tensors may share underlying storage (e.g. Megatron
    distributed-optimizer grad buckets). `encode` dedups by storage data_ptr —
    each unique storage is wrapped once as a uint8 tensor (no copy), plus a
    per-input view_meta record (storage_id, dtype, shape, stride,
    storage_offset). `decode` reconstructs the original views with
    `as_strided` over the storage bytes reinterpreted at the original dtype.
    """

    @staticmethod
    def encode(tensors: list[torch.Tensor]) -> tuple[list[torch.Tensor], list[dict]]:
        storage_id_by_key: dict[tuple[torch.device, int], int] = {}
        unique_storages: list[torch.Tensor] = []
        view_metas: list[dict] = []
        for t in tensors:
            storage = t.untyped_storage()
            key = (t.device, storage.data_ptr())
            if key not in storage_id_by_key:
                storage_id_by_key[key] = len(unique_storages)
                # Wrap full storage as uint8 tensor (no copy, shares memory).
                unique_storages.append(torch.tensor(storage, dtype=torch.uint8, device=t.device))
            view_metas.append(
                {
                    "storage_id": storage_id_by_key[key],
                    "dtype": t.dtype,
                    "shape": tuple(t.shape),
                    "stride": tuple(t.stride()),
                    "storage_offset": t.storage_offset(),
                }
            )
        return unique_storages, view_metas

    @staticmethod
    def decode(unique_storages: list[torch.Tensor], view_metas: list[dict]) -> list[torch.Tensor]:
        tensors: list[torch.Tensor] = []
        for vm in view_metas:
            storage_t = unique_storages[vm["storage_id"]]  # uint8 view of received storage
            # Reinterpret bytes as the original dtype, then apply stride/offset.
            dtype_view = storage_t.view(vm["dtype"])
            view = torch.as_strided(
                dtype_view,
                size=vm["shape"],
                stride=vm["stride"],
                storage_offset=vm["storage_offset"],
            )
            tensors.append(view)
        return tensors
