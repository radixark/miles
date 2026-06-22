import pytest
import torch

from miles.backends.megatron_utils.checkpoint_transfer import _TensorViewCodec


class TestTensorViewCodec:
    """Comprehensive UT for `_TensorViewCodec.encode/decode`.

    Round-trip semantics: `decode(*encode(tensors))` must produce a list of
    tensors that are *value-equal* to the inputs, while sharing storage with
    the encoded `unique_storages` (the whole point of the codec).
    """

    def test_empty_input_yields_empty_output(self):
        unique_storages, view_metas = _TensorViewCodec.encode([])
        assert unique_storages == []
        assert view_metas == []
        assert _TensorViewCodec.decode(unique_storages, view_metas) == []

    def test_single_tensor_round_trip(self):
        original = torch.arange(12, dtype=torch.float32).reshape(3, 4)

        unique_storages, view_metas = _TensorViewCodec.encode([original])

        assert len(unique_storages) == 1
        assert len(view_metas) == 1
        assert view_metas[0]["storage_id"] == 0
        assert view_metas[0]["dtype"] == torch.float32
        assert view_metas[0]["shape"] == (3, 4)
        assert view_metas[0]["stride"] == (4, 1)
        assert view_metas[0]["storage_offset"] == 0

        decoded = _TensorViewCodec.decode(unique_storages, view_metas)
        assert len(decoded) == 1
        assert torch.equal(decoded[0], original)
        assert decoded[0].dtype == original.dtype
        assert decoded[0].shape == original.shape

    def test_distinct_storages_yield_distinct_storage_ids(self):
        a = torch.arange(8, dtype=torch.float32)
        b = torch.arange(4, dtype=torch.int64)
        c = torch.zeros(5, dtype=torch.float64)

        unique_storages, view_metas = _TensorViewCodec.encode([a, b, c])

        assert len(unique_storages) == 3
        assert [vm["storage_id"] for vm in view_metas] == [0, 1, 2]

        decoded = _TensorViewCodec.decode(unique_storages, view_metas)
        assert torch.equal(decoded[0], a)
        assert torch.equal(decoded[1], b)
        assert torch.equal(decoded[2], c)

    def test_shared_storage_dedups_into_one_unique_storage(self):
        """The dedup invariant: N views over one storage produce 1 unique_storage."""
        base = torch.arange(100, dtype=torch.float32)
        view_a = base[10:20]  # offset=10, length=10
        view_b = base[20:60].view(4, 10)  # offset=20, shape=(4, 10)
        view_c = base[:5]  # offset=0, length=5

        unique_storages, view_metas = _TensorViewCodec.encode([view_a, view_b, view_c])

        assert len(unique_storages) == 1
        assert all(vm["storage_id"] == 0 for vm in view_metas)
        # storage_offset / shape / stride encode the difference
        assert view_metas[0]["storage_offset"] == 10
        assert view_metas[0]["shape"] == (10,)
        assert view_metas[1]["storage_offset"] == 20
        assert view_metas[1]["shape"] == (4, 10)
        assert view_metas[2]["storage_offset"] == 0
        assert view_metas[2]["shape"] == (5,)

        decoded = _TensorViewCodec.decode(unique_storages, view_metas)
        assert torch.equal(decoded[0], view_a)
        assert torch.equal(decoded[1], view_b)
        assert torch.equal(decoded[2], view_c)

    def test_partial_dedup_preserves_storage_id_assignment_order(self):
        """First-seen wins: storage_id increments only on a NEW storage_ptr."""
        s1 = torch.arange(10, dtype=torch.float32)  # storage A
        s2 = torch.arange(20, dtype=torch.int32)  # storage B
        s1_view = s1[2:8]  # storage A again
        s3 = torch.zeros(4, dtype=torch.float64)  # storage C
        s2_view = s2[5:10]  # storage B again

        unique_storages, view_metas = _TensorViewCodec.encode([s1, s2, s1_view, s3, s2_view])

        assert len(unique_storages) == 3
        assert [vm["storage_id"] for vm in view_metas] == [0, 1, 0, 2, 1]

    def test_dtype_preserved_across_round_trip(self):
        dtypes = [
            torch.float32,
            torch.float64,
            torch.float16,
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
            torch.uint8,
            torch.bool,
            torch.bfloat16,
        ]
        tensors = [torch.zeros(5, dtype=dt) for dt in dtypes]

        unique_storages, view_metas = _TensorViewCodec.encode(tensors)
        decoded = _TensorViewCodec.decode(unique_storages, view_metas)

        for d, original in zip(decoded, tensors, strict=True):
            assert d.dtype == original.dtype
            assert torch.equal(d, original)

    def test_non_contiguous_tensor_preserves_stride_and_values(self):
        """Transposed view: shape=(3,4), stride=(1,3) (column-major over storage)."""
        base = torch.arange(12, dtype=torch.float32).reshape(4, 3)
        transposed = base.t()  # shape=(3,4), non-contiguous
        assert not transposed.is_contiguous()

        unique_storages, view_metas = _TensorViewCodec.encode([transposed])

        assert view_metas[0]["shape"] == (3, 4)
        assert view_metas[0]["stride"] == transposed.stride()

        decoded = _TensorViewCodec.decode(unique_storages, view_metas)
        assert decoded[0].stride() == transposed.stride()
        assert torch.equal(decoded[0], transposed)

    def test_storage_offset_preserved_for_slice_view(self):
        base = torch.arange(20, dtype=torch.float32)
        sliced = base[7:15]
        assert sliced.storage_offset() == 7

        unique_storages, view_metas = _TensorViewCodec.encode([sliced])

        assert view_metas[0]["storage_offset"] == 7

        decoded = _TensorViewCodec.decode(unique_storages, view_metas)
        assert decoded[0].storage_offset() == 7
        assert torch.equal(decoded[0], sliced)

    def test_decoded_view_shares_storage_with_unique_storage(self):
        """Decoded views must alias the unique_storage (no copy)."""
        original = torch.arange(8, dtype=torch.float32)

        unique_storages, view_metas = _TensorViewCodec.encode([original])
        decoded = _TensorViewCodec.decode(unique_storages, view_metas)

        # Mutating the unique_storage (uint8 view) must propagate to decoded view.
        unique_storages[0].zero_()
        assert torch.equal(decoded[0], torch.zeros(8, dtype=torch.float32))

    def test_encode_aliases_input_storage_no_copy(self):
        """Encoded uint8 storage must alias the input tensor's storage."""
        original = torch.arange(8, dtype=torch.float32)

        unique_storages, _ = _TensorViewCodec.encode([original])

        # Mutating the input tensor must propagate to the encoded uint8 view.
        original.zero_()
        assert unique_storages[0].sum().item() == 0

    def test_multi_dtype_views_into_same_storage(self):
        """A storage can be viewed at multiple dtypes (e.g. fp32 vs int32)."""
        base = torch.arange(8, dtype=torch.float32)
        as_int = base.view(torch.int32)  # same storage, different dtype
        assert base.untyped_storage().data_ptr() == as_int.untyped_storage().data_ptr()

        unique_storages, view_metas = _TensorViewCodec.encode([base, as_int])

        assert len(unique_storages) == 1
        assert view_metas[0]["dtype"] == torch.float32
        assert view_metas[1]["dtype"] == torch.int32

        decoded = _TensorViewCodec.decode(unique_storages, view_metas)
        assert torch.equal(decoded[0], base)
        assert torch.equal(decoded[1], as_int)

    def test_round_trip_idempotent_under_repeated_encoding(self):
        """encode(decode(encode(t))) == encode(t)."""
        original = torch.arange(20, dtype=torch.float32).reshape(4, 5)

        s1, m1 = _TensorViewCodec.encode([original])
        decoded = _TensorViewCodec.decode(s1, m1)
        s2, m2 = _TensorViewCodec.encode(decoded)

        assert len(s1) == len(s2) == 1
        assert m1 == m2

    def test_round_trip_property_random_mix(self):
        """Property: for a random mix of tensors, decode round-trip is value-preserving."""
        torch.manual_seed(0)
        shared_base = torch.randn(64)
        tensors = [
            torch.randn(3, 4),
            torch.randn(7),
            shared_base[10:30],
            shared_base[5:15].view(2, 5),
            torch.randint(0, 100, (4, 4), dtype=torch.int64),
            torch.zeros(0, dtype=torch.float32),  # empty tensor
        ]

        unique_storages, view_metas = _TensorViewCodec.encode(tensors)

        assert len(view_metas) == len(tensors)
        # shared_base appears as 1 unique storage even though 2 views reference it
        assert len(unique_storages) == 5

        decoded = _TensorViewCodec.decode(unique_storages, view_metas)
        for d, original in zip(decoded, tensors, strict=True):
            assert d.dtype == original.dtype
            assert d.shape == original.shape
            assert torch.equal(d, original)

    def test_decode_after_storage_clone_preserves_values(self):
        """Simulates wire transfer: clone unique_storages (mimic NCCL recv copy)
        before decoding. Decoded views must still produce correct values, and
        be aliased to the clone (not the original)."""
        original = torch.arange(16, dtype=torch.float32).reshape(4, 4)
        sliced = original[1:3]

        unique_storages, view_metas = _TensorViewCodec.encode([original, sliced])
        cloned_storages = [u.clone() for u in unique_storages]

        decoded = _TensorViewCodec.decode(cloned_storages, view_metas)
        assert torch.equal(decoded[0], original)
        assert torch.equal(decoded[1], sliced)

        # Mutate the clone — decoded views follow the clone, not the original.
        cloned_storages[0].zero_()
        assert decoded[0].sum().item() == 0
        assert decoded[1].sum().item() == 0
        assert original.sum().item() != 0  # original untouched

    def test_same_tensor_twice_dedups_into_one_storage(self):
        """encode([t, t]) — two references to the same tensor share one storage."""
        t = torch.arange(8, dtype=torch.float32)

        unique_storages, view_metas = _TensorViewCodec.encode([t, t])

        assert len(unique_storages) == 1
        assert [vm["storage_id"] for vm in view_metas] == [0, 0]
        assert view_metas[0] == view_metas[1]

        decoded = _TensorViewCodec.decode(unique_storages, view_metas)
        assert torch.equal(decoded[0], t)
        assert torch.equal(decoded[1], t)

    def test_zero_dim_scalar_round_trip(self):
        """0-d (scalar) tensor: shape=(), stride=(), storage_offset=0."""
        scalar = torch.tensor(7.5, dtype=torch.float32)
        assert scalar.shape == ()
        assert scalar.stride() == ()

        unique_storages, view_metas = _TensorViewCodec.encode([scalar])

        assert view_metas[0]["shape"] == ()
        assert view_metas[0]["stride"] == ()

        decoded = _TensorViewCodec.decode(unique_storages, view_metas)
        assert decoded[0].shape == ()
        assert torch.equal(decoded[0], scalar)
        assert decoded[0].item() == 7.5

    def test_nn_parameter_input(self):
        """Production input via state_dict.pop_tensors() may include nn.Parameter.
        Codec uses .untyped_storage() which works on Parameter same as Tensor."""
        param = torch.nn.Parameter(torch.arange(6, dtype=torch.float32).reshape(2, 3))

        unique_storages, view_metas = _TensorViewCodec.encode([param])

        assert len(unique_storages) == 1
        assert view_metas[0]["dtype"] == torch.float32
        assert view_metas[0]["shape"] == (2, 3)

        decoded = _TensorViewCodec.decode(unique_storages, view_metas)
        assert torch.equal(decoded[0], param.data)

    def test_empty_slice_of_nonempty_storage(self):
        """Zero-length view (shape=(0,)) over a non-empty storage."""
        base = torch.arange(20, dtype=torch.float32)
        empty_view = base[5:5]
        assert empty_view.shape == (0,)
        # Empty view shares its base's storage (data_ptr is non-null).
        assert empty_view.untyped_storage().data_ptr() == base.untyped_storage().data_ptr()

        unique_storages, view_metas = _TensorViewCodec.encode([base, empty_view])

        assert len(unique_storages) == 1
        assert [vm["storage_id"] for vm in view_metas] == [0, 0]
        assert view_metas[1]["shape"] == (0,)
        assert view_metas[1]["storage_offset"] == 5

        decoded = _TensorViewCodec.decode(unique_storages, view_metas)
        assert decoded[0].shape == base.shape
        assert decoded[1].shape == (0,)
        assert torch.equal(decoded[0], base)
        assert torch.equal(decoded[1], empty_view)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
    def test_cuda_round_trip(self):
        """Production runs on CUDA — verify the codec round-trips on GPU tensors,
        including shared-storage dedup and dtype preservation."""
        base = torch.arange(64, dtype=torch.float32, device="cuda")
        view_a = base[10:30]  # shares storage with base
        view_b = base[30:50].view(4, 5)  # shares storage with base
        independent = torch.zeros(8, dtype=torch.bfloat16, device="cuda")

        unique_storages, view_metas = _TensorViewCodec.encode([base, view_a, view_b, independent])

        assert len(unique_storages) == 2
        assert all(s.device.type == "cuda" for s in unique_storages)
        assert [vm["storage_id"] for vm in view_metas] == [0, 0, 0, 1]

        decoded = _TensorViewCodec.decode(unique_storages, view_metas)
        assert all(d.device.type == "cuda" for d in decoded)
        assert torch.equal(decoded[0], base)
        assert torch.equal(decoded[1], view_a)
        assert torch.equal(decoded[2], view_b)
        assert torch.equal(decoded[3], independent)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
    def test_cuda_storage_aliasing_no_copy(self):
        """On CUDA, the encoded uint8 storage must alias the input tensor's
        device memory (no host-device or device-device copy)."""
        original = torch.arange(8, dtype=torch.float32, device="cuda")

        unique_storages, view_metas = _TensorViewCodec.encode([original])
        decoded = _TensorViewCodec.decode(unique_storages, view_metas)

        assert unique_storages[0].device.type == "cuda"
        # Mutate input — should propagate through the encoded storage to decoded view.
        original.zero_()
        assert decoded[0].sum().item() == 0

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
    def test_mixed_device_tensors_get_distinct_storages(self):
        """CPU and CUDA tensors never share a storage entry, and each decoded view keeps its device."""
        cpu_t = torch.arange(8, dtype=torch.float32)
        cuda_t = torch.arange(8, dtype=torch.float32, device="cuda")

        unique_storages, view_metas = _TensorViewCodec.encode([cpu_t, cuda_t])

        assert len(unique_storages) == 2
        assert [vm["storage_id"] for vm in view_metas] == [0, 1]
        assert unique_storages[0].device.type == "cpu"
        assert unique_storages[1].device.type == "cuda"

        decoded = _TensorViewCodec.decode(unique_storages, view_metas)
        assert decoded[0].device.type == "cpu"
        assert decoded[1].device.type == "cuda"
        assert torch.equal(decoded[0], cpu_t)
        assert torch.equal(decoded[1], cuda_t)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
    def test_mixed_device_dedup_stays_per_device(self):
        """Views sharing storage dedup within their device while the other device's tensor stays separate."""
        base = torch.arange(16, dtype=torch.float32, device="cuda")
        view = base[4:12]
        cpu_t = torch.arange(4, dtype=torch.float32)

        unique_storages, view_metas = _TensorViewCodec.encode([base, view, cpu_t])

        assert len(unique_storages) == 2
        assert [vm["storage_id"] for vm in view_metas] == [0, 0, 1]

        decoded = _TensorViewCodec.decode(unique_storages, view_metas)
        assert torch.equal(decoded[0], base)
        assert torch.equal(decoded[1], view)
        assert torch.equal(decoded[2], cpu_t)
