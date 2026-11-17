"""GPU-delta bindings for SGLang's DeepSeek-V2 model implementation (CUDA receiver only).

Mirrors ``sglang/srt/models/deepseek_v2.py`` and its ``DeepseekV2WeightLoaderMixin``:
fused gate/up and unequal Q/KV-A projections, the DSA indexer fusion, DCP-replicated
MLA consumers and the BF16 ``w_kc``/``w_vc`` caches. Admitted architectures are listed
only in ``models/__init__.py``. The MLA helpers also serve mappings of models whose
attention subclasses ``DeepseekV2AttentionMLA``.
"""

from __future__ import annotations

from dataclasses import replace
from typing import NamedTuple

import torch

from miles.backends.sglang_utils.gpu_delta.bindings import (
    ConsumerSnapshot,
    DerivedImage,
    ParameterBindings,
    _full_slices,
    _tensor_identity,
)
from miles.backends.sglang_utils.gpu_delta.byte_views import (
    DestinationSegment,
    _byte_view,
    _DtypeMismatch,
    _storage_key,
)


class DcpReplica(NamedTuple):
    """One consumer image a DCP group gathers from the MLA projection shards of its ranks."""

    image: torch.Tensor
    rows_per_head: int  # canonical projection rows of one head
    # Rows [start, stop) of each head the image holds as [heads, rows, cols]; None: whole heads stacked as rows.
    head_rows: tuple[int, int] | None = None
    # The image holds [heads, cols, rows] instead (SGLang main's gathered V image w_vc_qrep).
    transposed: bool = False


def _replica_segment(binding, replica):
    """The canonical rows one gathered image covers and its segment.

    A DCP group is consecutive attention-TP ranks, so its heads start at the local rows rounded
    down to the group's span.
    """
    image, per_head = replica.image, replica.rows_per_head
    heads = image.shape[0] // per_head if replica.head_rows is None else image.shape[0]
    span = heads * per_head
    start = binding.slices[0][0] // span * span
    coverage = [[start, start + span], [0, binding.shape[1]]]
    if replica.head_rows is None and not replica.transposed:
        source_shape = None
        slices = tuple(tuple(axis) for axis in coverage)
        expected = (span, binding.shape[1])
    elif replica.head_rows is not None:
        first, stop = replica.head_rows
        source_shape = (binding.shape[0] // per_head, per_head, binding.shape[1])
        slices = ((start // per_head, start // per_head + heads), (first, stop), (0, binding.shape[1]))
        if replica.transposed:
            expected = (heads, binding.shape[1], stop - first)
        else:
            expected = (heads, stop - first, binding.shape[1])
    else:
        raise ValueError("a transposed DCP replica holds a row range of each head")
    if tuple(image.shape) != expected or start + span > binding.shape[0] or binding.shape[0] % per_head:
        raise ValueError("unsupported DCP replicated projection geometry")
    destination = _byte_view(image)
    if replica.transposed:
        # [heads, cols, rows] storage read in the canonical [heads, rows, cols] byte order.
        destination = destination.unflatten(2, (expected[2], image.element_size())).permute(0, 2, 1, 3)
    return coverage, DestinationSegment(slices, (destination,), source_shape)


def _local_rows_alias(binding, replica, coverage):
    """Whether the local projection is a view of its rows in this gathered image, from live storage.

    The companion tree gathers separate copies. SGLang main re-points the local ``q_b_proj`` rows at
    their rows of the gathered ``q_b_proj_qrep_weight``, so those bytes take the replica's XOR only.
    """
    (local,) = binding.storage
    image = replica.image
    if _storage_key(local) != _storage_key(image):
        return False
    elif replica.head_rows is None and _tensor_identity(local) == _tensor_identity(
        image[binding.slices[0][0] - coverage[0][0] : binding.slices[0][1] - coverage[0][0]]
    ):
        return True
    else:
        raise ValueError("unsupported DCP replicated projection alias")


def with_dcp_replicas(binding, replicas):
    """Apply the same canonical mask to a DCP group's gathered MLA consumers.

    The decode payload remains complete. A logical head axis lets one affine
    descriptor select the rows of every gathered head, without per-head segments,
    temporary weights, or an all-gather while paused. When the local projection is
    a view of a gathered image, that image carries the local rows, so the local
    segment is dropped and they are XORed once.
    """
    segments, coverage, aliased = [], None, False
    for replica in replicas:
        span, segment = _replica_segment(binding, replica)
        if coverage is not None and span != coverage:
            raise ValueError("unsupported DCP replicated projection geometry")
        coverage = span
        aliased |= _local_rows_alias(binding, replica, span)
        segments.append(segment)
    local = ()
    if not aliased:
        local = binding.storage
        segments.insert(0, DestinationSegment(tuple(tuple(axis) for axis in binding.slices), binding.destinations))
    return replace(
        binding,
        slices=coverage,
        storage=(*local, *(replica.image for replica in replicas)),
        destinations=(),
        segments=tuple(segments),
    )


def fused_qkv_a_view(fused, stem, q_rows, kv_rows):
    """The rows of ``fused_qkv_a_proj_with_mqa`` that ``DeepseekV2WeightLoaderMixin`` fills from one tensor.

    The loader concatenates the ``q_a_proj`` rows above the ``kv_a_proj_with_mqa`` rows. The caller
    passes their row counts (the Q LoRA rank, and the KV LoRA rank plus the RoPE head size) from its
    model's own configuration.
    """
    if stem == "q_a_proj":
        start, rows = 0, q_rows
    elif stem == "kv_a_proj_with_mqa":
        start, rows = q_rows, kv_rows
    else:
        raise ValueError(f"unclassified fused Q/KV-A projection: {stem}")
    return fused.narrow(0, start, rows)


def _dcp_layout(attn):
    """Which SGLang tree laid out this MLA layer's DCP-replicated consumers, from the fields its class declares.

    Both trees declare the gathered K image ``w_kc_qrep`` on every MLA layer (None until
    ``--dcp-replicate-q-proj`` gathers it). SGLang main also declares the gathered, transposed V image
    ``w_vc_qrep`` and makes the local ``q_b_proj`` rows a view of ``q_b_proj_qrep_weight``; the
    companion tree declares no gathered V image.
    """
    key = hasattr(attn, "w_kc_qrep")  # config-access-exempt: Both SGLang trees declare the gathered K image.
    value = hasattr(attn, "w_vc_qrep")  # config-access-exempt: Only SGLang main declares the gathered V image.
    if key and value:
        return "main"
    elif key:
        return "companion"
    else:
        raise ValueError(f"unsupported DCP replicated projection layout: {type(attn).__name__} declares no w_kc_qrep")


def _kv_replicas(attn):
    """The consumer snapshot and the DCP-gathered K (and V) images of one MLA layer's kv_b_proj."""
    layout = _dcp_layout(attn)
    if layout == "main":
        value = attn.w_vc_qrep

        def read():
            return (
                attn.w_kc,
                attn.w_vc,
                attn.w_kc_qrep,
                attn.w_vc_qrep,
                attn.q_b_proj_qrep_weight,
                attn.q_b_proj_qrep,
            )

    elif layout == "companion":
        value = None

        def read():
            return (attn.w_kc, attn.w_vc, attn.w_kc_qrep, attn.q_b_proj_qrep_weight, attn.q_b_proj_qrep)

    else:
        raise ValueError(f"unsupported DCP replicated projection layout: {layout}")
    rows_per_head, nope = attn.qk_nope_head_dim + attn.v_head_dim, attn.qk_nope_head_dim
    replicas = []
    if attn.w_kc_qrep is not None:
        replicas.append(DcpReplica(attn.w_kc_qrep, rows_per_head, (0, nope)))
    if value is not None:
        if attn.w_kc_qrep is None:
            raise ValueError("unsupported DCP replicated projection layout: a gathered V image without its K image")
        replicas.append(DcpReplica(value, rows_per_head, (nope, rows_per_head), transposed=True))
    return ConsumerSnapshot(read), tuple(replicas)


class MlaKvLayer(NamedTuple):
    """An MLA layer whose ``kv_b_proj`` a binding covers, for the mapping to record."""

    prefix: str
    attn: torch.nn.Module  # its mla_derived_images source
    snapshot: ConsumerSnapshot  # pins its absorbed and DCP-gathered consumers


def mla_projection_binding(modules, name, target_name, binding):
    """Extend an MLA layer's ``q_b_proj`` and ``kv_b_proj`` bindings to the consumers derived from them.

    A Q projection (``q_b_proj``, or ``q_proj`` without a Q LoRA) also XORs its DCP-gathered image.
    ``kv_b_proj`` XORs the gathered K rows (and, on SGLang main, the gathered transposed V rows).
    Returns ``(binding, kv_layer)``: for ``kv_b_proj`` an ``MlaKvLayer`` the caller records in its
    consumers and for ``mla_derived_images``, otherwise None. Any other binding is returned unchanged.
    """
    if name.endswith((".q_b_proj.weight", ".q_proj.weight")):
        attn = modules[target_name.rsplit(".", 2)[0]]
        if attn.q_b_proj_qrep is not None:
            raise ValueError("GPU delta requires unquantized DCP Q replication")
        if attn.q_b_proj_qrep_weight is not None:
            binding = with_dcp_replicas(binding, (DcpReplica(attn.q_b_proj_qrep_weight, attn.qk_head_dim),))
        return binding, None
    elif ".kv_b_proj.weight" in name:
        prefix = target_name.rsplit(".kv_b_proj.weight", 1)[0]
        attn = modules[prefix]
        snapshot, replicas = _kv_replicas(attn)
        if replicas:
            binding = with_dcp_replicas(binding, replicas)
        return binding, MlaKvLayer(prefix, attn, snapshot)
    else:
        return binding, None


def mla_derived_images(prefix, attn):
    """The BF16 ``w_kc``/``w_vc`` absorbed caches, refreshed from the layer's local ``kv_b_proj``.

    With one head per rank, the loader's ``w_vc.contiguous().transpose(1, 2)`` is the identical view
    of ``kv_b_proj`` itself (a size-1 dimension never forces a copy). That cache is the bound
    canonical storage, so it gets no copy.
    """
    if attn.kv_b_proj.weight.dtype != torch.bfloat16:
        raise ValueError("MLA delta cache refresh currently requires BF16 canonical weights")

    key, value = attn.kv_b_proj.weight.unflatten(0, (-1, attn.qk_nope_head_dim + attn.v_head_dim)).split(
        [attn.qk_nope_head_dim, attn.v_head_dim], dim=1
    )
    return tuple(
        DerivedImage(f"{prefix}.{name}", cache, source)
        for name, cache, source in (("w_kc", attn.w_kc, key), ("w_vc", attn.w_vc, value.transpose(1, 2)))
        if _tensor_identity(cache) != _tensor_identity(source)
    )


class DeepseekV2Mapping:
    """DeepSeek MLA/DSA family, including GLM's shared runtime implementation."""

    # Canonical aliases used by DeepseekV2WeightLoaderMixin. The ordinary loader
    # owns its mappings locally; admission resolves only existing live targets.
    stacked_params = (
        ("gate_up_proj", "gate_proj", 0),
        ("gate_up_proj", "up_proj", 1),
        ("fused_qkv_a_proj_with_mqa", "q_a_proj", 0),
        ("fused_qkv_a_proj_with_mqa", "kv_a_proj_with_mqa", 1),
    )

    def __init__(self, model, shadows):
        # The DeepSeek-V2 loader serves every canonical tensor losslessly, so no binding uses shadows.
        self.parameters = ParameterBindings(model, self.stacked_params, shadows)
        self._config = model.config
        # The loader serves a fused shared expert from routed-expert slot n_routed_experts (SGLang fuses one).
        self._fused_shared_experts = model.num_fused_shared_experts > 0
        self._mla_layers = {}
        self.derived = []
        self.consumers = []

    def bind(self, name, meta):
        target_name = name
        if self._fused_shared_experts and "mlp.shared_experts" in name:
            target_name = name.replace("mlp.shared_experts", f"mlp.experts.{self._config.n_routed_experts}")
        fused = self._fused_rows(meta, target_name)
        if fused is None:
            binding = self.parameters.bind(name, meta, target_name)
        else:
            # A row view of a fused parameter, declared in its alias group.
            target_name, view = fused
            module = self.parameters.module(target_name.rsplit(".", 1)[0])
            binding = self.parameters.bind_view(name, meta, module, view, _full_slices(meta["shape"]))
            self.parameters.aliases.declare(name, target_name, binding, self.parameters.params[target_name])
        binding, kv_layer = mla_projection_binding(self.parameters.modules, name, target_name, binding)
        if kv_layer is not None:
            self._mla_layers[kv_layer.prefix] = kv_layer.attn
            self.consumers.append(kv_layer.snapshot)
        return binding

    def finish(self):
        for prefix, attn in self._mla_layers.items():
            self.derived.extend(mla_derived_images(prefix, attn))

    def _fused_rows(self, meta, target_name):
        """``(fused parameter name, rows)`` the loader fills from ``target_name``; None for any other tensor.

        ``DeepseekV2WeightLoaderMixin`` concatenates ``q_a_proj`` above ``kv_a_proj_with_mqa`` and the DSA
        indexer's ``wk`` above its ``weights_proj``.
        """
        params = self.parameters.params
        if target_name in params or ".experts." in target_name:
            return None
        for param_stem, source_stem, _ in self.stacked_params:
            candidate = target_name.replace(source_stem, param_stem)
            if param_stem == "fused_qkv_a_proj_with_mqa" and source_stem in target_name and candidate in params:
                config = self._config
                view = fused_qkv_a_view(
                    params[candidate],
                    source_stem,
                    q_rows=config.q_lora_rank,
                    kv_rows=config.kv_lora_rank + config.qk_rope_head_dim,
                )
                return candidate, view
        for source, at_end in (("wk", False), ("weights_proj", True)):
            marker = f".indexer.{source}.weight"
            candidate = target_name.removesuffix(marker) + ".indexer.wk_weights_proj.weight"
            if target_name.endswith(marker) and candidate in params:
                if meta["dtype"] != "BF16":
                    # The loader dequantizes other checkpoint dtypes into this BF16 parameter.
                    raise _DtypeMismatch(f"fused DSA indexer requires BF16 canonical weights, got {meta['dtype']}")
                fused, rows = params[candidate], meta["shape"][0]
                return candidate, fused[-rows:] if at_end else fused[:rows]
        return None
