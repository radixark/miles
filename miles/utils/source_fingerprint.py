import datetime
import decimal
import hashlib
import os
import struct
import uuid
from collections.abc import Mapping
from enum import Enum
from typing import cast

import numpy
import torch
from PIL import Image


def _digest_parts(tag: bytes, *parts: bytes) -> bytes:
    digest = hashlib.sha256()
    digest.update(tag)
    for part in parts:
        digest.update(len(part).to_bytes(8, byteorder="big"))
        digest.update(part)
    return digest.digest()


def _numpy_dtype_digest(dtype: numpy.dtype[numpy.generic]) -> bytes:
    metadata = canonical_source_digest(dtype.metadata)
    if dtype.fields is not None:
        fields = []
        for name in dtype.names or ():
            field = dtype.fields[name]
            field_dtype = field[0]
            offset = field[1]
            title = field[2] if len(field) == 3 else None
            fields.append(
                _digest_parts(
                    b"field",
                    canonical_source_digest(name),
                    _numpy_dtype_digest(field_dtype),
                    canonical_source_digest(offset),
                    canonical_source_digest(title),
                )
            )
        return _digest_parts(
            b"structured-dtype",
            canonical_source_digest(dtype.itemsize),
            canonical_source_digest(dtype.isalignedstruct),
            metadata,
            *fields,
        )
    if dtype.subdtype is not None:
        base_dtype, shape = dtype.subdtype
        return _digest_parts(
            b"subarray-dtype",
            _numpy_dtype_digest(base_dtype),
            canonical_source_digest(shape),
            metadata,
        )
    return _digest_parts(b"dtype", dtype.str.encode(), metadata)


def _numpy_has_extended_precision(dtype: numpy.dtype[numpy.generic]) -> bool:
    return (dtype.kind == "f" and dtype.itemsize > 8) or (dtype.kind == "c" and dtype.itemsize > 16)


def _numpy_extended_scalar_digest(value: numpy.generic) -> bytes:
    if value.dtype.kind == "f":
        formatted = numpy.format_float_scientific(cast(float, value), unique=True, trim="k")
        return _digest_parts(b"extended-float", formatted.encode())
    if value.dtype.kind == "c":
        complex_value = cast(complex, value)
        formatted_real = numpy.format_float_scientific(complex_value.real, unique=True, trim="k")
        formatted_imag = numpy.format_float_scientific(complex_value.imag, unique=True, trim="k")
        return _digest_parts(b"extended-complex", formatted_real.encode(), formatted_imag.encode())
    raise TypeError(f"NumPy dtype {value.dtype} is not an extended floating-point type.")


def canonical_source_digest(value: object) -> bytes:
    """Return a deterministic digest for one logical source value.

    Args:
        value: A source-owned scalar or nested container. Mapping and set
            iteration order does not affect the digest.

    Returns:
        A SHA-256 digest that preserves value types and sequence order.

    Raises:
        TypeError: If the value has no stable logical representation.
    """
    if value is None:
        return _digest_parts(b"none")
    if isinstance(value, Enum):
        enum_type = f"{value.__class__.__module__}.{value.__class__.__qualname__}"
        return _digest_parts(b"enum", enum_type.encode(), canonical_source_digest(value.value))
    if isinstance(value, numpy.generic):
        if value.dtype.hasobject or value.dtype.fields is not None:
            scalar_value = canonical_source_digest(value.tolist())
        elif _numpy_has_extended_precision(value.dtype):
            scalar_value = _numpy_extended_scalar_digest(value)
        else:
            scalar_value = value.tobytes()
        return _digest_parts(b"numpy-scalar", _numpy_dtype_digest(value.dtype), scalar_value)
    if isinstance(value, bool):
        return _digest_parts(b"bool", b"1" if value else b"0")
    if isinstance(value, int):
        return _digest_parts(b"int", str(value).encode())
    if isinstance(value, float):
        return _digest_parts(b"float", struct.pack("!d", value))
    if isinstance(value, str):
        return _digest_parts(b"str", value.encode())
    if isinstance(value, (bytes, bytearray, memoryview)):
        return _digest_parts(b"bytes", bytes(value))
    if isinstance(value, decimal.Decimal):
        return _digest_parts(b"decimal", str(value).encode())
    if isinstance(value, datetime.datetime):
        return _digest_parts(b"datetime", value.isoformat().encode())
    if isinstance(value, datetime.date):
        return _digest_parts(b"date", value.isoformat().encode())
    if isinstance(value, datetime.time):
        return _digest_parts(b"time", value.isoformat().encode())
    if isinstance(value, datetime.timedelta):
        nanoseconds = getattr(value, "nanoseconds", 0)
        return _digest_parts(
            b"timedelta",
            canonical_source_digest(value.days),
            canonical_source_digest(value.seconds),
            canonical_source_digest(value.microseconds),
            canonical_source_digest(nanoseconds),
        )
    if isinstance(value, uuid.UUID):
        return _digest_parts(b"uuid", value.bytes)
    if isinstance(value, os.PathLike):
        return _digest_parts(b"path", os.fsencode(value))
    if isinstance(value, torch.Tensor):
        if value.layout != torch.strided:
            raise TypeError(f"Cannot fingerprint tensor with layout {value.layout}.")
        if value.is_quantized:
            raise TypeError("Cannot fingerprint quantized tensors.")
        tensor = value.detach().resolve_conj().resolve_neg().cpu().contiguous()
        tensor_bytes = tensor.reshape(-1).view(torch.uint8).numpy().tobytes()
        return _digest_parts(
            b"tensor",
            str(tensor.dtype).encode(),
            canonical_source_digest(tuple(tensor.shape)),
            tensor_bytes,
        )
    if isinstance(value, numpy.ma.MaskedArray):
        return _digest_parts(
            b"masked-array",
            _numpy_dtype_digest(value.dtype),
            canonical_source_digest(value.shape),
            canonical_source_digest(value.data),
            canonical_source_digest(numpy.ma.getmaskarray(value)),
            canonical_source_digest(value.fill_value),
        )
    if isinstance(value, numpy.ndarray):
        if value.dtype.hasobject or value.dtype.fields is not None or _numpy_has_extended_precision(value.dtype):
            array_bytes = canonical_source_digest(value.tolist())
        else:
            array_bytes = numpy.ascontiguousarray(value).tobytes()
        return _digest_parts(
            b"ndarray",
            _numpy_dtype_digest(value.dtype),
            canonical_source_digest(value.shape),
            array_bytes,
        )
    if isinstance(value, Image.Image):
        return _digest_parts(
            b"image",
            value.mode.encode(),
            canonical_source_digest(value.size),
            canonical_source_digest(value.format),
            value.tobytes(),
            canonical_source_digest(value.getpalette()),
            canonical_source_digest(value.info),
            value.getexif().tobytes(),
        )
    if isinstance(value, Mapping):
        entries = sorted(
            _digest_parts(b"entry", canonical_source_digest(key), canonical_source_digest(item))
            for key, item in value.items()
        )
        return _digest_parts(b"mapping", *entries)
    if isinstance(value, list):
        return _digest_parts(b"list", *(canonical_source_digest(item) for item in value))
    if isinstance(value, tuple):
        return _digest_parts(b"tuple", *(canonical_source_digest(item) for item in value))
    if isinstance(value, set):
        return _digest_parts(
            b"set",
            *(sorted(canonical_source_digest(item) for item in value)),
        )
    if isinstance(value, frozenset):
        return _digest_parts(
            b"frozenset",
            *(sorted(canonical_source_digest(item) for item in value)),
        )
    raise TypeError(
        f"Cannot fingerprint rollout source value of type {value.__class__.__module__}.{value.__class__.__qualname__}."
    )
