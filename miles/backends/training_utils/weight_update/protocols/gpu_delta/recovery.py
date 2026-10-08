"""Owner-local base-relative bytes retained for recovery and checkpoint saves."""

from dataclasses import dataclass

from miles.utils.gpu_delta.publication import PublicationWriter


@dataclass
class RecoveryPayload:
    """Compressed matrices own their pinned wire buffers; raw targets own bytes.

    No tensor export, payload gathering, hashing or filesystem work is needed
    until an engine recovery or checkpoint save requests a publication.
    """

    matrices: list
    raw: dict[str, bytes]
    codec: str
    frame_bytes: int

    @classmethod
    def encode(cls, encoder, codec, batches, raw_names, base, target):
        names, encoded = [], []
        for batch in batches:
            names.extend(batch)
            encoded.extend(encoder.encode_device([(base[name], target[name]) for name in batch]))
        finalized = encoder.finish_device(encoded)
        return cls(
            list(zip(names, finalized, strict=True)),
            {name: target[name].numpy().tobytes() for name in raw_names},
            codec,
            encoder.frame_bytes,
        )

    def write(self, directory, metadata, owner, plan, base):
        writer = PublicationWriter(directory, owner=owner, codec=self.codec, frame_bytes=self.frame_bytes, **metadata)
        try:
            for name, (frames, payload, outer, changed, _) in self.matrices:
                spec = plan[name]
                writer.add_encoded_tensor(
                    name, frames, payload, outer, changed, spec["dtype"], spec["shape"], spec["views"]
                )
            for name, value in self.raw.items():
                spec = plan[name]
                writer.add_raw_tensor(name, base[name].numpy(), value, spec["dtype"], spec["shape"], spec["views"])
            return writer.finish_shard()
        finally:
            writer.close()
