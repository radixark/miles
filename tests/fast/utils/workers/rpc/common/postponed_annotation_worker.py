from __future__ import annotations

from miles.utils.pydantic_utils import StrictBaseModel


class LatePayload(StrictBaseModel):
    text: str


class PostponedWorker:
    def demo_transform(self, payload: LatePayload) -> LatePayload:
        return payload
