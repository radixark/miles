"""A cleanup transport failure must not discard the diagnostic row or its peers."""

import asyncio
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from payload_control import one_trial


class CreatedResponse:
    def raise_for_status(self) -> None:
        pass

    def json(self) -> dict:
        return {"session_id": "test"}


class FailingClient:
    async def post(self, url: str, **kwargs: object) -> CreatedResponse:
        if url.endswith("/sessions"):
            return CreatedResponse()
        raise RuntimeError("response transport failed")

    async def delete(self, url: str, **kwargs: object) -> None:
        raise RuntimeError("cleanup transport failed")


class CleanupTest(unittest.IsolatedAsyncioTestCase):
    async def test_preserves_failed_rows(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            args = SimpleNamespace(root=directory, port=33400, workers=1)
            rows = await asyncio.gather(*(
                one_trial(FailingClient(), args, i, {"request": {}}, [], "full") for i in range(2)
            ))
            self.assertEqual(len(rows), 2)
            for row in rows:
                self.assertIn("response transport failed", row["error"])
                self.assertIn("cleanup transport failed", row["delete_error"])
            self.assertEqual(len((Path(directory) / "trials-full.jsonl").read_text().splitlines()), 2)


if __name__ == "__main__":
    unittest.main()
