"""Own the driver-side engine update window."""

from contextlib import asynccontextmanager


@asynccontextmanager
async def update_weight_window(inference_controller):
    info = await inference_controller.start_update_weights()
    try:
        yield info
    finally:
        await inference_controller.end_update_weights()
