import functools
import os
from pathlib import Path, PurePosixPath

if os.environ.get("KIMI_EVAL_SYSTEM_DNS") == "1":
    try:
        import pyqwest
    except ImportError:
        pass
    else:
        pyqwest.HTTPTransport = functools.partial(pyqwest.HTTPTransport, use_system_dns=True)
        pyqwest.SyncHTTPTransport = functools.partial(pyqwest.SyncHTTPTransport, use_system_dns=True)


async def _stream_download_file(environment, source_path, target_path):
    stream = await environment._sandbox.files.read(source_path, format="stream", gzip=True)
    async with stream:
        with Path(target_path).open("wb") as output:
            async for chunk in stream:
                output.write(chunk)


async def _stream_upload_file(environment, source_path, target_path):
    with Path(source_path).open("rb") as source:
        await environment._sandbox.files.write(target_path, source, gzip=True)


async def _stream_upload_dir(environment, source_dir, target_dir):
    source_dir = Path(source_dir)
    for path in source_dir.rglob("*"):
        if path.is_file():
            target = str(PurePosixPath(target_dir) / path.relative_to(source_dir).as_posix())
            await _stream_upload_file(environment, path, target)


def _install_stream_transfers():
    from harbor.environments.e2b import E2BEnvironment

    E2BEnvironment._sandbox_download_file = _stream_download_file
    E2BEnvironment._sandbox_upload_file = _stream_upload_file
    E2BEnvironment._sandbox_upload_dir = _stream_upload_dir


if os.environ.get("KIMI_EVAL_STREAM_TRANSFERS") == "1":
    _install_stream_transfers()
