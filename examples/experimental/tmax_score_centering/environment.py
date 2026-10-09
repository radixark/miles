"""Preserve shell timeouts and background commands in TMax E2B trials."""

import asyncio
import shlex
from collections import defaultdict

from harbor.environments.e2b import E2BEnvironment

# The single rollout worker shares template builds across concurrent Harbor trials.
_TEMPLATE_LOCKS: dict[str, asyncio.Lock] = defaultdict(asyncio.Lock)


class TMaxE2BEnvironment(E2BEnvironment):
    async def start(self, force_build: bool):
        # Harbor's existence check and build race (radixark/AgentENV#74).
        async with _TEMPLATE_LOCKS[self._template_name]:
            if force_build:
                await super().start(force_build=True)
                return
            if self._prebuilt_template_id is None and not await self._does_template_exist():
                await self._create_template()
        await super().start(force_build=False)

    async def exec(self, command, cwd=None, env=None, timeout_sec=None, user=None):
        # E2B bounds the response stream; GNU timeout also terminates the process group.
        request_timeout = timeout_sec
        command = f"bash -c {shlex.quote(command)}"
        if timeout_sec:
            command = f"timeout --signal=TERM --kill-after=10 {shlex.quote(str(timeout_sec))} {command}"
            request_timeout = timeout_sec + 30
        # Regular files prevent background services from keeping the RPC's pipes open.
        script = (
            "capture_dir=$(mktemp -d /tmp/tmax-exec.XXXXXXXX) || exit 1\n"
            f'{command} </dev/null >"$capture_dir/stdout" 2>"$capture_dir/stderr"\n'
            "command_status=$?\n"
            'cat "$capture_dir/stdout"\n'
            'cat "$capture_dir/stderr" >&2\n'
            'rm -rf "$capture_dir"\n'
            'exit "$command_status"'
        )
        command = f"bash -c {shlex.quote(script)}"
        return await super().exec(command=command, cwd=cwd, env=env, timeout_sec=request_timeout, user=user)
