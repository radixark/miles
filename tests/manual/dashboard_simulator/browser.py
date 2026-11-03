"""Time the dashboard's views in headless Chromium against a running server.

Standalone (needs only ``playwright``): run with the interpreter that has it.

    python tests/manual/dashboard_simulator/browser.py --url http://127.0.0.1:7801 --dump-details <dump>
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from playwright.async_api import Page, async_playwright

_SETTLE_MS = 3000
_TIMEOUT_MS = 900_000

# a long task is main-thread work over 50 ms: the browser cannot paint or take input meanwhile
_OBSERVE_LONG_TASKS = """
performance.setResourceTimingBufferSize(1000000);  // default 250 entries: later views would record nothing
// resource entries appear only on completion: count requests still in flight
window.__inflight = 0;
const fetchOriginal = window.fetch;
window.fetch = (...args) => {
  window.__inflight += 1;
  return fetchOriginal(...args).finally(() => { window.__inflight -= 1; });
};
window.__longTasks = [];
new PerformanceObserver(list => list.getEntries().forEach(e => window.__longTasks.push(e.duration)))
    .observe({type: 'longtask', buffered: true});
"""

_SETTLE = """
async ([settleMs, timeoutMs]) => {
  const start = performance.now();
  let count = -1, stableSince = performance.now();
  while (performance.now() - start < timeoutMs) {
    await new Promise(r => setTimeout(r, 200));
    const n = performance.getEntriesByType('resource').filter(e => e.name.includes('/api/')).length;
    if (n !== count || window.__inflight > 0) { count = n; stableSince = performance.now(); }
    if (performance.now() - stableSince > settleMs) break;
  }
}
"""

_COLLECT = """
(viewStart) => {
  const api = performance.getEntriesByType('resource').filter(e => e.name.includes('/api/') && e.startTime >= viewStart);
  const tasks = window.__longTasks;
  window.__longTasks = [];
  return {
    settled_s: +((Math.max(viewStart, ...api.map(e => e.responseEnd)) - viewStart) / 1000).toFixed(2),
    long_task_total_s: +(tasks.reduce((a, b) => a + b, 0) / 1000).toFixed(2),
    long_task_max_s: +(Math.max(0, ...tasks) / 1000).toFixed(2),
    heap_mb: Math.round(performance.memory.usedJSHeapSize / 1e6),
    requests: api.map(e => `${new URL(e.name).pathname} ${(e.duration / 1000).toFixed(2)}s ${Math.round(e.encodedBodySize / 1024)}KB`),
  };
}
"""


async def _open_view(page: Page, hash_route: str) -> dict:
    view_start = await page.evaluate("performance.now()")
    await page.evaluate(f"location.hash = {json.dumps(hash_route)}")
    await page.evaluate(_SETTLE, [_SETTLE_MS, _TIMEOUT_MS])
    return await page.evaluate(_COLLECT, view_start)


def _sample_hash_with_conversation(dump_details: Path, rollout_id: int) -> str | None:
    """Route of a sample that has a conversation, read off disk: asking the
    server would warm the very caches being timed. The leaf is part of the
    route, as views_rollout.js sampleHash builds it."""
    path = dump_details / "trajectory" / f"{rollout_id}.jsonl"
    if not path.exists():  # single-turn runs record no conversations
        return None
    with open(path) as f:
        row = json.loads(f.readline())
    occurrence = f"?occurrence={row['sample_occurrence']}" if row["sample_occurrence"] else ""
    return f"#/rollout/{rollout_id}/sample/{row['sample_index']}{occurrence}"


async def time_views(url: str, dump_details: Path) -> dict:
    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch()
        page = await browser.new_page(viewport=dict(width=1600, height=1000))
        page.set_default_timeout(_TIMEOUT_MS)
        await page.add_init_script(_OBSERVE_LONG_TASKS)
        await page.goto(url + "/#/", wait_until="load")
        await page.evaluate(_SETTLE, [_SETTLE_MS, _TIMEOUT_MS])
        results = {"metrics (first load)": await page.evaluate(_COLLECT, 0)}
        results["timeline"] = await _open_view(page, "#/timeline")
        results["rollouts"] = await _open_view(page, "#/rollout/latest")
        await page.wait_for_url(lambda url: url.rsplit("/", 1)[-1].split("?")[0].isdigit())
        rollout_id = int(page.url.rsplit("/", 1)[-1].split("?")[0])
        conversation_hash = _sample_hash_with_conversation(dump_details, rollout_id)
        if conversation_hash is None:
            # no conversation: the sample view opens straight on its tokens
            first_index = await page.evaluate(
                f"fetch('/api/rollout/{rollout_id}/summary').then(r => r.json()).then(s => s.rows[0].sample_index)"
            )
            results["sample tokens"] = await _open_view(page, f"#/rollout/{rollout_id}/sample/{first_index}")
        else:
            results["sample conversation"] = await _open_view(page, conversation_hash)
            view_start = await page.evaluate("performance.now()")
            await page.get_by_role("button", name="Tokens").click()
            await page.evaluate(_SETTLE, [_SETTLE_MS, _TIMEOUT_MS])
            results["sample tokens"] = await page.evaluate(_COLLECT, view_start)
        await browser.close()
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--url", required=True)
    parser.add_argument("--dump-details", type=Path, required=True, help="the dump the server serves")
    args = parser.parse_args()
    results = asyncio.run(time_views(args.url, args.dump_details))
    for view, result in results.items():
        print(
            f"{view:22s} settled {result['settled_s']:6.2f}s  long tasks {result['long_task_total_s']:5.2f}s "
            f"(max {result['long_task_max_s']:.2f}s)  heap {result['heap_mb']} MB",
            flush=True,
        )
    print(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
