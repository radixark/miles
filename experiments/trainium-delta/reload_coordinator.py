"""Proposed H200 integration helper; not installed into Miles or called on live engines."""
import asyncio
async def reload_paused_pool(clients, model_path, version, concurrency=1):
    """Caller has already paused/flushed ALL workers. Failure leaves them paused.

    Never auto-resume mixed versions. A resume transport failure is also reported;
    caller must reconcile the pool before generating again.
    """
    if not clients or concurrency < 1:
        raise ValueError('Nonempty clients and positive concurrency required')
    semaphore=asyncio.Semaphore(concurrency)
    async def reload(client):
        async with semaphore:
            result=await client.update_weights_from_disk(model_path,weight_version=str(version))
            if not isinstance(result,dict) or result.get('success') is not True:
                raise RuntimeError(f'Reload did not report success: {result!r}')
    results=await asyncio.gather(*(reload(c) for c in clients),return_exceptions=True)
    failures=[r for r in results if isinstance(r,BaseException)]
    if failures:
        raise RuntimeError('Reload failed; pool remains paused') from failures[0]
    reported=await asyncio.gather(*(c.get_weight_version() for c in clients))
    if any(str(v)!=str(version) for v in reported):
        raise RuntimeError(f'Version mismatch; pool remains paused: {reported!r}')
    resumed=await asyncio.gather(*(c.continue_generation() for c in clients),return_exceptions=True)
    failures=[r for r in resumed if isinstance(r,BaseException)]
    if failures:
        raise RuntimeError('Resume incomplete; reconcile pool before generation') from failures[0]
    return {'version':str(version),'replicas':len(clients)}
