"""Release a prepared extension only after baseline timing and preflights finish."""

import json
import time
import urllib.request
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from tap import Tap


class Args(Tap):
    root: str
    control_url: str


def main(args: Args) -> None:
    root=Path(args.root)
    manifest=json.loads((root/'manifest.json').read_text())
    baseline=Path(manifest['baseline_root'])
    source=(root/'repo/SOURCE_REVISION').read_text().strip()
    previous=None
    while True:
        with urllib.request.urlopen(args.control_url+'/state',timeout=30) as response:
            state=json.load(response)
        assert state['source_revision']==source
        if (root/'control/matrix-released.json').exists():
            return
        ready=((baseline/'report/reference-throughput.json').exists(),
               (baseline/'report/ANALYSIS_COMPLETE').exists(),
               (root/'report/preflight-ready.json').exists(),len(state['generation'])==40)
        if ready!=previous:
            print('WAITING_FOR_BASELINE_TIMING_ANALYSIS_PREFLIGHT_SHARDS',ready,flush=True)
            previous=ready
        if all(ready):
            break
        time.sleep(15)
    timing=json.loads((baseline/'report/reference-throughput.json').read_text())
    assert len(timing)==27
    assert all(row['source_revision']==manifest['baseline_source_revision'] for row in timing)
    preflight=json.loads((root/'report/preflight-ready.json').read_text())
    assert preflight['source_revision']==source and len(preflight['checks'])==2
    assert all((root/'runs'/check['run_id']/'complete.json').exists() for check in preflight['checks'])
    assert state['claims_held'] and all(job['status']=='pending' for job in state['jobs'])
    request=urllib.request.Request(args.control_url+'/hold',data=b'{"held":false}',
                                   headers={'Content-Type':'application/json'})
    with urllib.request.urlopen(request,timeout=30) as response:
        assert json.load(response)['ok']
    (root/'control/matrix-released.json').write_text(json.dumps({
        'source_revision':source,'baseline_reference_records':27,
        'time_pt':datetime.now(ZoneInfo('America/Los_Angeles')).isoformat(),
        'preflight':preflight},indent=2))
    print('FIXED_MIXTURE_MATRIX_RELEASED',manifest['experiment_id'],flush=True)


if __name__=='__main__':
    main(Args().parse_args())
