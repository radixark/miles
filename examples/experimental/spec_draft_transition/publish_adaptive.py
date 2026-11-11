"""Publish completed devbox results without copying GitHub credentials there.

Runs on the Mac in tmux, using only the standard library and small HTTP reads.
Refuses to overwrite an intervening edit to the existing comparison gist.
"""

import hashlib
import json
import os
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path


GIST = '0b501bfbfda7c55383209e3ecb1969ae'
RESULTS = 'http://100.107.151.102:19089/report/'


def github_token() -> str:
    token = os.environ.get('GH_TOKEN') or os.environ.get('GITHUB_TOKEN')
    if token:
        return token
    result = subprocess.run(['git','credential','fill'],input='protocol=https\nhost=github.com\n\n',
                            env={**os.environ,'GIT_TERMINAL_PROMPT':'0'},
                            text=True,capture_output=True,check=False,timeout=20)
    fields=dict(line.split('=',1) for line in result.stdout.splitlines() if '=' in line)
    if result.returncode==0 and fields.get('password'):
        return fields['password']
    return subprocess.check_output(['gh','auth','token'],text=True,timeout=20).strip()


def github(payload: dict | None = None) -> dict:
    request=urllib.request.Request('https://api.github.com/gists/'+GIST,
        data=None if payload is None else json.dumps(payload).encode(),
        headers={'Authorization':'Bearer '+github_token(),'Accept':'application/vnd.github+json',
                 'Content-Type':'application/json','X-GitHub-Api-Version':'2022-11-28'},
        method='GET' if payload is None else 'PATCH')
    with urllib.request.urlopen(request,timeout=60) as response:
        return json.load(response)


def download(name: str) -> str:
    with urllib.request.urlopen(RESULTS+name,timeout=30) as response:
        return response.read().decode()


def main() -> None:
    before=github()
    assert before['owner']['login']=='Shi-Dong' and '00-results.md' in before['files']
    lease=before['history'][0]['version']
    Path('/tmp/b90205ad-adaptive-gist-before.json').write_text(json.dumps(before))
    print('WAITING_FOR_COMPLETE_COMPARISON',before['html_url'],'revision',lease,flush=True)
    deadline=time.monotonic()+48*3600
    while True:
        try:
            marker=json.loads(download('ANALYSIS_COMPLETE'))
            break
        except (urllib.error.URLError, TimeoutError):
            if time.monotonic()>deadline:
                raise RuntimeError('Comparison did not complete within 48 hours')
            time.sleep(15)
    assert marker['models']==150 and marker['adaptive_models']==30 and marker['all_switches_audited']
    summary=json.loads(download('adaptive-summary.json'))
    assert len(summary['means'])==15 and len(summary['switches'])==30
    assert summary['resamples']==20000 and summary['error_bars']=='one bootstrap standard error'
    content={name:download(name) for name in ['gist-results.md','accepted-prefix.svg','training-curves.svg']}
    version=hashlib.sha256(content['accepted-prefix.svg'].encode()).hexdigest()[:12]
    text=content.pop('gist-results.md')
    for name in ['accepted-prefix.svg','training-curves.svg']:
        assert '<svg ' in content[name]
        text=text.replace(']('+name+')','](https://gist.githubusercontent.com/Shi-Dong/'+GIST+'/raw/'+name+'?adaptive='+version+')')
    content['00-results.md']=text
    latest=github()
    if latest['history'][0]['version']!=lease:
        raise RuntimeError('Gist changed during training; completed report is ready but publication deferred to preserve that edit')
    result=github({'description':'Qwen3 drafter distillation: CE, L1 and three transition/mixture objectives',
                   'files':{name:{'content':value} for name,value in content.items()}})
    verify=github()
    for name,value in content.items():
        assert verify['files'][name]['content']==value,name
    receipt={'url':result['html_url'],'revision':result['history'][0]['version'],
             'means':summary['means'],'switches':summary['switches']}
    Path('/tmp/b90205ad-adaptive-gist-published.json').write_text(json.dumps(receipt,indent=2))
    print('ADAPTIVE_GIST_PUBLISHED',json.dumps(receipt),flush=True)


if __name__=='__main__':
    main()
