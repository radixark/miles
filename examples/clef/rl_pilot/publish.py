"""Publish a validated pilot to S3 and verify every uploaded object."""

import hashlib
import json
import zipfile
from pathlib import Path

import boto3
from tap import Tap


class Args(Tap):
    data: Path
    bucket: str = 'radixark-data'
    prefix: str = 'shi/decision/clef-rl-pilot-2048-v1'


def publish(args: Args) -> None:
    root = args.data
    assert (root/'validation-report.json').exists()
    api_key = Path('/home/ubuntu/openai.key').read_bytes().strip()
    package = root/'provenance.zip'
    with zipfile.ZipFile(package,'w',zipfile.ZIP_DEFLATED) as archive:
        for directory in ['accepted','rejected','provisional']:
            for p in sorted((root/directory).glob('*.json')):
                if api_key in p.read_bytes():
                    raise ValueError('credential found in artifact')
                archive.write(p,p.relative_to(root))
        for name in ['generate.py','validate.py','publish.py','probe.py','pyproject.toml','uv.lock','README.md']:
            p=Path(__file__).parent/name
            if api_key in p.read_bytes():
                raise ValueError('credential found in source')
            archive.write(p,'code/'+name)
        revision = Path(__file__).parent/'source-revision.txt'
        if revision.exists():
            archive.write(revision,'code/source-revision.txt')
    s3 = boto3.client('s3')
    verified = {}
    names = ['train.jsonl','validation.jsonl','manifest.json','validation-report.json','provenance.zip']
    if (root/'native-probe.json').exists():
        names.append('native-probe.json')
    for name in names:
        path = root/name
        data = path.read_bytes()
        if api_key in data:
            raise ValueError('credential found in published artifact')
        key = args.prefix+'/'+name
        s3.upload_file(str(path),args.bucket,key)
        readback = s3.get_object(Bucket=args.bucket,Key=key)['Body'].read()
        assert hashlib.sha256(data).digest() == hashlib.sha256(readback).digest()
        verified[name] = {'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()}
    receipt = {'s3':f's3://{args.bucket}/{args.prefix}/','objects':verified,'host':'shi-cpu-dev/ip-10-0-100-68'}
    (root/'publication.json').write_text(json.dumps(receipt,indent=2))
    s3.put_object(Bucket=args.bucket,Key=args.prefix+'/publication.json',Body=json.dumps(receipt).encode())
    print('PUBLISHED',json.dumps(receipt),flush=True)


if __name__ == '__main__':
    publish(Args().parse_args())
