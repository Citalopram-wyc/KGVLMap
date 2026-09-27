"""Download public, publisher-hosted pretrained weights; never fall back to random weights."""
import argparse
import hashlib
import json
from pathlib import Path
import requests

SOURCES = {
    'swin_base_in22k.safetensors': 'https://huggingface.co/timm/swin_base_patch4_window7_224.ms_in22k/resolve/main/model.safetensors',
    'RemoteCLIP-ViT-B-32.pt': 'https://huggingface.co/chendelong/RemoteCLIP/resolve/main/RemoteCLIP-ViT-B-32.pt',
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', default='weights')
    args = ap.parse_args()
    folder = Path(args.output)
    folder.mkdir(parents=True, exist_ok=True)
    provenance = {}
    for name, url in SOURCES.items():
        target = folder / name
        if not target.exists():
            temp = target.with_suffix(target.suffix + '.part')
            print('Downloading', name, flush=True)
            with requests.get(url, stream=True, timeout=(30, 120)) as response:
                response.raise_for_status()
                with temp.open('wb') as out:
                    count = 0
                    for block in response.iter_content(4 * 1024 * 1024):
                        out.write(block)
                        count += len(block)
                        if count % (64 * 1024 * 1024) == 0:
                            print(name, count // (1024 * 1024), 'MiB', flush=True)
            temp.replace(target)
        digest = hashlib.sha256()
        with target.open('rb') as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
                digest.update(block)
        provenance[name] = {'url': url, 'bytes': target.stat().st_size, 'sha256': digest.hexdigest()}
        print('Ready', name, flush=True)
    (folder / 'provenance.json').write_text(json.dumps(provenance, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
