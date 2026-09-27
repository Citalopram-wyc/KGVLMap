import hashlib
import json
import random
import time
from pathlib import Path
import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]


def load_config(path):
    with open(path, encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    for k in ('prepared_dir', 'run_dir', 'swin_weights', 'text_weights', 'text_cache'):
        p = Path(cfg[k])
        cfg[k] = str(p if p.is_absolute() else ROOT / p)
    return cfg


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = True


def dump_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding='utf-8')
    for attempt in range(30):
        try:
            tmp.replace(path)
            break
        except PermissionError:
            if attempt == 29:
                raise
            time.sleep(min(.05*(attempt+1), .25))


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for b in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()
