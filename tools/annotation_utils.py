"""Seven-function annotation codes and atomic annotation-file utilities."""

import hashlib
import json
import os
import time

CLASSES = {1: 'Residential', 2: 'Commercial', 3: 'Administrative', 4: 'Healthcare',
           5: 'Educational', 6: 'Sport and art', 7: 'Industrial'}

COLORS = {0: (0, 0, 0, 255), 1: (255, 165, 0, 255), 2: (255, 0, 0, 255),
          3: (255, 255, 255, 255), 4: (255, 0, 255, 255), 5: (0, 210, 80, 255),
          6: (255, 255, 0, 255), 7: (0, 110, 255, 255), 8: (150, 150, 150, 255),
          255: (0, 0, 0, 0)}

def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def save_json(path, obj):
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')
    for attempt in range(30):
        try:
            os.replace(temporary, path)
            break
        except PermissionError:
            # Windows readers/virus scanners can briefly open without delete sharing.
            if attempt == 29:
                raise
            time.sleep(min(.05 * (attempt + 1), .25))
