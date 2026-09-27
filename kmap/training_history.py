"""Read completed validation results without changing the active training process."""
import json
import math
from pathlib import Path
import re

METRICS = ('OA', 'Macro-P', 'Macro-R', 'Macro-F1', 'Kappa')


def epoch_history(run):
    history = []
    for path in Path(run).glob('validation_epoch_*.json'):
        match = re.fullmatch(r'validation_epoch_(\d+)\.json', path.name)
        if not match:
            continue
        try:
            record = json.loads(path.read_text(encoding='utf-8-sig'))
            values = {key: float(record[key]) for key in METRICS}
            if not all(math.isfinite(value) for value in values.values()):
                continue
        except (OSError, ValueError, KeyError, TypeError):
            # An incomplete result must not appear as a zero-accuracy epoch.
            continue
        history.append({'epoch': int(match.group(1)), **values,
                        'validation_seconds': record.get('seconds'),
                        'evaluated_pixels': record.get('pixels'), 'source': path.name})
    return sorted(history, key=lambda row: row['epoch'])


def validation_summary(history):
    return {'completed_validation_epochs': len(history),
            'latest_validation': history[-1] if history else None,
            'best_validation': max(history, key=lambda row: row['Macro-F1']) if history else None}
