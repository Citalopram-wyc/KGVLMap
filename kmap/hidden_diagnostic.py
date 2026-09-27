"""Post-training diagnostics on hidden training functions, never a test set."""
import json
from pathlib import Path

import numpy as np
import rasterio
import torch

from kmap.data import BuildingDataset, source_path


class HiddenTrainingDiagnostic(BuildingDataset):
    def __init__(self, cfg, diagnostic_manifest):
        super().__init__(cfg, 'train', augment=False)
        diagnostic = json.loads(Path(diagnostic_manifest).read_text(encoding='utf-8'))
        self.truth = {r['id']:r for r in diagnostic['records']}
        if not {r['id'] for r in self.records}.issubset(self.truth):
            raise ValueError('Hidden diagnostic manifest does not cover these training records.')
        policy = json.loads(Path(diagnostic['annotation_policy']).read_text(encoding='utf-8'))
        if policy['seed'] != cfg['seed'] or policy['visible_fraction'] != cfg['annotation_fraction']:
            raise ValueError('Hidden diagnostic annotation policy does not match the experiment.')

    def __getitem__(self, idx):
        item = super().__getitem__(idx)
        rec = self.records[idx]
        paths = self.truth[rec['id']]
        with rasterio.open(paths['full_truth_label']) as src:
            truth = src.read(1)
            grid = (src.crs,src.transform,src.shape)
        with rasterio.open(source_path(self.cfg,rec,'label')) as src:
            masked = src.read(1)
            if (src.crs,src.transform,src.shape) != grid:
                raise ValueError('Diagnostic truth and masked labels use different grids.')
        with rasterio.open(paths['valid_mask']) as src:
            valid = src.read(1) == 1
        hidden = (masked == 8) & valid
        if not np.isin(truth[hidden],np.arange(1,len(self.cfg['classes'])+1)).all():
            raise ValueError('Hidden diagnostic pixels lack known ground truth.')
        # Features, footprints and conditions stay as during training; full truth
        # reaches only the metric accumulator through this explicit evaluator.
        item['target'] = torch.from_numpy(np.where(hidden,truth,255).astype(np.int64))
        return item
