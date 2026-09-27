"""Label-independent geometry inputs and training-only statistics."""
import json
from pathlib import Path
import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
from scipy.ndimage import distance_transform_edt
import torch
from torch.utils.data import Dataset


def source_path(cfg, record, modality):
    """Resolve an explicit source path without copying rasters; retain legacy layouts."""
    if record.get('paths', {}).get(modality):
        return Path(record['paths'][modality])
    root = Path(cfg['data_root'])
    if modality == 'geometry':
        return root / cfg['json_folder'] / (record['id'] + '.json')
    folder = {'rgb': 'img', 'height': 'BH', 'ntl': 'NL', 'label': 'label'}[modality]
    return root / folder / record['image_file']


def read_aligned(path, reference, count=None, categorical=False):
    with rasterio.open(path) as src:
        if count is not None and src.count != count:
            raise ValueError(f'{path}: expected {count} bands, got {src.count}')
        data = src.read().astype(np.float32)
        masks = src.read_masks() > 0
        masks &= np.isfinite(data)
        data[~masks] = np.nan
        # Affine's default 1e-5 world-unit tolerance is almost a whole pixel in
        # geographic Shanghai rasters; do not mistake that offset for alignment.
        aligned = src.crs == reference['crs'] and src.transform.almost_equals(reference['transform'], precision=1e-9) and src.shape == reference['shape']
        if not aligned:
            out = np.full((src.count, *reference['shape']), np.nan, dtype=np.float32)
            reproject(data, out, src_transform=src.transform, src_crs=src.crs,
                      dst_transform=reference['transform'], dst_crs=reference['crs'],
                      src_nodata=np.nan, dst_nodata=np.nan,
                      resampling=Resampling.nearest if categorical else Resampling.bilinear)
            data = out
        return data, np.isfinite(data)


def normalization(values, valid, stats):
    x = np.log1p(np.maximum(values, 0))
    lo, hi, med = (np.asarray(stats[k]) for k in ('lo', 'hi', 'median'))
    x = np.where(valid, x, med)
    return np.clip((x - lo) / np.maximum(hi - lo, 1e-6), 0, 1).astype(np.float32)


def signed_distance(instances):
    mask = instances > 0
    # Include inter-instance boundaries rather than treating touching buildings as one.
    interior = mask.copy()
    interior[1:] &= instances[1:] == instances[:-1]
    interior[:-1] &= instances[:-1] == instances[1:]
    interior[:, 1:] &= instances[:, 1:] == instances[:, :-1]
    interior[:, :-1] &= instances[:, :-1] == instances[:, 1:]
    distance = distance_transform_edt(interior) - distance_transform_edt(~mask)
    return np.clip(distance / 32.0, -1, 1).astype(np.float32)


class BuildingDataset(Dataset):
    def __init__(self, cfg, split='train', augment=False):
        self.cfg, self.split, self.augment = cfg, split, augment
        self.folder = Path(cfg['prepared_dir'])
        self.manifest = json.loads((self.folder / 'manifest.json').read_text(encoding='utf-8'))
        self.stats = json.loads((self.folder / 'statistics.json').read_text(encoding='utf-8'))
        prepared_classes = self.manifest.get('config', {}).get('classes')
        if prepared_classes is not None and prepared_classes != cfg['classes']:
            raise ValueError('Prepared dataset class order differs from the training configuration; do not mix four- and seven-class caches.')
        self.records = [r for r in self.manifest['records'] if r['split'] == split]
        self.records_by_id = {r['id']: r for r in self.records}
        if not self.records:
            raise ValueError(f'No records for split {split}')

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        with rasterio.open(source_path(self.cfg, rec, 'rgb')) as ds:
            rgb = ds.read().astype(np.float32) / 255
            reference = {'crs': ds.crs, 'transform': ds.transform, 'shape': ds.shape}
        bh, bh_valid = read_aligned(source_path(self.cfg, rec, 'height'), reference, 1)
        ntl, ntl_valid = read_aligned(source_path(self.cfg, rec, 'ntl'), reference, self.cfg['ntl_channels'])
        with np.load(self.folder / 'tiles' / (rec['id'] + '.npz'), allow_pickle=False) as z:
            arrays = {k: z[k].copy() for k in z.files}
        if 'image_valid' in arrays:
            observed_image = arrays['image_valid'].astype(bool)
            bh_valid &= observed_image[None]
            ntl_valid &= observed_image[None]
        instances = arrays['instances'].astype(np.int64)
        attrs = normalization(arrays['attributes'], arrays['valid'], self.stats['attributes'])
        height = np.clip(np.log1p(np.maximum(np.nan_to_num(bh), 0)) / self.stats['height_log_p99'], 0, 1).astype(np.float32)
        ntl = np.clip(np.nan_to_num(ntl) / np.array(self.stats['ntl_p99'])[:, None, None], 0, 1).astype(np.float32)
        height[~bh_valid] = 0
        ntl[~ntl_valid] = 0
        morphology = np.stack([(instances > 0).astype(np.float32), arrays['distance'].astype(np.float32)])
        target = arrays['target'].astype(np.int64)
        image_arrays = [rgb, height, ntl, morphology, instances, target]
        if self.augment:
            k = np.random.randint(4)
            image_arrays = [np.rot90(x, k, axes=(-2, -1)) for x in image_arrays]
            if np.random.random() < .5:
                image_arrays = [np.flip(x, axis=-1) for x in image_arrays]
        rgb, height, ntl, morphology, instances, target = [np.ascontiguousarray(x) for x in image_arrays]
        rgb = (rgb - np.array([.485, .456, .406], dtype=np.float32)[:, None, None]) / np.array([.229, .224, .225], dtype=np.float32)[:, None, None]
        result = {'rgb': rgb, 'height': height, 'ntl': ntl, 'morphology': morphology,
                  'instances': instances, 'target': target, 'attrs': attrs,
                  'valid': arrays['valid'], 'complete': arrays['complete'], 'instance_label': arrays['instance_label']}
        result = {k: torch.from_numpy(v) for k, v in result.items()}
        result['id'] = rec['id']
        return result


def collate_buildings(items):
    result, instances, offset = {}, [], 0
    for item in items:
        ids = item['instances'].clone()
        ids[ids > 0] += offset
        instances.append(ids)
        offset += len(item['attrs'])
    for key in ('rgb', 'height', 'ntl', 'morphology', 'target'):
        result[key] = torch.stack([v[key] for v in items])
    result['instances'] = torch.stack(instances)
    for key in ('attrs', 'valid', 'complete', 'instance_label'):
        result[key] = torch.cat([v[key] for v in items])
    result['id'] = [v['id'] for v in items]
    return result


def to_device(batch, device):
    return {k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
