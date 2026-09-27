"""Read label-independent footprint geometry from the new global instance rasters."""
import json
from pathlib import Path

import numpy as np
import rasterio
from scipy.ndimage import binary_dilation


def load_instance_geometry(record, reference):
    def read_exact(key):
        with rasterio.open(record['paths'][key]) as src:
            if src.crs != reference['crs'] or src.transform != reference['transform'] or src.shape != reference['shape'] or src.count != 1:
                raise ValueError(f'{record["id"]}: {key} must exactly match the RGB grid.')
            return src.read(1)

    global_raster = read_exact('instance_id').astype(np.int64)
    raw_valid = read_exact('valid_mask')
    if not np.isin(raw_valid, [0, 1]).all():
        raise ValueError('valid_mask must contain only 0 and 1.')
    image_valid = raw_valid == 1
    if (global_raster[~image_valid] != 0).any():
        raise ValueError('Missing-image pixels cannot contain building instances.')
    global_ids = np.unique(global_raster)
    global_ids = global_ids[global_ids > 0]
    ids = np.searchsorted(np.concatenate([[0], global_ids]), global_raster).astype(np.int32)
    # Read only identity and full geometry area. Function fields are never features.
    metadata = json.loads(Path(record['paths']['instances']).read_text(encoding='utf-8'))
    areas_by_id = {int(row['building_id']): float(row['full_area_m2']) for row in metadata}
    if set(areas_by_id) != set(global_ids.tolist()):
        raise ValueError(f'{record["id"]}: footprint raster and geometry metadata disagree.')
    areas = np.array([areas_by_id[int(bid)] for bid in global_ids], dtype=np.float32)
    if not np.isfinite(areas).all() or (areas <= 0).any():
        raise ValueError('Building areas must be finite and positive.')
    # Conservatively exclude footprints touching a crop/no-image edge from auxiliary
    # losses because height and NTL currently come from the supplied tile rasters.
    boundary = binary_dilation(~image_valid, structure=np.ones((3, 3), bool))
    boundary[[0, -1], :] = True
    boundary[:, [0, -1]] = True
    touching = np.unique(ids[boundary])
    complete = ~np.isin(np.arange(1, len(global_ids) + 1), touching)
    return ids, areas, complete, global_ids, image_valid
