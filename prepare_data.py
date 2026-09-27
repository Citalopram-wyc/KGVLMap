"""Prepare geometry, spatial split, training-only normalization and fixed UPKG."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
import collections
import time
import numpy as np
import rasterio
from rasterio.features import rasterize
from rasterio.warp import transform_bounds
from scipy import ndimage
from shapely.geometry import Polygon, mapping
from pyproj import Geod, Transformer
import torch
from kmap.common import load_config, dump_json, sha256_file
from kmap.data import read_aligned, signed_distance, normalization, source_path
from kmap.graph import PhysicalKnowledgeGraph
from kmap.instance_raster import load_instance_geometry


def choose_split(records, cfg):
    if cfg.get('split_manifest'):
        supplied = json.loads(Path(cfg['split_manifest']).read_text(encoding='utf-8'))
        sets = {k: set(v) for k, v in supplied.items() if k in ('train', 'val', 'test')}
        for a in sets:
            for b in sets:
                if a != b and sets[a] & sets[b]:
                    raise ValueError('Supplied split contains repeated IDs.')
        for rec in records:
            rec['split'] = next((k for k, values in sets.items() if rec['id'] in values), 'excluded')
    else:
        block_m = cfg['split_block_m']
        blocks = {}
        for rec in records:
            x0, y0, x1, y1 = rec['bounds_metric']
            key = (int((x0 + x1) / (2 * block_m)), int((y0 + y1) / (2 * block_m)))
            rec['block'] = list(key)
            blocks.setdefault(key, []).append(rec)
        keys = sorted(blocks)
        np.random.default_rng(cfg['seed']).shuffle(keys)
        selected, n = set(), 0
        for k in keys:
            if n >= len(records) * cfg['validation_fraction']:
                break
            selected.add(k)
            n += len(blocks[k])
        for rec in records:
            rec['split'] = 'val' if tuple(rec['block']) in selected else 'train'
    # Purge overlapping/adjacent training tiles, including cross-block boundary leakage.
    val = np.array([r['bounds_metric'] for r in records if r['split'] in ('val', 'test')])
    if len(val) == 0:
        raise ValueError('No validation/test tiles.')
    buffer = cfg['split_buffer_m']
    for r in records:
        if r['split'] != 'train':
            continue
        x0, y0, x1, y1 = r['bounds_metric']
        intersects = (val[:, 0] < x1 + buffer) & (val[:, 2] > x0 - buffer) & (val[:, 1] < y1 + buffer) & (val[:, 3] > y0 - buffer)
        if intersects.any():
            r['split'] = 'excluded_buffer'
    if not any(r['split'] == 'train' for r in records):
        raise ValueError('Spatial split left no training tiles.')
    return records


def polygon_area(points, reference):
    transform = reference['transform']
    xy = np.asarray([transform * (float(x), float(y)) for x, y in points])
    if reference['crs'].is_geographic:
        area, _ = Geod(ellps='WGS84').polygon_area_perimeter(xy[:, 0], xy[:, 1])
        return abs(area)
    factor = reference['crs'].linear_units_factor[1]
    return abs(np.dot(xy[:, 0], np.roll(xy[:, 1], 1)) - np.dot(xy[:, 1], np.roll(xy[:, 0], 1))) * .5 * factor ** 2


def prepare_tile(args):
    cfg, rec = args
    folder, root = Path(cfg['prepared_dir']), Path(cfg['data_root'])
    image_file = rec['image_file']
    with rasterio.open(source_path(cfg, rec, 'rgb')) as ds:
        reference = {'crs': ds.crs, 'transform': ds.transform, 'shape': ds.shape}
    h, w = reference['shape']
    raw_labels, label_valid = read_aligned(source_path(cfg, rec, 'label'), reference, 1, True)
    raw_labels = np.where(label_valid[0], raw_labels[0], 255).astype(np.int32)
    label_map = {int(k): int(v) for k, v in cfg['source_label_map'].items()}
    unknown = set(np.unique(raw_labels).tolist()) - label_map.keys()
    if unknown:
        raise ValueError(f'{rec["id"]}: unmapped label codes {unknown}')
    target = np.full((h, w), 255, dtype=np.uint8)
    for src, dst in label_map.items():
        target[raw_labels == src] = dst
    raster_geometry = rec.get('geometry_format') == 'instance_raster'
    if raster_geometry and cfg['annotation_fraction'] < 1 and cfg.get('annotation_protocol') != 'predefined_global_ids':
        raise ValueError('Global footprint rasters require a fixed global-ID annotation protocol, not per-tile random hiding.')
    global_ids = None
    image_valid = np.ones((h, w), dtype=bool)
    if raster_geometry:
        ids, areas, complete, global_ids, image_valid = load_instance_geometry(rec, reference)
        if not np.all(raw_labels[~image_valid] == 255):
            raise ValueError('Missing-image pixels must retain source label 255.')
        if np.any((raw_labels == 255) & image_valid):
            raise ValueError('New source labels reserve 255 for missing imagery; unlabeled buildings must be 8.')
        if np.any((raw_labels == 8) & (ids == 0)):
            raise ValueError('Unlabeled code 8 must occur inside a valid building footprint.')
        n = len(areas)
    else:
        # Descriptions and functional shape labels are intentionally never read.
        doc = json.loads(source_path(cfg, rec, 'geometry').read_text(encoding='utf-8-sig'))
        if 'shapes' in doc:
            entries = [(s.get('points', []), s.get('shape_type', 'polygon')) for s in doc['shapes']]
        elif 'objects' in doc:
            entries = [(s.get('segmentation', []), 'polygon') for s in doc['objects']]
        else:
            raise ValueError(f'{rec["id"]}: unsupported geometry JSON schema')
        # ISAT / converted LabelMe points refer to the label raster grid, not necessarily RGB.
        # Only geometry coordinates enter the feature path; category names/descriptions do not.
        coordinate_reference = reference
        if rec.get('geometry_reference_path'):
            with rasterio.open(rec['geometry_reference_path']) as ds:
                coordinate_reference = {'transform': ds.transform, 'crs': ds.crs, 'shape': ds.shape}
        coordinate_transform = None
        if coordinate_reference['crs'] != reference['crs']:
            coordinate_transform = Transformer.from_crs(coordinate_reference['crs'], reference['crs'], always_xy=True)
        polygons, areas, complete = [], [], []
        for pts, shape_type in entries:
            if shape_type != 'polygon' or len(pts) < 3:
                continue
            original = np.asarray(pts, dtype=float)
            gh, gw = coordinate_reference['shape']
            is_complete = bool((original[:, 0] > 0).all() and (original[:, 0] < gw - 1).all()
                               and (original[:, 1] > 0).all() and (original[:, 1] < gh - 1).all())
            if rec.get('geometry_reference_path'):
                xy = np.array([coordinate_reference['transform'] * tuple(p) for p in original])
                if coordinate_transform:
                    x, y = coordinate_transform.transform(xy[:, 0], xy[:, 1])
                    xy = np.stack([x, y], axis=1)
                pts = [~reference['transform'] * tuple(p) for p in xy]
            poly = Polygon(pts)
            if not poly.is_valid:
                poly = poly.buffer(0)
            if poly.is_empty or poly.area <= 0:
                continue
            polygons.append(poly)
            areas.append(polygon_area(pts, reference))
            p = np.asarray(pts)
            complete.append(is_complete and bool((p[:, 0] > 0).all() and (p[:, 0] < w - 1).all() and (p[:, 1] > 0).all() and (p[:, 1] < h - 1).all()))
        n = len(polygons)
        # Paint larger polygons first, then smaller polygons; stable index breaks equal-area ties.
        order = sorted(range(n), key=lambda i: (-areas[i], -i))
        ids = rasterize([(mapping(polygons[i]), i + 1) for i in order], out_shape=(h, w), dtype='int32', all_touched=False) if n else np.zeros((h, w), np.int32)
        retained = np.unique(ids)
        retained = retained[retained > 0]
        lookup = np.zeros(n + 1, np.int32)
        lookup[retained] = np.arange(1, len(retained) + 1)
        ids = lookup[ids]
        areas = np.asarray(areas, np.float32)[retained - 1]
        complete = np.asarray(complete, bool)[retained - 1]
        n = len(retained)
    bh, bh_valid = read_aligned(source_path(cfg, rec, 'height'), reference, 1)
    ntl, ntl_valid = read_aligned(source_path(cfg, rec, 'ntl'), reference, cfg['ntl_channels'])
    if cfg['ntl_scalar'] == 'luminance_rgb':
        light = np.einsum('c,chw->hw', np.array([.2126, .7152, .0722]), ntl)
        light_valid = ntl_valid.all(0)
    elif cfg['ntl_scalar'] == 'first_band':
        light, light_valid = ntl[0], ntl_valid[0]
    else:
        raise ValueError('Unknown NTL scalar definition')
    index = np.arange(1, n + 1)
    size = np.bincount(ids.ravel(), minlength=n + 1)[1:]
    vh = bh_valid[0] & (bh[0] >= 0) & image_valid
    vn = light_valid & (light >= 0) & image_valid
    hc = np.bincount(ids[vh], minlength=n + 1)[1:]
    nc = np.bincount(ids[vn], minlength=n + 1)[1:]
    valid = np.stack([hc >= size * cfg['minimum_attribute_coverage'], np.ones(n, bool), nc >= size * cfg['minimum_attribute_coverage']], axis=1)
    # Labeled statistics on aligned observation pixels; never parse numerical values from descriptions.
    height = ndimage.median(np.where(vh, bh[0], 0), labels=np.where(vh, ids, 0), index=index) if n else np.empty(0)
    light_sum = np.bincount(ids[vn], weights=light[vn], minlength=n + 1)[1:]
    light_mean = light_sum / np.maximum(nc, 1)
    attrs = np.stack([height, areas, light_mean], axis=1).astype(np.float32)
    attrs[~np.isfinite(attrs)] = 0
    ilabel = np.zeros(n, np.int64)
    observed = (target > 0) & (target < len(cfg['classes']) + 1) & (ids > 0)
    class_counts = np.bincount(ids[observed] * (len(cfg['classes']) + 1) + target[observed], minlength=(n + 1) * (len(cfg['classes']) + 1)).reshape(n + 1, -1)[1:]
    if n:
        majority = class_counts.argmax(1)
        confident = class_counts.max(1) >= 0.5 * size
        ilabel[confident] = majority[confident]
    if rec['split'] == 'train' and cfg['annotation_fraction'] < 1 and cfg.get('annotation_protocol') != 'predefined_global_ids':
        for j in range(n):
            token = hashlib.sha256(f'{cfg["seed"]}:{rec["id"]}:{j}'.encode()).digest()
            if int.from_bytes(token[:8], 'little') / 2**64 >= cfg['annotation_fraction']:
                ilabel[j] = 0
                target[ids == j + 1] = 255
    # A footprint with no usable functional annotation is ignored by direct segmentation.
    if n:
        unknown_lut = np.concatenate([[False], ilabel == 0])
        target[unknown_lut[ids]] = 255
    tile_path = folder / 'tiles' / (rec['id'] + '.npz')
    extra = {'global_instance_ids': global_ids, 'image_valid': image_valid} if raster_geometry else {}
    atomic_npz(tile_path, instances=ids, attributes=attrs, valid=valid,
               complete=complete, instance_label=ilabel, target=target,
               distance=signed_distance(ids).astype(np.float16), **extra)
    counts = np.bincount(target[target != 255], minlength=len(cfg['classes']) + 1)
    valid_h = np.log1p(np.maximum(np.where(image_valid, bh[0], np.nan)[::16, ::16], 0)).ravel()
    sample_n = np.where(image_valid[None], ntl, np.nan)[:, ::16, ::16].reshape(ntl.shape[0], -1)
    return {'id': rec['id'], 'attrs': attrs, 'valid': valid, 'complete': complete,
            'labels': ilabel, 'counts': counts, 'height_sample': valid_h, 'ntl_sample': sample_n,
            'global_instance_ids': global_ids}


def atomic_npz(path, **arrays):
    temporary = path.with_suffix('.npz.tmp')
    with temporary.open('wb') as stream:
        np.savez_compressed(stream, **arrays)
    temporary.replace(path)


def prepare_cached_tile(args):
    cfg, rec, fingerprint = args
    folder = Path(cfg['prepared_dir'])
    summary = folder / 'tile_statistics' / (rec['id'] + '.npz')
    tile = folder / 'tiles' / (rec['id'] + '.npz')
    keys = ('rgb', 'height', 'ntl', 'label', 'instance_id', 'instances', 'valid_mask') if rec.get('geometry_format') == 'instance_raster' else ('rgb', 'height', 'ntl', 'label', 'geometry')
    signatures = []
    for key in keys:
        path = source_path(cfg, rec, key)
        stat = path.stat()
        signatures.append((str(path), stat.st_size, stat.st_mtime_ns))
    signature = hashlib.sha256(json.dumps([fingerprint, rec, signatures], sort_keys=True).encode()).hexdigest()
    if summary.exists() and tile.exists():
        with np.load(summary, allow_pickle=False) as saved:
            if str(saved['signature']) == signature and int(saved['tile_bytes']) == tile.stat().st_size:
                result = {k: saved[k] for k in saved.files if k not in ('signature', 'tile_bytes', 'has_global_ids')}
                result['id'] = str(result['id'])
                if not bool(saved['has_global_ids']):
                    result['global_instance_ids'] = None
                return result
    result = prepare_tile((cfg, rec))
    stored = dict(result)
    stored['has_global_ids'] = result['global_instance_ids'] is not None
    if not stored['has_global_ids']:
        stored['global_instance_ids'] = np.empty(0, np.uint64)
    atomic_npz(summary, **stored, signature=signature, tile_bytes=tile.stat().st_size)
    return result


def initialize_worker():
    # Eight raster workers must not each create a full CPU thread pool.
    torch.set_num_threads(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='configs/seven_class.yaml')
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--resume', action='store_true', help='Reuse completed tiles from an identical preparation job.')
    args = ap.parse_args()
    cfg = load_config(args.config)
    root, folder = Path(cfg['data_root']), Path(cfg['prepared_dir'])
    if (folder / 'manifest.json').exists():
        raise FileExistsError(f'{folder} is already prepared; choose a new prepared_dir to avoid overwriting a split.')
    (folder / 'tiles').mkdir(parents=True, exist_ok=True)
    (folder / 'tile_statistics').mkdir(parents=True, exist_ok=True)
    code_files = [Path(__file__), Path(__file__).parent / 'kmap' / 'data.py',
                  Path(__file__).parent / 'kmap' / 'instance_raster.py', Path(__file__).parent / 'kmap' / 'graph.py']
    provenance = {'config': cfg, 'code_sha256': {p.name: sha256_file(p) for p in code_files},
                  'source_manifest_sha256': sha256_file(cfg['source_manifest']) if cfg.get('source_manifest') else None}
    fingerprint = hashlib.sha256(json.dumps(provenance, sort_keys=True).encode()).hexdigest()
    job_path = folder / 'preparation_job.json'
    if job_path.exists():
        old_job = json.loads(job_path.read_text(encoding='utf-8'))
        if not args.resume or old_job['fingerprint'] != fingerprint:
            raise ValueError('Partial preparation exists. Use --resume with the identical code, configuration and source manifest, or choose a new directory.')
    dump_json(job_path, {'fingerprint': fingerprint, **provenance})
    started = time.monotonic()
    dump_json(folder / 'progress.json', {'phase': 'checking_sources', 'completed': 0})
    records, excluded = [], []
    source_catalog = None
    if cfg.get('source_manifest'):
        source_catalog = json.loads(Path(cfg['source_manifest']).read_text(encoding='utf-8'))
        records = [dict(r) for r in source_catalog['records'] if r['split'] in ('train', 'val', 'test')]
        excluded = [dict(r) for r in source_catalog['records'] if r['split'] not in ('train', 'val', 'test')]
        if source_catalog['classes'] != cfg['classes']:
            raise ValueError('Source catalog and training configuration have different class orders.')
        if cfg.get('annotation_protocol') == 'predefined_global_ids':
            policy = source_catalog.get('annotation_policy', {})
            if policy.get('visible_fraction') != cfg['annotation_fraction'] or policy.get('seed') != cfg['seed']:
                raise ValueError('Source annotation policy does not match configuration fraction and seed.')
        for rec in records:
            keys = ('rgb', 'height', 'ntl', 'label', 'instance_id', 'instances', 'valid_mask') if rec.get('geometry_format') == 'instance_raster' else ('rgb', 'height', 'ntl', 'label', 'geometry')
            for key in keys:
                if not source_path(cfg, rec, key).is_file():
                    raise FileNotFoundError(source_path(cfg, rec, key))
    for path in ([] if source_catalog else sorted((root / 'img').glob('*.tif'))):
        missing = [str(root / key / path.name) for key in ('BH', 'NL', 'label') if not (root / key / path.name).exists()]
        if not (root / cfg['json_folder'] / (path.stem + '.json')).exists():
            missing.append('geometry JSON')
        if missing:
            excluded.append({'id': path.stem, 'missing': missing})
            continue
        with rasterio.open(path) as ds:
            if ds.count != 3 or ds.shape != (cfg['crop_size'], cfg['crop_size']):
                raise ValueError(f'Unexpected RGB raster shape: {path}')
            bounds = transform_bounds(ds.crs, 'EPSG:3857', *ds.bounds)
            records.append({'id': path.stem, 'image_file': path.name, 'crs': str(ds.crs),
                            'bounds': list(ds.bounds), 'bounds_metric': list(bounds)})
    if not source_catalog:
        records = choose_split(records, cfg)
    print('Spatial split:', dict(collections.Counter(r['split'] for r in records)), flush=True)
    # Save a proposed split early; complete manifest is published only after all statistics exist.
    dump_json(folder / 'split.json', {key: [r['id'] for r in records if r['split'] == key] for key in ('train', 'val', 'test', 'excluded_buffer')})
    active = [r for r in records if r['split'] in ('train', 'val', 'test')]
    train_ids = {r['id'] for r in active if r['split'] == 'train'}
    train_results, val_counts = [], np.zeros(len(cfg['classes']) + 1, np.int64)
    with ProcessPoolExecutor(max_workers=args.workers, initializer=initialize_worker) as pool:
        for i, result in enumerate(pool.map(prepare_cached_tile, [(cfg, r, fingerprint) for r in active], chunksize=1)):
            if result['id'] in train_ids:
                train_results.append(result)
            else:
                val_counts += result['counts']
            if i % 50 == 0 or i + 1 == len(active):
                elapsed = time.monotonic() - started
                eta = elapsed / (i + 1) * (len(active) - i - 1)
                print(f'Prepared {i + 1}/{len(active)}; elapsed {elapsed/60:.1f} min; remaining ~{eta/60:.1f} min', flush=True)
                dump_json(folder / 'progress.json', {'phase': 'preparing_tiles', 'completed': i + 1,
                          'total': len(active), 'elapsed_seconds': elapsed, 'estimated_remaining_seconds': eta})
    dump_json(folder / 'progress.json', {'phase': 'fitting_training_statistics_and_graph', 'completed': len(active), 'total': len(active)})
    attrs = np.concatenate([r['attrs'] for r in train_results])
    valid = np.concatenate([r['valid'] for r in train_results])
    complete = np.concatenate([r['complete'] for r in train_results])
    labels = np.concatenate([r['labels'] for r in train_results])
    tile_instance_occurrences = len(attrs)
    if all(r['global_instance_ids'] is not None for r in train_results):
        global_ids = np.concatenate([r['global_instance_ids'] for r in train_results])
        # Each building contributes once to attribute statistics and UPKG fitting.
        # Prefer complete observations; incomplete appearances remain conditioning-only.
        order = np.lexsort((np.arange(len(global_ids)), -valid.sum(1), -complete.astype(int), global_ids))
        sorted_ids = global_ids[order]
        first = np.r_[True, sorted_ids[1:] != sorted_ids[:-1]]
        starts = np.flatnonzero(first)
        mins = np.minimum.reduceat(labels[order], starts)
        maxs = np.maximum.reduceat(labels[order], starts)
        if not np.array_equal(mins, maxs):
            raise ValueError('A global training building has inconsistent visible/hidden functional labels across tiles.')
        chosen = order[first]
        attrs, valid, complete, labels = (a[chosen] for a in (attrs, valid, complete, labels))
    lo, hi, med = [], [], []
    for d in range(3):
        x = np.log1p(attrs[valid[:, d] & complete, d])
        if not len(x):
            raise ValueError(f'No complete training observations for attribute {d}')
        q = np.percentile(x, [1, 99, 50])
        lo.append(float(q[0])); hi.append(float(q[1])); med.append(float(q[2]))
    attr_stats = {'lo': lo, 'hi': hi, 'median': med}
    counts = np.sum([r['counts'] for r in train_results], axis=0)
    positive = counts[counts > 0]
    class_weights = np.divide(np.median(positive), counts, out=np.zeros_like(counts, dtype=float), where=counts > 0)
    hs = np.concatenate([r['height_sample'] for r in train_results])
    ns = np.concatenate([r['ntl_sample'] for r in train_results], axis=1)
    stats = {'attributes': attr_stats, 'class_counts': counts.tolist(), 'class_weights': class_weights.tolist(),
             'height_log_p99': max(float(np.nanpercentile(hs, 99)), 1e-6),
             'ntl_p99': np.maximum(np.nanpercentile(ns, 99, axis=1), 1e-6).tolist(),
             'scope': 'training split only', 'validation_counts_for_audit_only': val_counts.tolist()}
    v = normalization(attrs, valid, attr_stats)
    graph_valid = valid.all(1) & complete
    labeled = graph_valid & (labels > 0)
    graph = PhysicalKnowledgeGraph(len(cfg['classes']), cfg['graph_rho'], cfg['graph_steps'], cfg['graph_smoothing'])
    graph.fit(v[graph_valid], v[labeled], labels[labeled])
    torch.save(graph.state_dict(), folder / 'graph.pt')
    dump_json(folder / 'statistics.json', stats)
    if all(r.get('geometry_format') == 'instance_raster' for r in active):
        geometry_limitations = [
            'Global source building IDs and full polygon areas are preserved. Height and light conditioning values are still measured per supplied tile.',
            'Footprints touching crop or missing-image edges are conservatively excluded from graph fitting, scaling statistics and auxiliary losses.',
            'Training attribute statistics and graph fitting count each global building once, preferring complete observations.',
            'Original native-grid physical radiometry is not restored from the supplied tile rasters.']
    else:
        geometry_limitations = [
            'Tile-clipped polygons retain local attributes for conditioning; excluded from graph fitting, attribute-scaling statistics and auxiliary losses.',
            'Attributes measured from supplied aligned rasters; original native-grid radiometry and full-scene vector geometry were not supplied.']
    dump_json(folder / 'manifest.json', {'records': records, 'missing': excluded, 'config': cfg,
        'split_counts': dict(collections.Counter(r['split'] for r in records)),
        'training_instances': len(attrs), 'labeled_graph_instances': int(labeled.sum()),
        'training_tile_instance_occurrences': tile_instance_occurrences,
        'unlabeled_valid_instances': int((graph_valid & (labels == 0)).sum()),
        'boundary_instances_excluded_from_auxiliary_losses': int((~complete).sum()),
        'limitations': (source_catalog.get('limitations', []) if source_catalog else ['Current supplied rasters and polygons, not the full manuscript dataset.',
                       'Spatial blocks and exclusion buffer replace the manuscript administrative split.']) + geometry_limitations + [
                       'For RGB NTL rasters, the scalar is Rec.709 brightness, not calibrated radiance.']})
    print('Preparation complete', folder, 'train_pixels', counts.tolist(), flush=True)
    dump_json(folder / 'progress.json', {'phase': 'complete', 'completed': len(active), 'total': len(active),
              'elapsed_seconds': time.monotonic() - started})


if __name__ == '__main__':
    main()
