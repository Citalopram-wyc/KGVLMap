"""Hide training functions by global building ID; preserve raw missing-image code 255."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import sys
import time

import numpy as np
import rasterio
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from annotation_utils import CLASSES, COLORS, digest, save_json

HIDDEN = None


def stratified_selection(class_ids, fraction, seed):
    codes = sorted(class_ids)
    total = sum(len(class_ids[c]) for c in codes)
    target = round(total * fraction)
    desired = {c: len(class_ids[c]) * fraction for c in codes}
    allocated = {c: int(np.floor(desired[c])) for c in codes}
    remainder = target - sum(allocated.values())
    ranked = sorted(codes, key=lambda c: (-(desired[c]-allocated[c]), c))
    for c in ranked[:remainder]:
        allocated[c] += 1
    rng = np.random.default_rng(seed)
    selected = []
    for c in codes:
        ids = np.array(sorted(class_ids[c]), dtype=np.int64)
        selected.extend(rng.choice(ids, allocated[c], replace=False).tolist())
    return sorted(selected), allocated


def initialize(hidden_path):
    global HIDDEN
    hidden = json.loads(Path(hidden_path).read_text(encoding='utf-8'))['hidden_building_ids']
    HIDDEN = np.zeros(max(hidden, default=0)+1, dtype=bool)
    HIDDEN[hidden] = True


def rewrite_label(task):
    record, destination = task
    paths = record['paths']
    with rasterio.open(paths['label']) as src:
        source = src.read(1)
        profile = src.profile
        crs, transform, shape = src.crs, src.transform, src.shape
    with rasterio.open(paths['instance_id']) as src:
        if (src.crs, src.transform, src.shape) != (crs, transform, shape):
            raise ValueError('Footprint and label grids differ.')
        ids = src.read(1)
    with rasterio.open(paths['valid_mask']) as src:
        image_valid = src.read(1) == 1
    if not np.array_equal(source == 255, ~image_valid):
        raise ValueError('255 must denote missing imagery, not hidden functional annotations.')
    if np.any((ids > 0) & ~image_valid):
        raise ValueError('Missing-image pixels cannot be building instances.')
    hidden_mask = np.zeros(shape, dtype=bool)
    within = ids < len(HIDDEN)
    hidden_mask[within] = HIDDEN[ids[within]]
    hidden_mask &= image_valid & (ids > 0)
    if not np.isin(source[hidden_mask], list(CLASSES)).all():
        raise ValueError('Can only hide known functional labels inside valid footprints.')
    label = source.copy()
    label[hidden_mask] = 8
    if not np.array_equal(label == 255, source == 255):
        raise ValueError('Missing-image mask changed.')
    path = Path(destination) / (Path(paths['label']).stem + '.tif')
    if path.exists():
        raise FileExistsError(path)
    with rasterio.open(path, 'w', **profile) as dst:
        dst.write(label, 1)
        dst.write_colormap(1, COLORS)
        dst.update_tags(annotation_protocol='80% visible + 20% hidden by global building ID',
                        unlabeled_building_code='8', missing_image_code='255',
                        reference_full_label=str(paths['label']))
    with rasterio.open(path) as check:
        if (check.crs, check.transform, check.shape) != (crs, transform, shape) or not np.array_equal(check.read(1), label):
            raise ValueError(f'Output validation failed: {path}')
    values, counts = np.unique(label, return_counts=True)
    return dict(id=record['id'], label=str(path), hidden_pixels=int(hidden_mask.sum()),
                hidden_ids_in_tile=np.unique(ids[hidden_mask]).tolist(),
                counts={int(k):int(v) for k,v in zip(values, counts)})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--annotations', required=True)
    parser.add_argument('--config', default='configs/seven_class.yaml')
    parser.add_argument('--output', required=True)
    parser.add_argument('--hidden-fraction', type=float, default=.2)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    if args.hidden_fraction != .2:
        raise ValueError('This named experiment is the requested 80/20 protocol.')
    root, output = Path(args.annotations).resolve(), Path(args.output).resolve()
    if output.exists():
        raise FileExistsError('Select a fresh output directory; original annotations are never overwritten.')
    source_snapshot_hash = digest(root / 'source_snapshot.json')
    audit = json.loads((root / 'identity_audit.json').read_text(encoding='utf-8'))
    if not audit['passed'] or audit['shared_train_val_building_count']:
        raise ValueError('Dataset must pass identity audit without shared train/validation buildings.')
    source_rows = [json.loads(line) for line in (root / 'aligned_sources.jsonl').read_text(encoding='utf-8').splitlines()]
    train = [r for r in source_rows if r['existing_split'] == 'train']
    truth = {}
    for row in train:
        for building in json.loads(Path(row['paths']['instances']).read_text(encoding='utf-8')):
            bid, code = building['building_id'], building['function_code']
            if code not in CLASSES or (bid in truth and truth[bid] != code):
                raise ValueError('Training building functions are missing or inconsistent.')
            truth[bid] = code
    class_ids = {c:[] for c in CLASSES}
    for bid, code in truth.items():
        class_ids[code].append(bid)
    hidden, allocated = stratified_selection(class_ids, args.hidden_fraction, args.seed)
    output.mkdir(parents=True)
    (output / 'label_train').mkdir()
    policy = dict(visible_fraction=.8, hidden_fraction=.2, seed=args.seed,
                  sampling='class-stratified without replacement; largest-remainder integer allocation',
                  unit='unique global building ID; all appearances hidden together',
                  training_buildings=len(truth), hidden_buildings=len(hidden), visible_buildings=len(truth)-len(hidden),
                  realized_hidden_fraction=len(hidden)/len(truth),
                  counts_by_class={str(c):dict(name=CLASSES[c],total=len(class_ids[c]),hidden=allocated[c],visible=len(class_ids[c])-allocated[c]) for c in CLASSES},
                  unlabeled_building_code=8, missing_image_code=255, background_code=0,
                  network_output_classes=8, annotation_protocol='predefined_global_ids',
                  source_snapshot_sha256=source_snapshot_hash, validation_labels_unchanged=True,
                  hidden_building_ids=hidden)
    save_json(output / 'annotation_policy.json', policy)
    started = time.monotonic()
    print(f'Train buildings: {len(truth)}; hide: {len(hidden)}; retain: {len(truth)-len(hidden)}', flush=True)
    results = {}
    with ProcessPoolExecutor(args.workers, initializer=initialize, initargs=(str(output/'annotation_policy.json'),)) as pool:
        for i, result in enumerate(pool.map(rewrite_label, [(r,str(output/'label_train')) for r in train], chunksize=4), 1):
            results[result['id']] = result
            if i % 500 == 0 or i == len(train):
                print(f'Masked and verified {i}/{len(train)} training tiles', flush=True)
                save_json(output/'status.json', dict(state='writing_training_labels',completed=i,total=len(train),elapsed_seconds=time.monotonic()-started))
    observed_hidden = {bid for r in results.values() for bid in r['hidden_ids_in_tile']}
    if observed_hidden != set(hidden):
        raise ValueError('Some selected hidden buildings were not present in written labels.')
    records, diagnostic = [], []
    for row in source_rows:
        paths = dict(row['paths'])
        split = row['existing_split']
        if split == 'train':
            diagnostic.append(dict(id=row['id'], full_truth_label=paths['label'], instance_id=paths['instance_id'],valid_mask=paths['valid_mask']))
            paths['label'] = results[row['id']]['label']
        records.append(dict(id=row['id'],city='HF',district=row['district'],source_stem=Path(paths['rgb']).stem,
                            image_file=Path(paths['rgb']).name,split=split,nominal_split=row['nominal_split'],paths=paths,
                            geometry_format='instance_raster'))
    compact_policy = {k:v for k,v in policy.items() if k != 'hidden_building_ids'}
    source_catalog = dict(schema_version=2,classes=list(CLASSES.values()),records=records,
                          annotation_policy=compact_policy,annotation_policy_path=str(output/'annotation_policy.json'),
                          limitations=['Hefei training/validation only; Shanghai semantic harmonization is separate.',
                                       'Height and NTL observations remain the existing tile rasters; RGB NTL brightness is not calibrated radiance.'])
    save_json(output/'source_manifest.json', source_catalog)
    save_json(output/'hidden_truth_diagnostic.json', dict(usage='Post-training diagnostic only; never use hidden truth for training, graph fitting, class weights or checkpoint/hyperparameter selection.',
                annotation_policy=str(output/'annotation_policy.json'),records=diagnostic))
    totals = Counter()
    for row in results.values():
        totals.update(row['counts'])
    report = dict(**compact_policy,training_tiles=len(train),validation_tiles=sum(r['split']=='val' for r in records),
                  training_pixel_counts=dict(totals),hidden_pixel_fraction_of_buildings=totals[8]/sum(totals[c] for c in range(1,9)),
                  source_labels_modified=False,missing_image_mask_unchanged=True,all_written_labels_readback_verified=True,
                  every_selected_global_id_was_hidden=True,elapsed_seconds=time.monotonic()-started)
    if digest(root/'source_snapshot.json') != source_snapshot_hash:
        raise ValueError('Annotation source snapshot changed during selection.')
    save_json(output/'summary.json', report)
    config_path = Path(args.config)
    cfg = yaml.safe_load(config_path.read_text(encoding='utf-8'))
    cfg.update(classes=list(CLASSES.values()),annotation_fraction=.8,annotation_protocol='predefined_global_ids',seed=args.seed,
               source_manifest=str(output/'source_manifest.json'),
               prepared_dir='outputs/prepared_80pct',
               run_dir='outputs/runs/seven_class_80pct',
               text_cache='weights/remoteclip_seven_class_text.pt')
    cfg['prompts'][2] = 'Administrative buildings used for government and public administration.'
    cfg['prompts'][3] = 'Healthcare buildings used for medical treatment and healthcare services, including hospitals and clinics.'
    cfg['prompts'][4] = 'Educational buildings used for teaching, learning, and educational activities.'
    (output/'training_config.yaml').write_text(yaml.safe_dump(cfg,sort_keys=False,allow_unicode=True),encoding='utf-8')
    baseline = dict(cfg,lambda_topo=0.0,graph_variant='none',run_dir=cfg['run_dir']+'_without_upkg')
    (output/'control_without_upkg.yaml').write_text(yaml.safe_dump(baseline,sort_keys=False,allow_unicode=True),encoding='utf-8')
    readme = f'''# 合肥 80% 标注 / 20% 未标注训练协议

只隐藏训练集功能标签。共 {len(truth):,} 个不同训练建筑，隐藏 {len(hidden):,} 个，保留 {len(truth)-len(hidden):,} 个。按七类分层、固定种子 {args.seed} 抽样，同一源建筑在所有切片中状态相同。由于整数取整，实际隐藏比例为 {100*len(hidden)/len(truth):.6f}%。

- 源标签 **255 仅表示无有效影像**，所有这些像素原样保留；它们没有实例编号，不参与分割、PICR 或 UPKG 辅助监督。
- 新标签 **8 表示有影像、有足迹但隐藏功能标签的建筑**。网络仍输出背景及七类功能（8 类输出），编码 8 不是新增预测类别。
- 数据加载时将源编码 8 与 255 都映射为分割损失的 ignore=255；通过有效影像掩膜、实例编号和可见实例标签区分二者。UPKG 只面向具有有效物理观测的未标注建筑。
- 原始完整标签、足迹、实例编号、RGB/BH/NL、区县划分和验证标签均未修改。`label_train` 仅保存本实验的训练标签。
- `annotation_policy.json` 固定隐藏名单；`source_manifest.json` 为训练读取入口；`training_config.yaml` 为本实验配置。
- 物理编码输入保持不变。UPKG 类别关联仅使用剩余可见训练标签，类别权重仅统计可见训练像素，隐藏标签不进入这些计算。
- `hidden_truth_diagnostic.json` 仅供训练结束后对隐藏建筑做诊断，不能用于损失、教师目标、选择模型或调参；正式效果看完整验证集。该诊断属于训练域内诊断，不是独立测试。
- `control_without_upkg.yaml` 使用完全相同的标注预算和数据，关闭 UPKG。它保留 PICR，可用于分析 UPKG 增益，不能称为完全纯监督对照。

本步骤仅生成训练标签、协议与配置，没有启动全量物理属性/图缓存准备或正式训练。读取器已适配新实例栅格。现阶段完整面积来自源多边形，高度和夜光仍从给定切片读取；触及切片或无影像边界的建筑保守地排除辅助损失。这不等同于论文要求的原生物理栅格全建筑统计。
'''
    (output/'README.md').write_text(readme,encoding='utf-8')
    save_json(output/'status.json', dict(state='complete',training_tiles=len(train),hidden_buildings=len(hidden),elapsed_seconds=time.monotonic()-started))
    print(json.dumps({k:report[k] for k in ('training_buildings','visible_buildings','hidden_buildings','training_tiles','validation_tiles','hidden_pixel_fraction_of_buildings')},ensure_ascii=False),flush=True)


if __name__ == '__main__':
    main()
