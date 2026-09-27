"""Single-GPU training, exact denominator-aware accumulation, validation and resume."""
import argparse
import datetime
import itertools
import json
import math
import os
from pathlib import Path
import random
import time
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from kmap.common import load_config, seed_everything, dump_json, sha256_file
from kmap.data import BuildingDataset, collate_buildings, to_device, source_path
from kmap.graph import PhysicalKnowledgeGraph
from kmap.losses import KGVLMapLoss
from kmap.metrics import SegmentationMetrics
from kmap.model import KGVLMap
from kmap.training_history import epoch_history, validation_summary


def worker_init(worker_id):
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)
    torch.set_num_threads(1)


def make_loader(dataset, cfg, train=False):
    return DataLoader(dataset, batch_size=cfg['micro_batch'], shuffle=train,
        num_workers=cfg['num_workers'], pin_memory=True, collate_fn=collate_buildings,
        worker_init_fn=worker_init, persistent_workers=cfg['num_workers'] > 0,
        **({'prefetch_factor': 2} if cfg['num_workers'] > 0 else {}))


def graph_from_config(cfg, device):
    graph = PhysicalKnowledgeGraph(len(cfg['classes']), cfg['graph_rho'], cfg['graph_steps'], cfg['graph_smoothing'])
    graph.load_state_dict(torch.load(Path(cfg['prepared_dir']) / 'graph.pt', map_location='cpu', weights_only=True))
    return graph.to(device).eval()


@torch.no_grad()
def validate(model, loader, device, amp=True, limit=None, output_dir=None, classes=None):
    model.eval()
    metric = SegmentationMetrics(len(classes) + 1)
    start = time.time()
    for i, cpu_batch in enumerate(loader):
        if limit is not None and i >= limit:
            break
        batch = to_device(cpu_batch, device)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp and device.type == 'cuda'):
            out = model(batch)
        if not torch.isfinite(out['logits']).all():
            raise FloatingPointError('Validation logits contain NaN/Inf')
        pred = out['logits'].argmax(1)
        metric.update(pred, batch['target'])
        if output_dir is not None:
            import rasterio
            folder = Path(output_dir)
            folder.mkdir(parents=True, exist_ok=True)
            for j, tile_id in enumerate(batch['id']):
                source = source_path(loader.dataset.cfg, loader.dataset.records_by_id[tile_id], 'rgb')
                with rasterio.open(source) as src:
                    profile = src.profile.copy()
                profile.update(count=1, dtype='uint8', nodata=None, compress='deflate')
                with rasterio.open(folder / (tile_id + '.tif'), 'w', **profile) as dst:
                    dst.write(pred[j].cpu().numpy().astype(np.uint8), 1)
        if (i + 1) % 25 == 0:
            print(f'validation {i + 1}/{len(loader)} ({time.time()-start:.0f}s)', flush=True)
    result = metric.compute()
    result['class_names'] = ['Background'] + classes
    result['seconds'] = time.time() - start
    model.train()
    return result


def learning_rate(step, steps_per_epoch, cfg):
    warm = max(1, cfg['warmup_epochs'] * steps_per_epoch)
    total = cfg['epochs'] * steps_per_epoch
    lo, hi = cfg['min_learning_rate'], cfg['learning_rate']
    if step < warm:
        return lo + (hi - lo) * step / warm
    progress = min(1, (step - warm) / max(1, total - warm))
    return lo + (hi - lo) * .5 * (1 + math.cos(math.pi * progress))


def atomic_checkpoint(path, payload):
    temp = path.with_suffix('.tmp')
    torch.save(payload, temp)
    temp.replace(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='configs/seven_class.yaml')
    ap.add_argument('--resume')
    ap.add_argument('--run-dir')
    ap.add_argument('--smoke', action='store_true', help='Real pretrained model, real data, one optimizer step and two validation tiles.')
    ap.add_argument('--micro-batch', type=int)
    ap.add_argument('--workers', type=int)
    args = ap.parse_args()
    cfg = load_config(args.config)
    if args.run_dir:
        cfg['run_dir'] = str(Path(args.run_dir).resolve())
    if args.workers is not None:
        cfg['num_workers'] = args.workers
    if args.micro_batch:
        cfg['micro_batch'] = args.micro_batch
    run = Path(cfg['run_dir'])
    if args.smoke and not args.run_dir:
        run = run.parent / (run.name + '_smoke')
    run.mkdir(parents=True, exist_ok=True)
    if (run / 'last.pt').exists() and not args.resume:
        raise FileExistsError('Run already contains a checkpoint. Use --resume or a new --run-dir.')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable in the selected Python environment.')
    device = torch.device('cuda:0')
    torch.set_num_threads(4)
    seed_everything(cfg['seed'])
    train_data = BuildingDataset(cfg, 'train', cfg['augment_flip_rotate'])
    val_data = BuildingDataset(cfg, 'val', False)
    prepared_cfg = train_data.manifest['config']
    if prepared_cfg.get('annotation_protocol') != cfg.get('annotation_protocol'):
        raise ValueError('Annotation protocol differs from prepared data.')
    for key in ('classes', 'source_label_map', 'annotation_fraction', 'ntl_scalar', 'seed'):
        # JSON turns integer mapping keys into strings.
        if json.dumps(prepared_cfg[key], sort_keys=True) != json.dumps(cfg[key], sort_keys=True):
            if key != 'source_label_map' or {str(k):v for k,v in cfg[key].items()} != prepared_cfg[key]:
                raise ValueError(f'Configuration differs from prepared data for {key}. Prepare a separate data directory.')
    model = KGVLMap(cfg).to(device)
    graph = graph_from_config(cfg, device)
    criterion = KGVLMapLoss(train_data.stats['class_weights'], cfg['gamma'], cfg['lambda_topo'], cfg['lambda_picr'], cfg['graph_variant']).to(device)
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if p.requires_grad:
            (no_decay if p.ndim <= 1 or name.endswith('.bias') else decay).append(p)
    optimizer = torch.optim.AdamW([{'params': decay, 'weight_decay': cfg['weight_decay']},
                                 {'params': no_decay, 'weight_decay': 0}], lr=cfg['learning_rate'], betas=(.9, .999), eps=1e-8)
    scaler = torch.cuda.amp.GradScaler(enabled=cfg['amp'])
    start_epoch, global_step, best = 0, 0, -1.0
    if args.resume:
        saved = torch.load(args.resume, map_location='cpu', weights_only=False)
        if saved['config']['classes'] != cfg['classes']:
            raise ValueError('Cannot resume a checkpoint with different classes.')
        if saved['manifest_sha256'] != sha256_file(Path(cfg['prepared_dir']) / 'manifest.json'):
            raise ValueError('Checkpoint/data manifest mismatch.')
        model.load_state_dict(saved['model'])
        optimizer.load_state_dict(saved['optimizer'])
        scaler.load_state_dict(saved['scaler'])
        graph.load_state_dict(saved['graph'])
        start_epoch, global_step, best = saved['epoch'] + 1, saved['global_step'], saved['best_macro_f1']
        torch.set_rng_state(saved['torch_rng'])
        torch.cuda.set_rng_state_all(saved['cuda_rng'])
        np.random.set_state(saved['numpy_rng'])
        random.setstate(saved['python_rng'])
    train_loader = make_loader(train_data, cfg, True)
    val_loader = make_loader(val_data, cfg, False)
    accumulation = 2 if args.smoke else cfg['accumulation_steps']
    steps_per_epoch = math.ceil(len(train_loader) / cfg['accumulation_steps'])
    metadata = {'config': cfg, 'started_at': datetime.datetime.now().isoformat(), 'pid': os.getpid(),
        'gpu': torch.cuda.get_device_name(0), 'torch': torch.__version__, 'train_tiles': len(train_data), 'val_tiles': len(val_data),
        'total_instantiated_parameters': sum(p.numel() for p in model.parameters()),
        'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
        'note': 'Frozen RemoteCLIP text outputs are cached; unused RemoteCLIP visual tower is not instantiated during training.',
        'manifest_sha256': sha256_file(Path(cfg['prepared_dir']) / 'manifest.json'), 'smoke_test': args.smoke}
    dump_json(run / 'run_config.json', metadata)
    print(json.dumps(metadata, indent=2), flush=True)
    writer = SummaryWriter(str(run / 'tensorboard'))
    wall_start = time.time()
    validation_state = validation_summary(epoch_history(run))
    try:
        for epoch in range(start_epoch, start_epoch + 1 if args.smoke else cfg['epochs']):
            model.train()
            iterator = iter(train_loader)
            epoch_loss, n_micro, update_idx = 0.0, 0, 0
            while True:
                window = list(itertools.islice(iterator, accumulation))
                if not window:
                    break
                # Normalize losses over this complete accumulation window, including sparse labels.
                counts = []
                for item in window:
                    ok = item['valid'].all(-1) & item['complete']
                    counts.append((int((item['target'] != 255).sum()), int((ok & (item['instance_label'] == 0)).sum()), int(ok.sum())))
                denominator = np.maximum(np.sum(counts, axis=0), 1)
                optimizer.zero_grad(set_to_none=True)
                lr = learning_rate(global_step, steps_per_epoch, cfg)
                for group in optimizer.param_groups:
                    group['lr'] = lr
                step_stats = {'loss': 0., 'seg': 0., 'topo': 0., 'picr': 0.}
                for cpu_batch, local_counts in zip(window, counts):
                    cpu_batch['loss_scales'] = tuple(np.array(local_counts) / denominator)
                    batch = to_device(cpu_batch, device)
                    with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=cfg['amp']):
                        output = model(batch)
                        loss, log = criterion(output, batch, graph)
                    if not torch.isfinite(loss):
                        raise FloatingPointError(f'Nonfinite loss for {batch["id"]}: {log}')
                    scaler.scale(loss).backward()
                    epoch_loss += log['loss']
                    n_micro += 1
                    for key in step_stats:
                        step_stats[key] += log[key]
                    del batch, output, loss
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], cfg['gradient_clip'])
                if not torch.isfinite(grad_norm):
                    # AMP can overflow early; GradScaler skips this update and lowers its scale.
                    print('AMP gradient overflow; GradScaler will skip this update.', flush=True)
                scaler.step(optimizer)
                scaler.update()
                global_step += 1
                update_idx += 1
                status = {'epoch': epoch + 1, 'epochs': cfg['epochs'], 'step': global_step, 'update_in_epoch': update_idx,
                    'updates_per_epoch': steps_per_epoch, 'learning_rate': lr,
                    'loss': step_stats['loss'], 'seg': step_stats['seg'] / len(window),
                    'topo': step_stats['topo'] / len(window), 'picr': step_stats['picr'] / len(window),
                    'grad_norm': float(grad_norm), 'amp_scale': scaler.get_scale(),
                    'gpu_peak_GiB': torch.cuda.max_memory_allocated() / 1024**3,
                    'elapsed_seconds': time.time() - wall_start, 'state': 'training', **validation_state}
                dump_json(run / 'status.json', status)
                with (run / 'steps.jsonl').open('a', encoding='utf-8') as f:
                    f.write(json.dumps(status) + '\n')
                for k in ('loss', 'seg', 'topo', 'picr', 'learning_rate'):
                    writer.add_scalar('train/' + k, status[k], global_step)
                if global_step % cfg['log_every'] == 0 or global_step <= 3:
                    print(f'epoch {epoch+1}/{cfg["epochs"]} update {update_idx}/{steps_per_epoch} loss={status["loss"]:.5f} seg={status["seg"]:.5f} topo={status["topo"]:.5f} picr={status["picr"]:.5f} lr={lr:.2e} peak={status["gpu_peak_GiB"]:.2f}GiB', flush=True)
                if args.smoke:
                    break
            print(f'Running validation for epoch {epoch+1}', flush=True)
            dump_json(run / 'status.json', {**status, 'state': 'validating', **validation_state})
            result = validate(model, val_loader, device, cfg['amp'], 2 if args.smoke else None, classes=cfg['classes'])
            print('Validation:', json.dumps({k:result[k] for k in ('OA', 'Macro-P', 'Macro-R', 'Macro-F1', 'Kappa')}), flush=True)
            dump_json(run / f'validation_epoch_{epoch+1:03d}.json', result)
            validation_state = validation_summary(epoch_history(run))
            dump_json(run / 'status.json', {**status, 'state': 'saving', **validation_state})
            is_best = result['Macro-F1'] > best
            best = max(best, result['Macro-F1'])
            payload = {'model': model.state_dict(), 'graph': graph.state_dict(), 'optimizer': optimizer.state_dict(),
                'scaler': scaler.state_dict(), 'epoch': epoch, 'global_step': global_step, 'best_macro_f1': best,
                'config': cfg, 'manifest_sha256': metadata['manifest_sha256'], 'metrics': result,
                'torch_rng': torch.get_rng_state(), 'cuda_rng': torch.cuda.get_rng_state_all(),
                'numpy_rng': np.random.get_state(), 'python_rng': random.getstate()}
            atomic_checkpoint(run / 'last.pt', payload)
            if is_best:
                atomic_checkpoint(run / 'best.pt', payload)
            for key in ('OA', 'Macro-P', 'Macro-R', 'Macro-F1', 'Kappa'):
                writer.add_scalar('validation/' + key, result[key], epoch + 1)
            writer.flush()
        dump_json(run / 'status.json', {'state': 'complete', 'last_epoch': epoch + 1, 'best_macro_f1': best,
                                       'elapsed_seconds': time.time() - wall_start, 'smoke_test': args.smoke, **validation_state})
    except Exception as exc:
        dump_json(run / 'status.json', {'state': 'failed', 'error': repr(exc), 'step': global_step})
        raise
    finally:
        writer.close()


if __name__ == '__main__':
    main()
