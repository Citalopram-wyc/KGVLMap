"""Check every prepared tile and independently verify annotation and graph inputs."""
import argparse
import collections
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from kmap.common import load_config, dump_json, sha256_file
from kmap.data import normalization
from kmap.graph import PhysicalKnowledgeGraph


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--wait', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(1)
    cfg = load_config(args.config)
    folder = Path(cfg['prepared_dir'])
    source = Path(cfg['source_manifest'])
    catalog = json.loads(source.read_text(encoding='utf-8'))
    policy = json.loads((source.parent/'annotation_policy.json').read_text(encoding='utf-8'))
    budget = json.loads((source.parent/'summary.json').read_text(encoding='utf-8'))
    hidden = np.asarray(sorted(policy['hidden_building_ids']), dtype=np.uint64)
    records = [r for r in catalog['records'] if r['split'] in ('train','val','test')]
    expected_ids = {r['id'] for r in records}
    pixels = collections.defaultdict(lambda: np.zeros(10, np.int64))
    unique = collections.defaultdict(set)
    chosen = {}
    appearances = 0
    empty_tiles = 0
    representative = {}
    started = time.monotonic()

    def wait_for(path):
        while not path.exists():
            if not args.wait:
                raise FileNotFoundError(path)
            time.sleep(2)

    for i, rec in enumerate(records):
        summary_path = folder/'tile_statistics'/f'{rec["id"]}.npz'
        wait_for(summary_path)  # Summary is committed only after the tile is complete.
        with np.load(folder/'tiles'/f'{rec["id"]}.npz', allow_pickle=False) as z:
            arrays = {k:z[k] for k in z.files}  # Also checks every compressed member's CRC.
        with np.load(summary_path, allow_pickle=False) as z:
            summary = {k:z[k] for k in z.files}
        ids, target = arrays['instances'], arrays['target']
        image_valid = arrays['image_valid']
        gids, labels = arrays['global_instance_ids'], arrays['instance_label']
        attrs, valid, complete = arrays['attributes'], arrays['valid'], arrays['complete']
        n = len(gids)
        assert ids.shape == target.shape == image_valid.shape == (cfg['crop_size'],cfg['crop_size']), rec['id']
        assert attrs.shape == valid.shape == (n,3)
        assert labels.shape == complete.shape == (n,)
        assert np.isfinite(attrs).all() and (attrs >= 0).all()
        assert valid[:,1].all() and (attrs[:,1] > 0).all()
        assert np.all(gids[1:] > gids[:-1])
        assert ids.min() >= 0 and ids.max() <= n
        assert np.bincount(ids.ravel(), minlength=n+1)[1:].min(initial=1) > 0
        assert (ids[~image_valid] == 0).all()
        assert (target[~image_valid] == 255).all()
        assert np.isfinite(arrays['distance']).all()
        assert (np.abs(arrays['distance']) <= 1).all()
        lookup = np.r_[0, np.where(labels > 0, labels, 255)].astype(np.uint8)
        expected = lookup[ids]
        expected[~image_valid] = 255
        np.testing.assert_array_equal(target, expected, err_msg=rec['id'])
        np.testing.assert_array_equal(summary['attrs'],attrs)
        np.testing.assert_array_equal(summary['labels'],labels)
        counts = np.bincount(target[target != 255], minlength=8)
        np.testing.assert_array_equal(summary['counts'],counts)
        masked = np.isin(gids,hidden)
        if rec['split'] == 'train':
            np.testing.assert_array_equal(labels == 0,masked,err_msg=rec['id'])
            appearances += n
            for j, gid in enumerate(gids.tolist()):
                old = chosen.get(gid)
                rank = (bool(complete[j]), int(valid[j].sum()))
                if old is not None:
                    assert old[3] == labels[j], f'Global label conflict: {gid}'
                if old is None or rank > old[4]:
                    chosen[gid] = (attrs[j].copy(),valid[j].copy(),complete[j],int(labels[j]),rank)
        else:
            assert not masked.any(), 'Hidden training IDs overlap evaluation buildings.'
            assert (labels > 0).all(), 'Evaluation labels were hidden.'
        unique[rec['split']].update(gids.tolist())
        pixels[rec['split']][:8] += counts
        hidden_lut = np.r_[False,labels == 0]
        pixels[rec['split']][8] += hidden_lut[ids].sum()
        pixels[rec['split']][9] += (~image_valid).sum()
        if not n:
            empty_tiles += 1
            representative.setdefault('empty',rec['id'])
        if (~image_valid).any():
            representative.setdefault('missing_image',rec['id'])
        if (labels == 0).any():
            representative.setdefault('unlabeled',rec['id'])
        if rec.get('district') == 'CH':
            representative.setdefault('chaohu_2m',rec['id'])
        representative.setdefault(rec['split'],rec['id'])
        if (i+1)%500 == 0 or i+1 == len(records):
            print(f'Audited {i+1}/{len(records)} tiles',flush=True)
            dump_json(folder/'audit_progress.json',{'phase':'checking_tiles','completed':i+1,'total':len(records)})

    assert {p.stem for p in (folder/'tiles').glob('*.npz')} == expected_ids
    assert {p.stem for p in (folder/'tile_statistics').glob('*.npz')} == expected_ids
    assert not unique['train'] & unique['val']
    assert len(unique['train']) == policy['training_buildings']
    train_counts = pixels['train']
    for i, code in enumerate([*range(9),255]):
        assert int(train_counts[i]) == budget['training_pixel_counts'][str(code)], (code,int(train_counts[i]))
    wait_for(folder/'manifest.json')
    manifest = json.loads((folder/'manifest.json').read_text(encoding='utf-8'))
    stats = json.loads((folder/'statistics.json').read_text(encoding='utf-8'))
    rows = [chosen[k] for k in sorted(chosen)]
    attrs = np.stack([r[0] for r in rows])
    valid = np.stack([r[1] for r in rows])
    complete = np.asarray([r[2] for r in rows],bool)
    labels = np.asarray([r[3] for r in rows])
    assert np.count_nonzero(labels == 0) == policy['hidden_buildings']
    for d in range(3):
        percentiles = np.percentile(np.log1p(attrs[valid[:,d] & complete,d]),[1,99,50])
        np.testing.assert_allclose(percentiles,[stats['attributes'][k][d] for k in ('lo','hi','median')])
    np.testing.assert_array_equal(stats['class_counts'],train_counts[:8])
    np.testing.assert_allclose(stats['class_weights'],np.median(train_counts[:8])/train_counts[:8])
    eligible = complete & valid.all(1)
    visible = eligible & (labels > 0)
    values = normalization(attrs,valid,stats['attributes'])
    graph = PhysicalKnowledgeGraph(7,cfg['graph_rho'],cfg['graph_steps'],cfg['graph_smoothing'])
    actual = torch.load(folder/'graph.pt',map_location='cpu',weights_only=True)
    # Rebuild from independently checked visible labels; no hidden truth is loaded.
    graph.fit(values[eligible],values[visible],labels[visible])
    for key,value in graph.state_dict().items():
        torch.testing.assert_close(value,actual[key],rtol=1e-6,atol=1e-7)
    expected_counts = np.zeros((7,9),np.int64)
    for d in range(3):
        state = np.searchsorted(actual['tertiles'][d].numpy(),values[visible,d],side='right')
        # Use double-precision quantiles as in fitting, avoiding a float32 boundary tie.
        cuts = np.quantile(values[eligible].astype(np.float64),[1/3,2/3],axis=0).T[d]
        state = np.searchsorted(cuts,values[visible,d],side='right')
        np.add.at(expected_counts,(labels[visible]-1,3*d+state),1)
    smooth = cfg['graph_smoothing']
    expected_edges = (expected_counts+smooth)/(expected_counts.sum(1,keepdims=True)+9*smooth)
    np.testing.assert_allclose(actual['associations'],expected_edges,rtol=1e-6,atol=1e-7)
    priors, reliability = graph(torch.from_numpy(values[eligible & (labels == 0)]))
    assert torch.isfinite(priors).all() and torch.isfinite(reliability).all()
    assert (priors[:,0] == 0).all()
    torch.testing.assert_close(priors.sum(1),torch.ones(len(priors)),rtol=1e-5,atol=1e-6)
    assert (reliability >= 0).all() and (reliability <= 1).all()
    assert manifest['training_instances'] == len(chosen)
    assert manifest['training_tile_instance_occurrences'] == appearances
    assert manifest['labeled_graph_instances'] == int(visible.sum())
    assert manifest['unlabeled_valid_instances'] == int((eligible & (labels == 0)).sum())
    report = dict(status='passed',tiles_checked=len(records),split_counts=manifest['split_counts'],
        empty_tiles=empty_tiles,unique_buildings={k:len(v) for k,v in unique.items()},
        training_buildings=len(chosen),visible_buildings=int((labels>0).sum()),hidden_buildings=int((labels==0).sum()),
        training_pixel_counts={str(c):int(train_counts[i]) for i,c in enumerate([*range(9),255])},
        eligible_visible_graph_buildings=int(visible.sum()),eligible_unlabeled_distillation_buildings=int((eligible & (labels==0)).sum()),
        incomplete_training_buildings=int((~complete).sum()),complete_but_missing_physical_attributes=int((complete & ~valid.all(1)).sum()),
        graph_visible_class_counts=np.bincount(labels[visible],minlength=8)[1:].tolist(),
        class_order=cfg['classes'],no_train_validation_building_overlap=True,
        raw_255_missing_image_and_raw_8_unlabeled_distinction_verified=True,
        hidden_ground_truth_used_for_graph_or_statistics=False,source_manifest_sha256=sha256_file(source),
        prepared_manifest_sha256=sha256_file(folder/'manifest.json'),graph_sha256=sha256_file(folder/'graph.pt'),
        representative_tiles=representative,elapsed_seconds=time.monotonic()-started)
    dump_json(folder/'audit_report.json',report)
    dump_json(folder/'audit_progress.json',{'phase':'complete','completed':len(records),'total':len(records)})
    print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)


if __name__ == '__main__':
    main()
