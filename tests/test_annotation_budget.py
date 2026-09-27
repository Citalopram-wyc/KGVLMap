"""Hidden functional labels and missing imagery must remain semantically distinct."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import rasterio
from rasterio.transform import from_origin
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tools'))
from prepare_data import prepare_tile, prepare_cached_tile
from kmap.losses import KGVLMapLoss
from kmap.data import BuildingDataset
from kmap.hidden_diagnostic import HiddenTrainingDiagnostic
from create_annotation_budget import stratified_selection


class AnnotationBudgetTests(unittest.TestCase):
    def test_fixed_stratified_global_ids(self):
        classes = {c: list(range(c*1000,c*1000+13+c)) for c in range(1,8)}
        first, allocated = stratified_selection(classes,.2,42)
        second, _ = stratified_selection(classes,.2,42)
        self.assertEqual(first, second)
        self.assertEqual(len(first), round(sum(map(len,classes.values()))*.2))
        self.assertEqual(len(first), len(set(first)))
        for c, ids in classes.items():
            self.assertEqual(len(set(ids)&set(first)), allocated[c])
            self.assertLess(abs(allocated[c]-.2*len(ids)), 1)

    def test_no_image_255_is_not_unlabeled_building_8(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root/'cache'/'tiles').mkdir(parents=True)
            transform = from_origin(500000,3500000,1,1)
            def write(name, data, nodata=None):
                path = root/f'{name}.tif'
                if data.ndim == 2:
                    data = data[None]
                with rasterio.open(path,'w',driver='GTiff',width=32,height=32,count=len(data),
                                   dtype=data.dtype,crs='EPSG:32650',transform=transform,nodata=nodata) as dst:
                    dst.write(data)
                return str(path)
            ids = np.zeros((32,32),np.uint32)
            ids[3:9,3:9] = 101
            ids[17:25,17:25] = 9999
            ids[18:25,:3] = 30
            label = np.zeros((32,32),np.uint8)
            label[ids==101] = 8
            label[ids==9999] = 7
            label[ids==30] = 8
            valid = np.ones((32,32),np.uint8)
            valid[:2] = 0
            label[valid==0] = 255
            height = np.full((1,32,32),12,np.float32)
            height[:,:2] = 999999
            metadata = [dict(building_id=bid,full_area_m2=area,function_code=2,function_name='HIDDEN TRUTH DO NOT READ')
                        for bid,area in [(30,100),(101,36),(9999,64)]]
            meta_path = root/'instances.json'
            meta_path.write_text(json.dumps(metadata),encoding='utf-8')
            paths = dict(rgb=write('rgb',np.full((3,32,32),128,np.uint8)),
                         height=write('height',height),ntl=write('ntl',np.full((3,32,32),40,np.uint8)),
                         label=write('label',label,255),instance_id=write('ids',ids,0),
                         valid_mask=write('valid',valid),instances=str(meta_path))
            cfg = dict(prepared_dir=str(root/'cache'),data_root=str(root),classes=list(range(7)),
                       source_label_map={**{i:i for i in range(8)},8:255,255:255},
                       ntl_channels=3,ntl_scalar='luminance_rgb',minimum_attribute_coverage=.5,
                       annotation_fraction=.8,annotation_protocol='predefined_global_ids',seed=42)
            record = dict(id='HF_test',image_file='rgb.tif',split='train',geometry_format='instance_raster',paths=paths)
            result = prepare_tile((cfg,record))
            with np.load(root/'cache/tiles/HF_test.npz') as saved:
                arrays = {k:saved[k].copy() for k in saved.files}
            np.testing.assert_array_equal(arrays['global_instance_ids'],[30,101,9999])
            np.testing.assert_array_equal(arrays['instance_label'],[0,0,7])
            np.testing.assert_array_equal(arrays['complete'],[False,True,True])
            self.assertTrue((arrays['target'][:2]==255).all())
            self.assertTrue((arrays['instances'][:2]==0).all())
            self.assertEqual(arrays['target'][4,4],255)
            self.assertGreater(arrays['instances'][4,4],0)
            self.assertTrue(np.allclose(result['attrs'][:,0],12))
            self.assertTrue(np.isnan(result['height_sample'][0]))
            for item in metadata:
                item['function_code'] = 5
                item['function_name'] = 'DIFFERENT HIDDEN CLASS'
            meta_path.write_text(json.dumps(metadata),encoding='utf-8')
            prepare_tile((cfg,record))
            with np.load(root/'cache/tiles/HF_test.npz') as saved:
                for key,value in arrays.items():
                    np.testing.assert_array_equal(saved[key],value)

            (root/'cache/tile_statistics').mkdir()
            cached = prepare_cached_tile((cfg,record,'test-fingerprint'))
            with patch('prepare_data.prepare_tile', side_effect=AssertionError('Cache hit must not regenerate.')):
                resumed = prepare_cached_tile((cfg,record,'test-fingerprint'))
            for key in cached:
                np.testing.assert_array_equal(cached[key],resumed[key])
            meta_path.write_text(json.dumps(metadata)+' ',encoding='utf-8')
            with patch('prepare_data.prepare_tile', wraps=prepare_tile) as regenerate:
                prepare_cached_tile((cfg,record,'test-fingerprint'))
                regenerate.assert_called_once()

            # Auxiliary gradients reach valid unlabeled building 101, never no-image
            # pixels or clipped building 30; both raw 8 and raw 255 ignore segmentation.
            logits = torch.zeros(1,8,32,32,requires_grad=True)
            batch = {k:torch.from_numpy(arrays[k]) for k in ('target','instances','valid','complete','instance_label')}
            batch['target'] = batch['target'][None]
            batch['instances'] = batch['instances'][None]
            batch['attrs'] = torch.zeros(3,3)
            def teacher(values, variant):
                return torch.cat([torch.zeros(len(values),1),torch.full((len(values),7),1/7)],1),torch.ones(len(values))
            criterion = KGVLMapLoss([0]*8,lambda_topo=1,lambda_picr=0)
            loss, _ = criterion(dict(logits=logits,ntl_pred=torch.zeros(3,requires_grad=True)),batch,teacher)
            loss.backward()
            self.assertEqual(float(logits.grad[:,:,:2].abs().sum()),0)
            self.assertGreater(float(logits.grad[:,:,3:9,3:9].abs().sum()),0)
            self.assertEqual(float(logits.grad[:,:,18:25,:3].abs().sum()),0)

            full_truth = label.copy()
            full_truth[ids==101] = 2
            full_truth[ids==30] = 1
            truth_path = write('truth_for_diagnostic_only',full_truth,255)
            (root/'cache/manifest.json').write_text(json.dumps(dict(records=[record],config=cfg)),encoding='utf-8')
            stats = dict(attributes=dict(lo=[0,0,0],hi=[4,10,5],median=[1,1,1]),height_log_p99=5,ntl_p99=[255]*3)
            (root/'cache/statistics.json').write_text(json.dumps(stats),encoding='utf-8')
            policy_path = root/'policy.json'
            policy_path.write_text(json.dumps(dict(seed=42,visible_fraction=.8)),encoding='utf-8')
            diagnostic = root/'diagnostic.json'
            diagnostic.write_text(json.dumps(dict(annotation_policy=str(policy_path),records=[dict(id=record['id'],full_truth_label=truth_path,valid_mask=paths['valid_mask'])])),encoding='utf-8')
            training_item = BuildingDataset(cfg)[0]
            diagnostic_item = HiddenTrainingDiagnostic(cfg,diagnostic)[0]
            for key in ('rgb','height','ntl','morphology','instances','attrs','valid','complete','instance_label'):
                self.assertTrue(torch.equal(training_item[key],diagnostic_item[key]),key)
            self.assertEqual(int(diagnostic_item['target'][4,4]),2)
            self.assertEqual(int(diagnostic_item['target'][19,1]),1)
            self.assertEqual(int(diagnostic_item['target'][19,19]),255)
            self.assertTrue((diagnostic_item['target'][:2]==255).all())


if __name__ == '__main__':
    unittest.main()
