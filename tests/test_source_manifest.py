"""Source-path and coordinate checks for the confirmed seven-class data."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import rasterio
from rasterio.transform import from_origin

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kmap.data import source_path, read_aligned
from prepare_data import prepare_tile


class SourceManifestTests(unittest.TestCase):
    def test_geographic_grid_shift_is_not_ignored(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'geographic_label.tif'
            source = from_origin(121, 31, 0.00001, 0.00001)
            destination = from_origin(121.000008, 31, 0.00001, 0.00001)
            values = np.tile(np.arange(4, dtype=np.uint8), (4, 1))
            with rasterio.open(path, 'w', driver='GTiff', width=4, height=4,
                               count=1, dtype='uint8', crs='EPSG:4326', transform=source) as dst:
                dst.write(values, 1)
            result, valid = read_aligned(path, {'crs': rasterio.crs.CRS.from_epsg(4326),
                                                'transform': destination, 'shape': (4, 4)}, 1, True)
            self.assertTrue(np.array_equal(result[0, :, :3], values[:, 1:]))
            self.assertTrue((~valid[0, :, 3]).all())

    def test_in_place_sources_label_codes_and_label_grid_geometry(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / 'cache' / 'tiles').mkdir(parents=True)
            rgb_transform = from_origin(500000, 3500000, 1, 1)
            label_transform = from_origin(500002, 3500000, 1, 1)

            def write(name, data, transform):
                path = root / (name + '.tif')
                with rasterio.open(path, 'w', driver='GTiff', height=32, width=32,
                                   count=data.shape[0], dtype=str(data.dtype),
                                   crs='EPSG:32650', transform=transform) as dst:
                    dst.write(data)
                return str(path)

            label = np.zeros((1, 32, 32), np.uint8)
            label[0, 4:12, 4:12] = 8
            label[0, 16:24, 16:24] = 7
            paths = {
                'rgb': write('arbitrary_image_name', np.full((3, 32, 32), 128, np.uint8), rgb_transform),
                'height': write('height_other_name', np.full((1, 32, 32), 12, np.float32), rgb_transform),
                'ntl': write('light_other_name', np.full((3, 32, 32), 40, np.uint8), rgb_transform),
                'label': write('correct_relabel', label, label_transform),
                'geometry': str(root / 'geometry.json'),
            }
            points = [[[4, 4], [12, 4], [12, 12], [4, 12]],
                      [[16, 16], [24, 16], [24, 24], [16, 24]]]
            doc = {'objects': [{'category': 'DO NOT USE', 'segmentation': p} for p in points]}
            Path(paths['geometry']).write_text(json.dumps(doc), encoding='utf-8')
            cfg = {'prepared_dir': str(root / 'cache'), 'data_root': str(root / 'nonexistent'),
                   'source_label_map': {**{i: i for i in range(8)}, 8: 255, 255: 255},
                   'classes': list(range(7)), 'ntl_channels': 3, 'ntl_scalar': 'luminance_rgb',
                   'minimum_attribute_coverage': .5, 'annotation_fraction': 1}
            rec = {'id': 'HF_unique_id', 'image_file': 'does_not_exist.tif', 'paths': paths,
                   'geometry_reference_path': paths['label'], 'split': 'train'}
            self.assertEqual(source_path(cfg, rec, 'rgb'), Path(paths['rgb']))
            result = prepare_tile((cfg, rec))
            with np.load(root / 'cache' / 'tiles' / 'HF_unique_id.npz') as z:
                first = {k: z[k].copy() for k in z.files}
            self.assertEqual(first['target'][5, 7], 255)  # Unknown function, not background.
            self.assertEqual(first['target'][17, 19], 7)  # Industrial stays class 7.
            self.assertEqual(first['instances'][5, 4], 0)  # Label grid is two pixels east of RGB.
            self.assertGreater(first['instances'][5, 6], 0)
            self.assertEqual(first['target'][0, 0], 255)  # Outside label coverage is not valid background.
            self.assertEqual(first['target'][0, 3], 0)
            self.assertTrue(np.allclose(result['attrs'][:, 0], 12))
            self.assertTrue(np.allclose(result['attrs'][:, 1], 64))
            doc['objects'][0]['category'] = 'A DIFFERENT HIDDEN FUNCTION'
            doc['img_description_eg'] = 'IGNORE THIS DESCRIPTION'
            Path(paths['geometry']).write_text(json.dumps(doc), encoding='utf-8')
            prepare_tile((cfg, rec))
            with np.load(root / 'cache' / 'tiles' / 'HF_unique_id.npz') as z:
                for key, value in first.items():
                    self.assertTrue(np.array_equal(value, z[key]), key)


if __name__ == '__main__':
    unittest.main()
