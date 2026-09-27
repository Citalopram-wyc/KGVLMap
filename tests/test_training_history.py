"""Per-epoch display must use completed validation results, never training loss."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/'tools'))
from kmap.training_history import epoch_history, validation_summary
from training_dashboard import snapshot


class TrainingHistoryTests(unittest.TestCase):
    def test_waiting_for_first_epoch_is_not_zero_accuracy(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root/'status.json').write_text(json.dumps(dict(epoch=1,epochs=100,state='training',
                update_in_epoch=100,updates_per_epoch=835,loss=.05,grad_norm=float('nan'))))
            result = snapshot(root)
            self.assertEqual(result['history'],[])
            self.assertIsNone(result['latest_validation'])
            self.assertIsNone(result['current']['grad_norm'])
            json.dumps(result,allow_nan=False)

    def test_sorted_complete_results_and_best_f1(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            def write(epoch, oa, f1):
                data = {'OA':oa,'Macro-P':40.,'Macro-R':60.,'Macro-F1':f1,'Kappa':.3,'seconds':90,'pixels':100}
                (root/f'validation_epoch_{epoch:03d}.json').write_text(json.dumps(data))
            write(10,80.,55.)
            write(2,82.,50.)
            (root/'validation_epoch_003.json').write_text('{')
            (root/'validation_epoch_004.json.tmp').write_text('{}')
            history = epoch_history(root)
            self.assertEqual([r['epoch'] for r in history],[2,10])
            summary = validation_summary(history)
            self.assertEqual(summary['best_validation']['epoch'],10)
            self.assertEqual(history[0]['OA'],82.)  # Already percent; no second x100.
            (root/'status.json').write_text(json.dumps(dict(epoch=11,state='training',update_in_epoch=835,updates_per_epoch=835)))
            result = snapshot(root)
            self.assertEqual(result['phase'],'awaiting_validation')
            self.assertEqual(result['latest_validation']['epoch'],10)


if __name__ == '__main__':
    unittest.main()
