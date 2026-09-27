import numpy as np
import torch


class SegmentationMetrics:
    """One global confusion matrix including background; macro metrics exclude it."""
    def __init__(self, classes):
        self.classes = classes
        self.matrix = np.zeros((classes, classes), dtype=np.int64)

    def update(self, pred, target):
        if isinstance(pred, torch.Tensor):
            valid = (target >= 0) & (target < self.classes)
            flat = self.classes * target[valid].long() + pred[valid].long()
            cm = torch.bincount(flat, minlength=self.classes ** 2).reshape(self.classes, self.classes)
            self.matrix += cm.cpu().numpy()
        else:
            valid = (target >= 0) & (target < self.classes)
            self.matrix += np.bincount(self.classes * target[valid].astype(np.int64) + pred[valid], minlength=self.classes ** 2).reshape(self.classes, self.classes)

    def compute(self):
        cm = self.matrix.astype(np.float64)
        rows, cols, tp = cm.sum(1), cm.sum(0), cm.diagonal()
        safe = lambda x, y: np.divide(x, y, out=np.zeros_like(x), where=y > 0)
        p, r = safe(tp, cols), safe(tp, rows)
        f1 = safe(2 * p * r, p + r)
        iou = safe(tp, rows + cols - tp)
        n = cm.sum()
        oa = tp.sum() / n if n else 0.0
        pe = np.dot(rows, cols) / n ** 2 if n else 0.0
        return {'OA': float(100 * oa), 'Macro-P': float(100 * p[1:].mean()),
                'Macro-R': float(100 * r[1:].mean()), 'Macro-F1': float(100 * f1[1:].mean()),
                'Kappa': float((oa - pe) / (1 - pe)) if pe < 1 else 0.0,
                'mIoU-functions': float(100 * iou[1:].mean()), 'pixels': int(n),
                'precision_per_class': p.tolist(), 'recall_per_class': r.tolist(),
                'f1_per_class': f1.tolist(), 'iou_per_class': iou.tolist(),
                'confusion_matrix': self.matrix.tolist(),
                'row_percentages': (100 * safe(cm, rows[:, None])).tolist()}
