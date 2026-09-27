"""Scientific invariants that catch changes to the method, not just tensor shapes."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import unittest
import numpy as np
import torch
from kmap.graph import PhysicalKnowledgeGraph
from kmap.losses import footprint_average, KGVLMapLoss
from kmap.metrics import SegmentationMetrics
from kmap.model import PhysicalSemanticAlignment, SemanticModalityRouter, coverage_matrix


class MethodTests(unittest.TestCase):
    def test_coverage_fusion_equals_literal_pixel_assembly(self):
        torch.manual_seed(2)
        ids = torch.tensor([[[0,0,1,1],[0,1,1,1],[2,2,2,1],[2,2,0,0]]])
        router = SemanticModalityRouter(8, 4)
        cond = torch.randn(2, 8, requires_grad=True)
        maps = [torch.randn(1, 4, 1, 1, requires_grad=True) for _ in range(4)]
        fused, gates, beta = router(maps, cond, ids)
        expected = (ids == 0).float().mean() * maps[0]
        for j in range(2):
            local = beta[j] * gates[j, 0, :, None, None] * maps[0]
            local += (1 - beta[j]) * sum(gates[j, m, :, None, None] * maps[m] for m in range(1, 4))
            expected = expected + (ids == j+1).float().mean() * local
        self.assertTrue(torch.allclose(fused, expected, atol=1e-6))
        fused.sum().backward()
        self.assertGreater(cond.grad.abs().sum().item(), 0)
        self.assertTrue(torch.allclose(coverage_matrix(ids, 2).to_dense().sum(1), torch.ones(1)))

    def test_background_is_retained_and_kl_decomposes(self):
        p = torch.tensor([.2, .4, .3, .1], requires_grad=True)
        q = torch.tensor([0., .2, .3, .5])
        kl = (q * (q.clamp_min(1e-12).log() - p.log())).sum()
        cond = p[1:] / p[1:].sum()
        decomposed = (q[1:] * (q[1:].log() - cond.log())).sum() - p[1:].sum().log()
        self.assertTrue(torch.allclose(kl, decomposed))
        logits = p.log()[None, :, None, None].expand(1,4,4,4)
        pooled, size = footprint_average(logits.softmax(1), torch.ones(1,4,4,dtype=torch.long), 1)
        self.assertTrue(torch.allclose(pooled[0], p))
        self.assertEqual(size.item(), 16)

    def test_fixed_graph_and_one_two_step_equivalence(self):
        rng = np.random.default_rng(7)
        x = rng.random((100, 3))
        labels = np.tile(np.arange(1,5),25)
        graph = PhysicalKnowledgeGraph(4)
        graph.fit(x, x, labels)
        self.assertEqual(sum(p.numel() for p in graph.parameters()), 0)
        v = torch.from_numpy(x[:8]).float().requires_grad_()
        graph.steps = 1
        q1, _ = graph(v)
        graph.steps = 2
        q2, _ = graph(v)
        self.assertTrue(torch.allclose(q1,q2,atol=1e-6))
        self.assertFalse(q1.requires_grad)
        self.assertTrue((q1[:,0] == 0).all())
        self.assertTrue(torch.allclose(q1.sum(1),torch.ones(8)))
        graph.steps = 10
        _, r = graph(v)
        qu, ru = graph(v,'uniform')
        self.assertTrue(torch.equal(r,ru))
        self.assertTrue(torch.allclose(qu[:,1:],torch.full((8,4),.25)))

    def test_picr_does_not_use_ntl_token(self):
        torch.manual_seed(3)
        model = PhysicalSemanticAlignment(torch.randn(4,32),32,8,32)
        x = torch.rand(5,3,requires_grad=True)
        _, pred = model(x,torch.ones(5,3,dtype=torch.bool))
        pred.sum().backward()
        self.assertTrue(torch.equal(x.grad[:,2],torch.zeros(5)))
        self.assertGreater(x.grad[:,:2].abs().sum().item(),0)

    def test_masked_attribute_cannot_change_semantic_condition(self):
        torch.manual_seed(4)
        model = PhysicalSemanticAlignment(torch.randn(4,32),32,8,32)
        x = torch.rand(5,3)
        valid = torch.ones(5,3,dtype=torch.bool)
        valid[:,2] = False
        c1,_ = model(x,valid)
        x[:,2] += 100
        c2,_ = model(x,valid)
        self.assertTrue(torch.allclose(c1,c2))

    def test_metrics_include_background_errors_exclude_ignore(self):
        metric = SegmentationMetrics(3)
        metric.update(np.array([0,1,0,1,2,1,2]),np.array([0,0,1,1,2,2,255]))
        result = metric.compute()
        self.assertEqual(result['pixels'],6)
        self.assertAlmostEqual(result['OA'],50)
        self.assertAlmostEqual(result['Macro-R'],50)
        self.assertAlmostEqual(result['Macro-P'],100*(1/3+1)/2)
        self.assertAlmostEqual(result['Macro-F1'],100*(.4+2/3)/2)


if __name__ == '__main__':
    unittest.main()
