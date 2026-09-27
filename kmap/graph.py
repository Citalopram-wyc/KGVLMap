"""Fixed category--attribute graph, equations (11)--(14) of the manuscript."""
import numpy as np
import torch
from torch import nn


class PhysicalKnowledgeGraph(nn.Module):
    def __init__(self, classes, rho=0.85, steps=10, smoothing=1.0):
        super().__init__()
        self.classes, self.rho, self.steps, self.smoothing = classes, rho, steps, smoothing
        self.register_buffer('operator', torch.zeros(classes + 9, classes + 9))
        self.register_buffer('associations', torch.zeros(classes, 9))
        self.register_buffer('centers', torch.zeros(3, 3))
        self.register_buffer('widths', torch.ones(3, 3))
        self.register_buffer('tertiles', torch.zeros(3, 2))

    @torch.no_grad()
    def fit(self, all_train_values, labeled_values, labels):
        """Statistics use training observations; edges use visible training labels only."""
        x = np.asarray(all_train_values, dtype=np.float64)
        v = np.asarray(labeled_values, dtype=np.float64)
        labels = np.asarray(labels, dtype=np.int64)
        if len(x) == 0 or len(v) == 0:
            raise ValueError('UPKG requires valid training observations and visible labels.')
        cuts = np.quantile(x, [1 / 3, 2 / 3], axis=0).T
        means, widths = np.zeros((3, 3)), np.zeros((3, 3))
        counts = np.zeros((self.classes, 9))
        for d in range(3):
            state = np.searchsorted(cuts[d], x[:, d], side='right')
            labeled_state = np.searchsorted(cuts[d], v[:, d], side='right')
            for s in range(3):
                z = x[state == s, d]
                means[d, s] = z.mean() if len(z) else np.quantile(x[:, d], (s + 0.5) / 3)
                widths[d, s] = max(float(z.std()) if len(z) > 1 else float(x[:, d].std()), 0.03)
            np.add.at(counts, (labels - 1, 3 * d + labeled_state), 1)
        edges = (counts + self.smoothing) / (counts.sum(1, keepdims=True) + 9 * self.smoothing)
        a = np.zeros((self.classes + 9, self.classes + 9))
        a[:self.classes, self.classes:] = edges
        a[self.classes:, :self.classes] = edges.T
        inv = 1 / np.sqrt(np.maximum(a.sum(1), 1e-12))
        self.operator.copy_(torch.from_numpy(a * inv[:, None] * inv[None, :]))
        self.associations.copy_(torch.from_numpy(edges))
        self.centers.copy_(torch.from_numpy(means))
        self.widths.copy_(torch.from_numpy(widths))
        self.tertiles.copy_(torch.from_numpy(cuts))

    @torch.no_grad()
    def forward(self, values, variant='full'):
        v = values.float()
        log_m = -0.5 * ((v[:, :, None] - self.centers) / self.widths.clamp_min(1e-4)).square()
        membership = log_m.softmax(dim=-1).flatten(1) / 3.0
        x0 = torch.cat([v.new_zeros((len(v), self.classes)), membership], dim=1)
        x = x0
        for _ in range(self.steps):
            x = self.rho * (x @ self.operator.T) + (1 - self.rho) * x0
        scores = x[:, :self.classes]
        total = scores.sum(-1, keepdim=True)
        prior = torch.where(total > 1e-12, scores / total.clamp_min(1e-12), torch.full_like(scores, 1 / self.classes))
        reliability = (1 + (prior * prior.clamp_min(1e-12).log()).sum(-1) / np.log(self.classes)).clamp(0, 1)
        reliability = torch.where(total[:, 0] > 1e-12, reliability, 0)
        # Control variants keep the reliability of the original full prior fixed.
        if variant == 'uniform':
            prior = torch.full_like(prior, 1 / self.classes)
        elif variant == 'one_step':
            scores = (x0 @ self.operator.T)[:, :self.classes]
            prior = scores / scores.sum(-1, keepdim=True).clamp_min(1e-12)
        elif variant == 'direct':
            scores = membership @ self.associations.T
            prior = scores / scores.sum(-1, keepdim=True).clamp_min(1e-12)
        elif variant not in ('full', 'foreground', 'none'):
            raise ValueError(variant)
        return torch.cat([v.new_zeros((len(v), 1)), prior], -1).detach(), reliability.detach()
