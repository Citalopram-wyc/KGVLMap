import torch
from torch import nn
from torch.nn import functional as F


def footprint_average(probabilities, instances, count):
    """Average the full (background included) pixel distribution in each footprint."""
    c = probabilities.shape[1]
    ids = instances.flatten().long()
    vals = probabilities.permute(0, 2, 3, 1).reshape(-1, c).float()
    pooled = vals.new_zeros((count + 1, c)).index_add_(0, ids, vals)
    denom = torch.bincount(ids, minlength=count + 1).float()
    return pooled[1:] / denom[1:, None].clamp_min(1), denom[1:]


class KGVLMapLoss(nn.Module):
    def __init__(self, class_weights, gamma=2, lambda_topo=0.3, lambda_picr=0.1, variant='full'):
        super().__init__()
        self.register_buffer('weights', torch.as_tensor(class_weights, dtype=torch.float32))
        self.gamma, self.lambda_topo, self.lambda_picr, self.variant = gamma, lambda_topo, lambda_picr, variant

    def forward(self, output, batch, graph):
        logits = output['logits'].float()
        target = batch['target'].long()
        mask = target != 255
        logp = F.log_softmax(logits, dim=1)
        zero = logits.sum() * 0
        if mask.any():
            y = target[mask]
            lp = logp.permute(0, 2, 3, 1)[mask].gather(1, y[:, None])[:, 0]
            seg = -(self.weights[y] * (1 - lp.exp()).pow(self.gamma) * lp).mean()
        else:
            seg = zero
        eligible = batch['valid'].all(-1) & batch['complete']
        picr = (output['ntl_pred'][eligible].float() - batch['attrs'][eligible, 2].detach()).square().mean() if eligible.any() else output['ntl_pred'].sum() * 0
        topology = zero
        unlabeled = eligible & (batch['instance_label'] == 0)
        if self.lambda_topo > 0 and self.variant != 'none' and unlabeled.any():
            pooled, pixels = footprint_average(logp.exp(), batch['instances'], len(batch['attrs']))
            use = unlabeled & (pixels > 0)
            q, r = graph(batch['attrs'][use], self.variant)
            p = pooled[use].clamp_min(1e-7)
            if self.variant == 'foreground':
                terms = -p[:, 1:].sum(-1).clamp_min(1e-7).log()
            else:
                terms = (q * (q.clamp_min(1e-7).log() - p.log())).sum(-1)
            topology = (r * terms).mean() if len(terms) else zero
        scale = batch.get('loss_scales', (1.0, 1.0, 1.0))
        total = scale[0] * seg + self.lambda_topo * scale[1] * topology + self.lambda_picr * scale[2] * picr
        return total, {'loss': float(total.detach()), 'seg': float(seg.detach()), 'topo': float(topology.detach()), 'picr': float(picr.detach()), 'unlabeled_instances': int(unlabeled.sum())}
