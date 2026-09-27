"""KGVLMap spatial encoders, MGPE/PSA/PICR, instance SDMR and dense decoder."""
import hashlib
import json
import math
import types
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
import timm
from timm.models.swin_transformer import SwinTransformerBlock, window_partition, window_reverse


def padding_masked_attention(self, x):
    """Retain timm's pretrained attention, masking both shifted regions and padded keys."""
    b, h, w, c = x.shape
    shift = any(self.shift_size)
    y = torch.roll(x, (-self.shift_size[0], -self.shift_size[1]), (1, 2)) if shift else x
    ph, pw = (-h) % self.window_size[0], (-w) % self.window_size[1]
    y = F.pad(y, (0, 0, 0, pw, 0, ph))
    hp, wp = y.shape[1:3]
    win = window_partition(y, self.window_size).reshape(-1, self.window_area, c)
    # Cache masks by input resolution/device/dtype, independent of batch or data.
    key = (h, w, x.device, x.dtype)
    cached = getattr(self, '_kg_mask_cache', None)
    if cached is None or cached[0] != key:
        region_mask = self.get_attn_mask(y)
        valid = torch.ones((1, h, w, 1), device=x.device, dtype=x.dtype)
        valid = F.pad(valid, (0, 0, 0, pw, 0, ph))
        valid = window_partition(valid, self.window_size).reshape(-1, self.window_area)
        pad_mask = (1 - valid[:, None, :]).expand(-1, self.window_area, -1) * -10000.0
        mask = pad_mask if region_mask is None else pad_mask + region_mask.to(x.dtype)
        self._kg_mask_cache = (key, mask)
    mask = self._kg_mask_cache[1]
    y = self.attn(win, mask=mask)
    y = window_reverse(y.reshape(-1, *self.window_size, c), self.window_size, hp, wp)
    y = y[:, :h, :w].contiguous()
    return torch.roll(y, self.shift_size, (1, 2)) if shift else y


def install_padding_masks(model):
    for module in model.modules():
        if isinstance(module, SwinTransformerBlock):
            module._attn = types.MethodType(padding_masked_attention, module)


def load_text_embeddings(cfg):
    """Cache frozen RemoteCLIP text outputs, without retaining an unused visual encoder."""
    signature = hashlib.sha256(json.dumps({'prompts': cfg['prompts'], 'weights': str(Path(cfg['text_weights']).resolve()),
                                         'weight_bytes': Path(cfg['text_weights']).stat().st_size}, sort_keys=True).encode()).hexdigest()
    path = Path(cfg['text_cache'])
    if path.exists():
        saved = torch.load(path, map_location='cpu', weights_only=True)
        if saved['signature'] == signature:
            return saved['embeddings']
    import open_clip
    model = open_clip.create_model('ViT-B-32', pretrained=None, force_quick_gelu=True)
    weights = torch.load(cfg['text_weights'], map_location='cpu', weights_only=True)
    model.load_state_dict(weights, strict=True)
    model.requires_grad_(False).eval()
    tokenizer = open_clip.get_tokenizer('ViT-B-32')
    with torch.no_grad():
        # CLIP's normalized projected sentence representation supplies shared queries.
        embeddings = model.encode_text(tokenizer(cfg['prompts']), normalize=True).float().cpu()
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({'signature': signature, 'embeddings': embeddings}, path)
    return embeddings


class ConvBlock(nn.Sequential):
    def __init__(self, cin, cout, stride=1):
        super().__init__(nn.Conv2d(cin, cout, 3, stride, 1, bias=False), nn.GroupNorm(min(8, cout), cout), nn.GELU())


class PhysicalEncoder(nn.Module):
    """Four-level lightweight CNN. GroupNorm permits micro-batch size one."""
    def __init__(self, channels):
        super().__init__()
        self.stem = nn.Sequential(ConvBlock(channels, 32, 2), ConvBlock(32, 64, 2))
        self.stages = nn.ModuleList([nn.Sequential(ConvBlock(a, b, 2), ConvBlock(b, b)) for a, b in [(64, 128), (128, 256), (256, 512)]])

    def forward(self, x):
        x = self.stem(x)
        out = [x]
        for layer in self.stages:
            x = layer(x)
            out.append(x)
        return out


class PyramidNeck(nn.Module):
    def __init__(self, in_channels, width=256):
        super().__init__()
        self.lateral = nn.ModuleList([nn.Conv2d(c, width, 1) for c in in_channels])
        self.refine = ConvBlock(width, width)

    def forward(self, maps):
        z = self.lateral[-1](maps[-1])
        for i in range(len(maps) - 2, -1, -1):
            z = self.lateral[i](maps[i]) + F.interpolate(z, size=maps[i].shape[-2:], mode='bilinear', align_corners=False)
        return self.refine(z)


class MGPE(nn.Module):
    def __init__(self, width=512, bases=16):
        super().__init__()
        self.centers = nn.Parameter(torch.linspace(0, 1, bases).repeat(3, 1))
        self.log_width = nn.Parameter(torch.full((3, bases), math.log(math.expm1(1.5 / bases))))
        self.projections = nn.ModuleList([nn.Linear(bases, width, bias=False) for _ in range(3)])
        self.norms = nn.ModuleList([nn.LayerNorm(width) for _ in range(3)])

    def forward(self, values):
        widths = F.softplus(self.log_width) + 1e-4
        responses = torch.exp(-0.5 * ((values[:, :, None] - self.centers) / widths).square())
        return torch.stack([self.norms[i](self.projections[i](responses[:, i])) for i in range(3)], dim=1)


class PhysicalSemanticAlignment(nn.Module):
    def __init__(self, text_embeddings, width=512, bases=16, attention_dim=512):
        super().__init__()
        self.register_buffer('text_embeddings', text_embeddings.float())
        self.mgpe = MGPE(width, bases)
        self.q = nn.Linear(width, attention_dim, bias=False)
        self.k = nn.Linear(width, attention_dim, bias=False)
        self.v = nn.Linear(width, width, bias=False)
        self.residual_norm = nn.LayerNorm(width)
        self.condition_norm = nn.LayerNorm(width)
        self.geometry_projection = nn.Sequential(nn.Linear(2 * width, width), nn.GELU(), nn.LayerNorm(width))
        self.regressor = nn.Sequential(nn.Linear(width, 128), nn.GELU(), nn.Linear(128, 1), nn.Sigmoid())

    def forward(self, attributes, valid):
        tokens = self.mgpe(attributes)
        q, k, v = self.q(self.text_embeddings), self.k(tokens), self.v(tokens)
        scores = (q[None] @ k.transpose(-1, -2)) / math.sqrt(q.shape[-1])
        scores = scores.masked_fill(~valid[:, None, :], -10000)
        aligned = scores.softmax(-1) @ v
        condition = self.condition_norm(self.residual_norm(self.text_embeddings[None] + aligned).mean(1))
        ntl = self.regressor(self.geometry_projection(tokens[:, :2].flatten(1)))[:, 0]
        return condition, ntl


def coverage_matrix(instances, num_instances, factor=4):
    """Exact fractions from the full-resolution ownership raster, including background."""
    b, h, w = instances.shape
    if h % factor or w % factor:
        raise ValueError('Ownership raster must be divisible by the feature stride.')
    hh, ww = h // factor, w // factor
    cells = torch.arange(b * hh * ww, device=instances.device).reshape(b, hh, ww)
    cells = cells.repeat_interleave(factor, 1).repeat_interleave(factor, 2).flatten()
    ids = instances.flatten().long()
    encoded, counts = torch.unique(cells * (num_instances + 1) + ids, return_counts=True)
    index = torch.stack([encoded // (num_instances + 1), encoded % (num_instances + 1)])
    return torch.sparse_coo_tensor(index, counts.float() / (factor * factor), (b * hh * ww, num_instances + 1)).coalesce()


class SemanticModalityRouter(nn.Module):
    def __init__(self, token_width=512, spatial_width=256):
        super().__init__()
        self.width = spatial_width
        self.gates = nn.Sequential(nn.Linear(token_width, token_width), nn.GELU(), nn.Linear(token_width, 4 * spatial_width))
        self.beta = nn.Linear(token_width, 1)

    def forward(self, maps, condition, instances):
        n, c = condition.shape[0], self.width
        gates = self.gates(condition).reshape(n, 4, c).sigmoid()
        beta = self.beta(condition).sigmoid()
        effective = gates * torch.cat([beta, (1 - beta).expand(-1, 3)], 1)[:, :, None]
        background = effective.new_zeros((1, 4, c))
        background[:, 0] = 1  # equation (10): ungated RGB outside supplied footprints
        weights = torch.cat([background, effective], 0)
        coverage = coverage_matrix(instances, n)
        b, _, h, w = maps[0].shape
        fused = 0
        # Sparse coverage aggregation is algebraically identical to per-instance fusion then scatter.
        with torch.autocast(device_type=condition.device.type, enabled=False):
            dense_weights = [torch.sparse.mm(coverage.float(), weights[:, m].float()).reshape(b, h, w, c).permute(0, 3, 1, 2) for m in range(4)]
        for feature, weight in zip(maps, dense_weights):
            fused = fused + feature * weight.to(feature.dtype)
        return fused, gates, beta


class DenseDecoder(nn.Module):
    def __init__(self, classes, width=256, heads=8, window=7, checkpoint_blocks=True):
        super().__init__()
        self.checkpoint_blocks = checkpoint_blocks
        self.blocks = nn.ModuleList([SwinTransformerBlock(dim=width, input_resolution=(256, 256), num_heads=heads,
            window_size=window, shift_size=(window // 2 if i % 2 else 0), always_partition=True, dynamic_mask=True) for i in range(4)])
        self.up1, self.up2 = ConvBlock(width, 128), ConvBlock(128, 64)
        self.head = nn.Conv2d(64, classes, 1)
        install_padding_masks(self)

    def forward(self, x, size):
        x = x.permute(0, 2, 3, 1)
        for layer in self.blocks:
            x = checkpoint(layer, x, use_reentrant=False) if self.training and self.checkpoint_blocks else layer(x)
        x = x.permute(0, 3, 1, 2)
        x = self.up1(F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False))
        x = self.up2(F.interpolate(x, size=size, mode='bilinear', align_corners=False))
        return self.head(x)


class KGVLMap(nn.Module):
    def __init__(self, cfg, text_embeddings=None, pretrained=True):
        super().__init__()
        text_embeddings = load_text_embeddings(cfg) if text_embeddings is None else text_embeddings
        if pretrained and not Path(cfg['swin_weights']).exists():
            raise FileNotFoundError('The required ImageNet-22K Swin-B checkpoint is missing.')
        self.rgb_encoder = timm.create_model('swin_base_patch4_window7_224.ms_in22k', pretrained=pretrained,
            pretrained_cfg_overlay={'file': cfg['swin_weights']} if pretrained else None,
            features_only=True, out_indices=(0, 1, 2, 3), img_size=cfg['crop_size'], strict_img_size=False)
        self.rgb_encoder.requires_grad_(False).eval()
        install_padding_masks(self.rgb_encoder)
        self.physical_encoders = nn.ModuleList([PhysicalEncoder(1), PhysicalEncoder(cfg['ntl_channels']), PhysicalEncoder(2)])
        c = cfg['spatial_width']
        self.necks = nn.ModuleList([PyramidNeck([128, 256, 512, 1024], c)] + [PyramidNeck([64, 128, 256, 512], c) for _ in range(3)])
        self.psa = PhysicalSemanticAlignment(text_embeddings, cfg['token_width'], cfg['gaussian_bases'], cfg['attention_dim'])
        self.router = SemanticModalityRouter(cfg['token_width'], c)
        self.decoder = DenseDecoder(len(cfg['classes']) + 1, c, cfg['decoder_heads'], cfg['decoder_window'], cfg['activation_checkpointing'])

    def train(self, mode=True):
        super().train(mode)
        self.rgb_encoder.eval()
        return self

    def forward(self, batch):
        with torch.no_grad():
            rgb_pyramid = self.rgb_encoder(batch['rgb'])
            rgb_pyramid = [v.permute(0, 3, 1, 2).contiguous() for v in rgb_pyramid]
        maps = [self.necks[0](rgb_pyramid)]
        for encoder, neck, key in zip(self.physical_encoders, self.necks[1:], ['height', 'ntl', 'morphology']):
            maps.append(neck(encoder(batch[key])))
        cond, ntl_pred = self.psa(batch['attrs'], batch['valid'])
        fused, gates, beta = self.router(maps, cond, batch['instances'])
        logits = self.decoder(fused, batch['rgb'].shape[-2:])
        return {'logits': logits, 'ntl_pred': ntl_pred, 'gates': gates, 'beta': beta}
