"""MVDet-style learned multi-view BEV occupancy detector (H2, fugu design Cycle 16).

Breaks the precision-recall TRADE of late fusion (greedy/NMS/bev-vote) by aggregating
per-view CNN FEATURES on the shared BEV ground plane and detecting THERE — so a person is
recovered from accumulated multi-view evidence (recall) while the learned head suppresses
spurious activations (precision), instead of forcing a hard rule on independent per-view
detections.

Pipeline (fugu):
    imgs [B,V,3,H,W]
      -> shared ResNet-18 backbone (stride 8) -> feat [B*V,512,H/8,W/8]
      -> 1x1 conv compress -> [B*V,64,h,w]
      -> grid_sample with precomputed BEV->view grids [V,Hg,Wg,2] -> proj [B,V,64,Hg,Wg]
      -> concat over views (+2 normalised BEV-coord channels) -> [B, V*64+2, Hg,Wg]
      -> BEV head (3x conv3x3 BN-ReLU + 1x1) -> occupancy logit [B,1,Hg,Wg]

WILDTRACK BEV grid: 12x36 m, 10 cm -> Hg,Wg = 120,360. Train target = bev_gt_heatmap.
Requires torch + torchvision (Colab GPU); this module is import-guarded so the rest of the
repo (numpy/cv2 sandbox) is unaffected.
"""
from __future__ import annotations


def _torch():
    import torch  # noqa: F401
    return True


def build_mvdet(n_views: int = 7, feat_ch: int = 64, head_ch: int = 128, pretrained: bool = True):
    """Construct the MVDet module. Imports torch lazily so importing this file is cheap."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torchvision

    class MVDet(nn.Module):
        def __init__(self):
            super().__init__()
            bb = torchvision.models.resnet18(weights="DEFAULT" if pretrained else None)
            # keep up to layer2 -> stride 8, 128 channels
            self.backbone = nn.Sequential(bb.conv1, bb.bn1, bb.relu, bb.maxpool, bb.layer1, bb.layer2)
            self.compress = nn.Conv2d(128, feat_ch, 1)
            self.head = nn.Sequential(
                nn.Conv2d(n_views * feat_ch + 2, head_ch, 3, padding=2, dilation=2), nn.BatchNorm2d(head_ch), nn.ReLU(True),
                nn.Conv2d(head_ch, head_ch, 3, padding=2, dilation=2), nn.BatchNorm2d(head_ch), nn.ReLU(True),
                nn.Conv2d(head_ch, head_ch, 3, padding=1), nn.BatchNorm2d(head_ch), nn.ReLU(True),
                nn.Conv2d(head_ch, 1, 1),
            )
            self.n_views = n_views
            self.feat_ch = feat_ch

        def forward(self, imgs, grids, coord):
            # imgs [B,V,3,H,W]; grids [V,Hg,Wg,2] (normalised BEV->view); coord [2,Hg,Wg]
            B, V, C, H, W = imgs.shape
            f = self.backbone(imgs.view(B * V, C, H, W))
            f = self.compress(f)                                  # [B*V,feat,h,w]
            _, fc, h, w = f.shape
            f = f.view(B, V, fc, h, w)
            Hg, Wg = grids.shape[1], grids.shape[2]
            proj = []
            for v in range(V):
                g = grids[v].unsqueeze(0).expand(B, Hg, Wg, 2)    # [B,Hg,Wg,2]
                proj.append(F.grid_sample(f[:, v], g, align_corners=False, padding_mode="zeros"))
            bev = torch.cat(proj, dim=1)                          # [B,V*feat,Hg,Wg]
            cc = coord.unsqueeze(0).expand(B, 2, Hg, Wg)
            bev = torch.cat([bev, cc], dim=1)
            return self.head(bev)                                 # [B,1,Hg,Wg] logits

    return MVDet()


def focal_bev_loss(logit, target, alpha: float = 2.0, beta: float = 4.0, eps: float = 1e-6):
    """CenterNet-style penalty-reduced focal loss on a BEV Gaussian occupancy target in [0,1]."""
    import torch
    p = torch.sigmoid(logit).clamp(eps, 1 - eps)
    pos = (target >= 1.0 - 1e-4).float()
    pos_loss = -((1 - p) ** alpha) * torch.log(p) * pos
    neg_loss = -((1 - target) ** beta) * (p ** alpha) * torch.log(1 - p) * (1 - pos)
    n = pos.sum().clamp(min=1.0)
    return (pos_loss.sum() + neg_loss.sum()) / n
