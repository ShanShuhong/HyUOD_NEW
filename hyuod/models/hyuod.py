import torch
import torch.nn as nn
from .backbone import ResNet
from .neck import FPN
from .head import HybridDetectionHead
from ..utils import UnderwaterImageEnhancer


class HyUOD(nn.Module):
    """Hybrid Underwater Object Detection (HyUOD) model.

    This model uses a hybrid approach that combines raw and enhanced underwater
    images to exploit complementary information for improved detection.

    Args:
        num_classes (int): Number of object categories (excluding background).
        backbone_depth (int): ResNet backbone depth (50 or 101). Default: 50.
        use_enhancement (bool): Whether to use the underwater image enhancement
            branch. Default: True.
        pretrained (str, optional): Path to pretrained backbone weights.
    """

    def __init__(
        self,
        num_classes,
        backbone_depth=50,
        use_enhancement=True,
        pretrained=None,
    ):
        super(HyUOD, self).__init__()
        self.num_classes = num_classes
        self.use_enhancement = use_enhancement

        # Backbone for raw image features
        self.backbone = ResNet(depth=backbone_depth, pretrained=pretrained)

        # Optional underwater image enhancer and its backbone
        if use_enhancement:
            self.enhancer = UnderwaterImageEnhancer()
            self.enh_backbone = ResNet(depth=backbone_depth, pretrained=pretrained)
            # 1x1 conv to fuse raw and enhanced features at each FPN level
            self.fusion_convs = nn.ModuleList([
                nn.Conv2d(512, 256, kernel_size=1) if i == 0 else
                nn.Conv2d(1024, 256, kernel_size=1) if i == 1 else
                nn.Conv2d(2048, 256, kernel_size=1) if i == 2 else
                nn.Conv2d(4096, 256, kernel_size=1)
                for i in range(4)
            ])

        # Neck: Feature Pyramid Network
        self.neck = FPN(
            in_channels=[256, 512, 1024, 2048],
            out_channels=256,
            num_outs=5,
        )

        # Detection head
        self.head = HybridDetectionHead(
            num_classes=num_classes,
            in_channels=256,
            feat_channels=256,
            num_levels=5,
        )

    def extract_features(self, img):
        """Extract multi-scale features from the input image."""
        feats = self.backbone(img)
        return feats

    def forward(self, img, img_metas=None, gt_bboxes=None, gt_labels=None):
        """Forward pass.

        Args:
            img (Tensor): Input images of shape (N, C, H, W).
            img_metas (list[dict]): List of image metadata.
            gt_bboxes (list[Tensor]): Ground-truth bounding boxes (training only).
            gt_labels (list[Tensor]): Ground-truth class labels (training only).

        Returns:
            dict or list: During training, returns a dict of losses.
                During inference, returns a list of detection results.
        """
        raw_feats = self.extract_features(img)

        if self.use_enhancement:
            enhanced = self.enhancer(img)
            enh_feats = self.enh_backbone(enhanced)
            # Fuse raw and enhanced features by concatenation + 1x1 conv
            fused_feats = []
            for i, (r, e) in enumerate(zip(raw_feats, enh_feats)):
                cat = torch.cat([r, e], dim=1)
                fused_feats.append(self.fusion_convs[i](cat))
            backbone_feats = tuple(fused_feats)
        else:
            backbone_feats = raw_feats

        neck_feats = self.neck(backbone_feats)

        if self.training:
            losses = self.head.forward_train(
                neck_feats, img_metas, gt_bboxes, gt_labels
            )
            return losses
        else:
            results = self.head.forward_test(neck_feats, img_metas)
            return results
