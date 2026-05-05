import torch
import torch.nn as nn
import torch.nn.functional as F


class HybridDetectionHead(nn.Module):
    """Hybrid detection head with separate classification and regression branches.

    Args:
        num_classes (int): Number of foreground object categories.
        in_channels (int): Input feature channel size from neck.
        feat_channels (int): Hidden feature channel size in head convolutions.
        num_levels (int): Number of FPN feature pyramid levels.
        num_stacked_convs (int): Number of stacked 3x3 convs in each branch.
        strides (list[int]): Strides of each FPN level. Default: [8, 16, 32, 64, 128].
    """

    def __init__(
        self,
        num_classes,
        in_channels=256,
        feat_channels=256,
        num_levels=5,
        num_stacked_convs=4,
        strides=None,
    ):
        super(HybridDetectionHead, self).__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.feat_channels = feat_channels
        self.num_levels = num_levels
        self.num_stacked_convs = num_stacked_convs
        self.strides = strides or [8, 16, 32, 64, 128]

        self._build_layers()

    def _build_layers(self):
        cls_convs = []
        reg_convs = []
        for i in range(self.num_stacked_convs):
            in_ch = self.in_channels if i == 0 else self.feat_channels
            cls_convs.append(
                nn.Sequential(
                    nn.Conv2d(in_ch, self.feat_channels, 3, padding=1),
                    nn.GroupNorm(32, self.feat_channels),
                    nn.ReLU(inplace=True),
                )
            )
            reg_convs.append(
                nn.Sequential(
                    nn.Conv2d(in_ch, self.feat_channels, 3, padding=1),
                    nn.GroupNorm(32, self.feat_channels),
                    nn.ReLU(inplace=True),
                )
            )

        self.cls_convs = nn.Sequential(*cls_convs)
        self.reg_convs = nn.Sequential(*reg_convs)

        # Final prediction layers
        self.cls_pred = nn.Conv2d(self.feat_channels, self.num_classes, 3, padding=1)
        self.reg_pred = nn.Conv2d(self.feat_channels, 4, 3, padding=1)
        self.centerness_pred = nn.Conv2d(self.feat_channels, 1, 3, padding=1)

        # Per-level learnable scale for regression
        self.scales = nn.ParameterList(
            [nn.Parameter(torch.ones(1)) for _ in range(self.num_levels)]
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        # Initialize classification bias with prior probability
        import math
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        nn.init.constant_(self.cls_pred.bias, bias_value)

    def forward_single(self, feat, scale):
        """Forward pass for a single FPN level feature map."""
        cls_feat = self.cls_convs(feat)
        reg_feat = self.reg_convs(feat)

        cls_score = self.cls_pred(cls_feat)
        reg_pred = F.relu(scale * self.reg_pred(reg_feat))
        centerness = self.centerness_pred(reg_feat)

        return cls_score, reg_pred, centerness

    def forward_train(self, feats, img_metas, gt_bboxes, gt_labels):
        """Compute losses for a batch during training.

        Args:
            feats (tuple[Tensor]): Multi-level feature maps from the neck.
            img_metas (list[dict]): Image metadata for each image in the batch.
            gt_bboxes (list[Tensor]): Ground-truth bounding boxes per image,
                each of shape (num_gt, 4) in (x1, y1, x2, y2) format.
            gt_labels (list[Tensor]): Ground-truth class labels per image,
                each of shape (num_gt,).

        Returns:
            dict[str, Tensor]: Loss dict with keys 'loss_cls', 'loss_bbox',
                and 'loss_centerness'.
        """
        all_cls, all_reg, all_ctr = zip(
            *[self.forward_single(feat, self.scales[i]) for i, feat in enumerate(feats)]
        )

        # Build FCOS-style targets and compute losses
        losses = self._compute_losses(
            all_cls, all_reg, all_ctr, img_metas, gt_bboxes, gt_labels
        )
        return losses

    def forward_test(self, feats, img_metas):
        """Generate detections during inference.

        Args:
            feats (tuple[Tensor]): Multi-level feature maps from the neck.
            img_metas (list[dict]): Image metadata.

        Returns:
            list[list[ndarray]]: Detection results per image per class,
                each ndarray of shape (num_det, 5) with columns
                [x1, y1, x2, y2, score].
        """
        all_cls, all_reg, all_ctr = zip(
            *[self.forward_single(feat, self.scales[i]) for i, feat in enumerate(feats)]
        )

        results = self._decode_predictions(all_cls, all_reg, all_ctr, img_metas)
        return results

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_losses(self, all_cls, all_reg, all_ctr, img_metas,
                        gt_bboxes, gt_labels):
        """Compute classification, bbox regression, and centerness losses."""
        loss_cls_list = []
        loss_bbox_list = []
        loss_ctr_list = []

        for level_idx, (cls_scores, reg_preds, ctr_preds) in enumerate(
            zip(all_cls, all_reg, all_ctr)
        ):
            stride = self.strides[level_idx]
            h, w = cls_scores.shape[-2:]
            device = cls_scores.device

            # Build pixel-wise targets for the current level
            cls_targets, reg_targets, ctr_targets, pos_mask = self._build_targets(
                h, w, stride, gt_bboxes, gt_labels, device
            )

            # Flatten spatial dims: (N, C, H, W) -> (N*H*W, C)
            cls_flat = cls_scores.permute(0, 2, 3, 1).reshape(-1, self.num_classes)
            reg_flat = reg_preds.permute(0, 2, 3, 1).reshape(-1, 4)
            ctr_flat = ctr_preds.permute(0, 2, 3, 1).reshape(-1)

            num_pos = pos_mask.sum().clamp(min=1)

            # Classification loss (sigmoid focal)
            loss_cls = self._sigmoid_focal_loss(
                cls_flat, cls_targets, pos_mask
            ) / num_pos
            loss_cls_list.append(loss_cls)

            # Bounding-box regression loss (IoU loss on positive locations)
            if pos_mask.any():
                loss_bbox = self._iou_loss(
                    reg_flat[pos_mask], reg_targets[pos_mask]
                )
                loss_ctr = F.binary_cross_entropy_with_logits(
                    ctr_flat[pos_mask], ctr_targets[pos_mask], reduction='mean'
                )
            else:
                loss_bbox = reg_flat.sum() * 0
                loss_ctr = ctr_flat.sum() * 0

            loss_bbox_list.append(loss_bbox)
            loss_ctr_list.append(loss_ctr)

        return dict(
            loss_cls=sum(loss_cls_list),
            loss_bbox=sum(loss_bbox_list),
            loss_centerness=sum(loss_ctr_list),
        )

    def _build_targets(self, h, w, stride, gt_bboxes_list, gt_labels_list, device):
        """Build FCOS-style per-pixel targets for one FPN level.

        Returns:
            cls_targets  (N*H*W, num_classes) float32
            reg_targets  (N*H*W, 4) float32 – (l, t, r, b) distances
            ctr_targets  (N*H*W,) float32
            pos_mask     (N*H*W,) bool
        """
        batch = len(gt_bboxes_list)

        # Center coordinates of each feature-map cell in image space
        xs = (torch.arange(w, device=device).float() + 0.5) * stride
        ys = (torch.arange(h, device=device).float() + 0.5) * stride
        ys, xs = torch.meshgrid(ys, xs, indexing='ij')
        xs = xs.reshape(-1)
        ys = ys.reshape(-1)

        all_cls_tgt, all_reg_tgt, all_ctr_tgt, all_pos = [], [], [], []

        for img_idx in range(batch):
            gt_boxes = gt_bboxes_list[img_idx].to(device)   # (G, 4) x1y1x2y2
            gt_cls = gt_labels_list[img_idx].to(device)     # (G,)

            num_pts = h * w
            cls_tgt = torch.zeros(num_pts, self.num_classes, device=device)
            reg_tgt = torch.zeros(num_pts, 4, device=device)
            ctr_tgt = torch.zeros(num_pts, device=device)
            pos_flag = torch.zeros(num_pts, dtype=torch.bool, device=device)

            if gt_boxes.numel() > 0:
                # l, t, r, b distances for all (point, gt) pairs
                l = xs[:, None] - gt_boxes[None, :, 0]   # (P, G)
                t = ys[:, None] - gt_boxes[None, :, 1]
                r = gt_boxes[None, :, 2] - xs[:, None]
                b = gt_boxes[None, :, 3] - ys[:, None]
                ltrb = torch.stack([l, t, r, b], dim=-1)  # (P, G, 4)

                inside = ltrb.min(dim=-1).values > 0      # (P, G)

                # Assign each positive point to the smallest GT it falls in
                areas = (gt_boxes[:, 2] - gt_boxes[:, 0]) * \
                        (gt_boxes[:, 3] - gt_boxes[:, 1])  # (G,)
                # Penalise points outside any GT
                filled = inside.float() * areas[None, :]
                filled[~inside] = float('inf')
                min_area, min_idx = filled.min(dim=1)     # (P,)

                pos = min_area < float('inf')
                pos_flag[pos] = True

                assigned_ltrb = ltrb[pos, min_idx[pos]]   # (pos_pts, 4)
                reg_tgt[pos] = assigned_ltrb

                # Centerness target
                min_lr = torch.min(assigned_ltrb[:, 0], assigned_ltrb[:, 2])
                max_lr = torch.max(assigned_ltrb[:, 0], assigned_ltrb[:, 2])
                min_tb = torch.min(assigned_ltrb[:, 1], assigned_ltrb[:, 3])
                max_tb = torch.max(assigned_ltrb[:, 1], assigned_ltrb[:, 3])
                ctr_tgt[pos] = torch.sqrt(
                    (min_lr / max_lr.clamp(min=1e-6)) *
                    (min_tb / max_tb.clamp(min=1e-6))
                )

                # One-hot class target
                assigned_cls = gt_cls[min_idx[pos]]
                cls_tgt[pos, assigned_cls] = 1.0

            all_cls_tgt.append(cls_tgt)
            all_reg_tgt.append(reg_tgt)
            all_ctr_tgt.append(ctr_tgt)
            all_pos.append(pos_flag)

        return (
            torch.cat(all_cls_tgt, dim=0),
            torch.cat(all_reg_tgt, dim=0),
            torch.cat(all_ctr_tgt, dim=0),
            torch.cat(all_pos, dim=0),
        )

    @staticmethod
    def _sigmoid_focal_loss(pred, target, pos_mask, alpha=0.25, gamma=2.0):
        """Sigmoid focal loss for dense classification."""
        pred_sigmoid = pred.sigmoid()
        p_t = pred_sigmoid * target + (1 - pred_sigmoid) * (1 - target)
        alpha_t = alpha * target + (1 - alpha) * (1 - target)
        loss = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        focal_weight = alpha_t * (1 - p_t).pow(gamma)
        return (focal_weight * loss).sum()

    @staticmethod
    def _iou_loss(pred_ltrb, target_ltrb, eps=1e-6):
        """IoU loss for bounding-box regression (ltrb format)."""
        pred_area = (pred_ltrb[:, 0] + pred_ltrb[:, 2]) * \
                    (pred_ltrb[:, 1] + pred_ltrb[:, 3])
        gt_area = (target_ltrb[:, 0] + target_ltrb[:, 2]) * \
                  (target_ltrb[:, 1] + target_ltrb[:, 3])
        inter_w = torch.min(pred_ltrb[:, 0], target_ltrb[:, 0]) + \
                  torch.min(pred_ltrb[:, 2], target_ltrb[:, 2])
        inter_h = torch.min(pred_ltrb[:, 1], target_ltrb[:, 1]) + \
                  torch.min(pred_ltrb[:, 3], target_ltrb[:, 3])
        inter_area = inter_w.clamp(min=0) * inter_h.clamp(min=0)
        union = pred_area + gt_area - inter_area + eps
        iou = inter_area / union
        return (1 - iou).mean()

    def _decode_predictions(self, all_cls, all_reg, all_ctr, img_metas):
        """Decode raw predictions to bounding boxes (for inference)."""
        import numpy as np

        batch_size = all_cls[0].shape[0]
        results = [[] for _ in range(batch_size)]

        for level_idx, (cls_scores, reg_preds, ctr_preds) in enumerate(
            zip(all_cls, all_reg, all_ctr)
        ):
            stride = self.strides[level_idx]
            h, w = cls_scores.shape[-2:]
            device = cls_scores.device

            xs = (torch.arange(w, device=device).float() + 0.5) * stride
            ys = (torch.arange(h, device=device).float() + 0.5) * stride
            ys, xs = torch.meshgrid(ys, xs, indexing='ij')
            xs = xs.reshape(-1)
            ys = ys.reshape(-1)

            for img_idx in range(batch_size):
                cls_s = cls_scores[img_idx].sigmoid()         # (C, H, W)
                ctr_s = ctr_preds[img_idx].sigmoid().reshape(-1)  # (H*W,)
                reg_s = reg_preds[img_idx].permute(1, 2, 0).reshape(-1, 4)

                scores = (cls_s.permute(1, 2, 0).reshape(-1, self.num_classes) *
                          ctr_s[:, None])                      # (H*W, C)

                x1 = xs - reg_s[:, 0]
                y1 = ys - reg_s[:, 1]
                x2 = xs + reg_s[:, 2]
                y2 = ys + reg_s[:, 3]
                bboxes = torch.stack([x1, y1, x2, y2], dim=1)  # (H*W, 4)

                results[img_idx].append((
                    bboxes.cpu().detach().numpy(),
                    scores.cpu().detach().numpy(),
                ))

        return results
