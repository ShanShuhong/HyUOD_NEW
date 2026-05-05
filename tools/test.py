"""Evaluate a trained HyUOD model on a test dataset."""

import argparse
import importlib.util
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from hyuod import HyUOD, DUODataset
from hyuod.utils import Compose, Resize, Normalize, ToTensor


def parse_args():
    parser = argparse.ArgumentParser(description='Test / evaluate HyUOD')
    parser.add_argument('config', help='Path to config file')
    parser.add_argument('checkpoint', help='Path to model checkpoint')
    parser.add_argument('--eval', nargs='+', default=['bbox'],
                        help='Evaluation metrics (bbox)')
    parser.add_argument('--score-thr', type=float, default=0.05,
                        help='Score threshold for NMS')
    parser.add_argument('--nms-iou-thr', type=float, default=0.5,
                        help='IoU threshold for NMS')
    parser.add_argument('--out', default=None,
                        help='Path to save detection results JSON')
    return parser.parse_args()


def load_config(config_path):
    spec = importlib.util.spec_from_file_location('cfg', config_path)
    cfg = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cfg)
    return cfg


def nms(boxes, scores, iou_threshold):
    """Non-maximum suppression.

    Args:
        boxes (np.ndarray): Shape (N, 4) in (x1, y1, x2, y2) format.
        scores (np.ndarray): Shape (N,).
        iou_threshold (float): IoU threshold.

    Returns:
        np.ndarray: Indices of kept detections.
    """
    if len(boxes) == 0:
        return np.array([], dtype=np.int64)

    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
        order = order[np.where(iou <= iou_threshold)[0] + 1]

    return np.array(keep, dtype=np.int64)


def main():
    args = parse_args()
    cfg = load_config(args.config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Build test dataset
    test_transforms = Compose([
        Resize(cfg.dataset['img_size'], keep_ratio=True),
        Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
        ToTensor(),
    ])
    test_dataset = DUODataset(
        img_dir=cfg.dataset['test']['img_dir'],
        ann_file=cfg.dataset['test']['ann_file'],
        transforms=test_transforms,
        img_size=cfg.dataset['img_size'],
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=cfg.dataset['num_workers'],
        collate_fn=DUODataset.collate_fn,
    )

    # Load model
    model = HyUOD(**cfg.model).to(device)
    ckpt = torch.load(args.checkpoint, map_location='cpu')
    state = ckpt.get('model', ckpt)
    model.load_state_dict(state)
    model.eval()

    # Run inference
    coco_results = []
    with torch.no_grad():
        for batch in test_loader:
            imgs = batch['img'].to(device)
            img_metas = batch['img_metas']

            raw_results = model(imgs, img_metas)

            for img_idx, img_meta in enumerate(img_metas):
                img_id = img_meta['img_id']
                all_boxes = []
                all_scores = []
                all_cls_ids = []

                for level_result in raw_results[img_idx]:
                    bboxes, scores = level_result    # scores: (N, num_classes)
                    for cls_idx in range(scores.shape[1]):
                        cls_scores = scores[:, cls_idx]
                        mask = cls_scores > args.score_thr
                        if not mask.any():
                            continue
                        all_boxes.append(bboxes[mask])
                        all_scores.append(cls_scores[mask])
                        all_cls_ids.append(
                            np.full(mask.sum(), cls_idx, dtype=np.int32)
                        )

                if not all_boxes:
                    continue

                boxes_cat = np.concatenate(all_boxes, axis=0)
                scores_cat = np.concatenate(all_scores, axis=0)
                cls_ids_cat = np.concatenate(all_cls_ids, axis=0)

                for cls_idx in range(cfg.model['num_classes']):
                    mask = cls_ids_cat == cls_idx
                    if not mask.any():
                        continue
                    cls_boxes = boxes_cat[mask]
                    cls_scores = scores_cat[mask]
                    keep = nms(cls_boxes, cls_scores, args.nms_iou_thr)
                    for k in keep:
                        x1, y1, x2, y2 = cls_boxes[k].tolist()
                        coco_results.append({
                            'image_id': img_id,
                            'category_id': cls_idx + 1,
                            'bbox': [x1, y1, x2 - x1, y2 - y1],
                            'score': float(cls_scores[k]),
                        })

    # COCO evaluation
    if 'bbox' in args.eval:
        coco_gt = COCO(cfg.dataset['test']['ann_file'])
        coco_dt = coco_gt.loadRes(coco_results) if coco_results else COCO()
        coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

    if args.out:
        import json
        out_dir = os.path.dirname(os.path.abspath(args.out))
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(args.out, 'w') as f:
            json.dump(coco_results, f)
        print(f'Detection results saved to {args.out}')


if __name__ == '__main__':
    main()
