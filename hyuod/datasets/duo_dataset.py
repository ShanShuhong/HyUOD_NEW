import os
import json
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset


class DUODataset(Dataset):
    """Dataset class for the Detecting Underwater Objects (DUO) benchmark.

    The dataset follows the COCO annotation format.

    Args:
        img_dir (str): Directory containing the images.
        ann_file (str): Path to the COCO-format JSON annotation file.
        transforms (callable, optional): Optional transform applied to each sample.
        img_size (tuple[int, int]): Target image size ``(height, width)``.
            Default: ``(800, 1333)``.

    References:
        Liu C. et al., "A Dataset and Benchmark of Underwater Object Detection for
        Robot Picking", arXiv 2021. https://github.com/chongweiliu/DUO
    """

    CLASSES = ('holothurian', 'echinus', 'scallop', 'starfish')

    def __init__(self, img_dir, ann_file, transforms=None, img_size=(800, 1333)):
        self.img_dir = img_dir
        self.ann_file = ann_file
        self.transforms = transforms
        self.img_size = img_size

        self._load_annotations()

    def _load_annotations(self):
        with open(self.ann_file, 'r') as f:
            coco_data = json.load(f)

        # Build id → filename mapping
        self.img_infos = {
            img['id']: img for img in coco_data['images']
        }
        self.img_ids = [img['id'] for img in coco_data['images']]

        # Build img_id → list of annotations mapping
        self.ann_dict = {img_id: [] for img_id in self.img_ids}
        for ann in coco_data.get('annotations', []):
            self.ann_dict[ann['image_id']].append(ann)

        # category_id → class index (0-based)
        self.cat_id_to_idx = {
            cat['id']: idx for idx, cat in enumerate(coco_data['categories'])
        }

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        img_info = self.img_infos[img_id]
        img_path = os.path.join(self.img_dir, img_info['file_name'])

        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f'Image not found: {img_path}')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Collect ground-truth boxes and labels
        gt_bboxes = []
        gt_labels = []
        for ann in self.ann_dict[img_id]:
            if ann.get('ignore', False):
                continue
            x, y, w, h = ann['bbox']
            gt_bboxes.append([x, y, x + w, y + h])
            gt_labels.append(self.cat_id_to_idx[ann['category_id']])

        gt_bboxes = np.array(gt_bboxes, dtype=np.float32).reshape(-1, 4)
        gt_labels = np.array(gt_labels, dtype=np.int64)

        sample = dict(
            img=img,
            gt_bboxes=gt_bboxes,
            gt_labels=gt_labels,
            img_meta=dict(
                img_id=img_id,
                ori_shape=img.shape[:2],
                filename=img_path,
            ),
        )

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample

    @staticmethod
    def collate_fn(batch):
        """Collate a list of samples into a mini-batch."""
        imgs = torch.stack([s['img'] for s in batch])
        gt_bboxes = [torch.as_tensor(s['gt_bboxes']) for s in batch]
        gt_labels = [torch.as_tensor(s['gt_labels']) for s in batch]
        img_metas = [s['img_meta'] for s in batch]
        return dict(
            img=imgs,
            gt_bboxes=gt_bboxes,
            gt_labels=gt_labels,
            img_metas=img_metas,
        )
