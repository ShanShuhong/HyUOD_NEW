import random
import numpy as np
import cv2
import torch


class Compose:
    """Compose multiple transforms together."""

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, sample):
        for t in self.transforms:
            sample = t(sample)
        return sample


class Resize:
    """Resize the image (and scale bboxes) to a target size.

    Args:
        size (tuple[int, int]): Target ``(height, width)``.
        keep_ratio (bool): If True, resize while keeping aspect ratio and
            pad the shorter side. Default: False.
    """

    def __init__(self, size, keep_ratio=False):
        self.size = size
        self.keep_ratio = keep_ratio

    def __call__(self, sample):
        img = sample['img']
        h, w = img.shape[:2]
        new_h, new_w = self.size

        if self.keep_ratio:
            scale = min(new_h / h, new_w / w)
            new_h = int(round(h * scale))
            new_w = int(round(w * scale))

        img = cv2.resize(img, (new_w, new_h))

        scale_x = new_w / w
        scale_y = new_h / h

        if sample['gt_bboxes'].size > 0:
            sample['gt_bboxes'][:, [0, 2]] *= scale_x
            sample['gt_bboxes'][:, [1, 3]] *= scale_y

        sample['img'] = img
        sample['img_meta']['img_shape'] = (new_h, new_w)
        sample['img_meta']['scale_factor'] = (scale_x, scale_y)
        return sample


class RandomHorizontalFlip:
    """Randomly flip the image horizontally with a given probability.

    Args:
        prob (float): Probability of flipping. Default: 0.5.
    """

    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, sample):
        if random.random() < self.prob:
            img = sample['img']
            w = img.shape[1]
            sample['img'] = np.fliplr(img).copy()

            if sample['gt_bboxes'].size > 0:
                bboxes = sample['gt_bboxes'].copy()
                bboxes[:, [0, 2]] = w - bboxes[:, [2, 0]]
                sample['gt_bboxes'] = bboxes

        return sample


class Normalize:
    """Normalize image pixel values with mean and standard deviation.

    Args:
        mean (list[float]): Per-channel mean values.
        std (list[float]): Per-channel standard deviation values.
        to_float (bool): Convert image to float32 and scale to [0, 1]
            before normalizing. Default: True.
    """

    def __init__(self, mean, std, to_float=True):
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
        self.to_float = to_float

    def __call__(self, sample):
        img = sample['img'].astype(np.float32)
        if self.to_float:
            img /= 255.0
        img = (img - self.mean) / self.std
        sample['img'] = img
        return sample


class ToTensor:
    """Convert numpy ndarray image and labels to PyTorch tensors."""

    def __call__(self, sample):
        img = sample['img']
        # HWC -> CHW
        if img.ndim == 3:
            img = np.transpose(img, (2, 0, 1))
        sample['img'] = torch.from_numpy(np.ascontiguousarray(img)).float()
        sample['gt_bboxes'] = torch.from_numpy(
            np.ascontiguousarray(sample['gt_bboxes'])
        ).float()
        sample['gt_labels'] = torch.from_numpy(
            np.ascontiguousarray(sample['gt_labels'])
        ).long()
        return sample
