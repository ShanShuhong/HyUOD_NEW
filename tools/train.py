"""Train HyUOD on an underwater object detection dataset."""

import argparse
import importlib.util
import os
import sys
import time

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from hyuod import HyUOD, DUODataset
from hyuod.utils import Compose, Resize, RandomHorizontalFlip, Normalize, ToTensor


def parse_args():
    parser = argparse.ArgumentParser(description='Train HyUOD')
    parser.add_argument('config', help='Path to config file')
    parser.add_argument(
        '--resume', default=None, help='Path to checkpoint to resume from'
    )
    parser.add_argument(
        '--launcher', default='none', choices=['none', 'pytorch'],
        help='Job launcher for distributed training'
    )
    parser.add_argument('--local_rank', type=int, default=0)
    return parser.parse_args()


def load_config(config_path):
    """Load a Python config file as a module."""
    spec = importlib.util.spec_from_file_location('cfg', config_path)
    cfg = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cfg)
    return cfg


def build_transforms(aug_cfg_list):
    transform_map = {
        'Resize': Resize,
        'RandomHorizontalFlip': RandomHorizontalFlip,
        'Normalize': Normalize,
        'ToTensor': ToTensor,
    }
    transforms = []
    for t_cfg in aug_cfg_list:
        t_type = t_cfg['type']
        kwargs = {k: v for k, v in t_cfg.items() if k != 'type'}
        transforms.append(transform_map[t_type](**kwargs))
    return Compose(transforms)


def main():
    args = parse_args()
    cfg = load_config(args.config)

    # Distributed training setup
    distributed = False
    if args.launcher == 'pytorch':
        torch.distributed.init_process_group(backend='nccl')
        local_rank = args.local_rank
        torch.cuda.set_device(local_rank)
        distributed = True

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Build dataset and dataloader
    train_transforms = build_transforms(
        [dict(**t) for t in cfg.augmentation['train']]
    )
    train_dataset = DUODataset(
        img_dir=cfg.dataset['train']['img_dir'],
        ann_file=cfg.dataset['train']['ann_file'],
        transforms=train_transforms,
        img_size=cfg.dataset['img_size'],
    )

    train_sampler = None
    if distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.train['batch_size'],
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=cfg.dataset['num_workers'],
        collate_fn=DUODataset.collate_fn,
        pin_memory=True,
    )

    # Build model
    model = HyUOD(**cfg.model).to(device)
    if distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank]
        )

    # Optimizer
    opt_cfg = cfg.train['optimizer']
    optimizer = optim.SGD(
        model.parameters(),
        lr=opt_cfg['lr'],
        momentum=opt_cfg['momentum'],
        weight_decay=opt_cfg['weight_decay'],
    )

    # LR scheduler
    sched_cfg = cfg.train['lr_scheduler']
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=sched_cfg['milestones'],
        gamma=sched_cfg['gamma'],
    )

    # Resume from checkpoint
    start_epoch = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        start_epoch = ckpt.get('epoch', 0) + 1
        print(f'Resumed from checkpoint: {args.resume} (epoch {start_epoch})')

    os.makedirs(cfg.checkpoint['save_dir'], exist_ok=True)

    # Training loop
    for epoch in range(start_epoch, cfg.train['num_epochs']):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        model.train()
        epoch_start = time.time()

        for iter_idx, batch in enumerate(train_loader):
            imgs = batch['img'].to(device)
            gt_bboxes = [b.to(device) for b in batch['gt_bboxes']]
            gt_labels = [l.to(device) for l in batch['gt_labels']]
            img_metas = batch['img_metas']

            # Warmup learning rate
            total_iters = epoch * len(train_loader) + iter_idx
            warmup_cfg = cfg.train.get('warmup', {})
            if total_iters < warmup_cfg.get('iters', 0):
                warmup_ratio = (
                    warmup_cfg['ratio'] +
                    (1 - warmup_cfg['ratio']) * total_iters / warmup_cfg['iters']
                )
                for pg in optimizer.param_groups:
                    pg['lr'] = opt_cfg['lr'] * warmup_ratio

            losses = model(imgs, img_metas, gt_bboxes, gt_labels)
            total_loss = sum(losses.values())

            optimizer.zero_grad()
            total_loss.backward()
            if cfg.train.get('clip_grad_norm'):
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), cfg.train['clip_grad_norm']
                )
            optimizer.step()

            if iter_idx % cfg.log['interval'] == 0:
                loss_str = '  '.join(
                    f'{k}: {v.item():.4f}' for k, v in losses.items()
                )
                print(
                    f'Epoch [{epoch+1}/{cfg.train["num_epochs"]}] '
                    f'Iter [{iter_idx+1}/{len(train_loader)}]  '
                    f'{loss_str}  '
                    f'lr: {optimizer.param_groups[0]["lr"]:.6f}'
                )

        scheduler.step()

        # Save checkpoint
        if (epoch + 1) % cfg.checkpoint['interval'] == 0:
            ckpt_path = os.path.join(
                cfg.checkpoint['save_dir'], f'epoch_{epoch+1}.pth'
            )
            torch.save(
                dict(model=model.state_dict(),
                     optimizer=optimizer.state_dict(),
                     epoch=epoch),
                ckpt_path,
            )
            # Keep a symlink to the latest checkpoint
            latest = os.path.join(cfg.checkpoint['save_dir'], 'latest.pth')
            if os.path.islink(latest):
                os.remove(latest)
            os.symlink(os.path.abspath(ckpt_path), latest)

        elapsed = time.time() - epoch_start
        print(f'Epoch {epoch+1} finished in {elapsed:.1f}s')

    print('Training complete.')


if __name__ == '__main__':
    main()
