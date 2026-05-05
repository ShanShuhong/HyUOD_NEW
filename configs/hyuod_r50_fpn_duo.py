# HyUOD configuration for the DUO dataset
# Training: 3x schedule (36 epochs), ResNet-50 + FPN backbone

model = dict(
    num_classes=4,
    backbone_depth=50,
    use_enhancement=True,
    pretrained=None,  # Set to a path to load pretrained backbone weights
)

# Dataset settings
dataset = dict(
    train=dict(
        img_dir='data/DUO/images/train',
        ann_file='data/DUO/annotations/train.json',
    ),
    test=dict(
        img_dir='data/DUO/images/test',
        ann_file='data/DUO/annotations/test.json',
    ),
    img_size=(800, 1333),
    num_workers=4,
)

# Training settings
train = dict(
    batch_size=2,
    num_epochs=36,
    optimizer=dict(
        type='SGD',
        lr=0.01,
        momentum=0.9,
        weight_decay=1e-4,
    ),
    lr_scheduler=dict(
        type='MultiStepLR',
        milestones=[24, 33],
        gamma=0.1,
    ),
    warmup=dict(
        iters=500,
        ratio=0.001,
    ),
    clip_grad_norm=35.0,
)

# Augmentation
augmentation = dict(
    train=[
        dict(type='Resize', size=(800, 1333), keep_ratio=True),
        dict(type='RandomHorizontalFlip', prob=0.5),
        dict(
            type='Normalize',
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
        dict(type='ToTensor'),
    ],
    test=[
        dict(type='Resize', size=(800, 1333), keep_ratio=True),
        dict(
            type='Normalize',
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
        dict(type='ToTensor'),
    ],
)

# Checkpoint & logging
checkpoint = dict(
    save_dir='work_dirs/hyuod_r50_fpn_duo',
    interval=1,
)

log = dict(
    interval=50,
)
