import torch.nn as nn
import torchvision.models as tv_models
from torchvision.models import ResNet50_Weights, ResNet101_Weights


class Bottleneck(nn.Module):
    """ResNet Bottleneck block."""

    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(
            planes, planes * self.expansion, kernel_size=1, bias=False
        )
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class ResNet(nn.Module):
    """ResNet backbone for feature extraction.

    Outputs C2, C3, C4, C5 feature maps (stride 4, 8, 16, 32).

    Args:
        depth (int): Network depth, from {50, 101}. Default: 50.
        pretrained (str, optional): Path to pretrained weights.
            If None, uses ImageNet-pretrained torchvision weights.
    """

    arch_settings = {
        50: (tv_models.resnet50, ResNet50_Weights.IMAGENET1K_V1),
        101: (tv_models.resnet101, ResNet101_Weights.IMAGENET1K_V1),
    }

    def __init__(self, depth=50, pretrained=None):
        super(ResNet, self).__init__()
        if depth not in self.arch_settings:
            raise ValueError(f'Unsupported ResNet depth: {depth}. '
                             f'Supported depths: {list(self.arch_settings)}')

        model_fn, default_weights = self.arch_settings[depth]
        weights = default_weights if pretrained is None else None
        base = model_fn(weights=weights)

        self.layer0 = nn.Sequential(base.conv1, base.bn1, base.relu, base.maxpool)
        self.layer1 = base.layer1  # C2: stride 4,  256 ch
        self.layer2 = base.layer2  # C3: stride 8,  512 ch
        self.layer3 = base.layer3  # C4: stride 16, 1024 ch
        self.layer4 = base.layer4  # C5: stride 32, 2048 ch

        if pretrained is not None:
            self._load_pretrained(pretrained)

    def _load_pretrained(self, path):
        import torch
        state_dict = torch.load(path, map_location='cpu')
        if 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']
        self.load_state_dict(state_dict, strict=False)

    def forward(self, x):
        x = self.layer0(x)
        c2 = self.layer1(x)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)
        return (c2, c3, c4, c5)
