import torch
import torch.nn as nn


class UnderwaterImageEnhancer(nn.Module):
    """Lightweight real-time underwater image enhancement (UIE) module.

    Uses a sequence of depthwise-separable convolutions with residual
    connections to enhance low-contrast underwater images before passing
    them to the main detector backbone.

    The architecture is intentionally lightweight so that it adds minimal
    computational overhead during end-to-end training.
    """

    def __init__(self, num_channels=32):
        super(UnderwaterImageEnhancer, self).__init__()
        self.encoder = nn.Sequential(
            _DepthwiseSeparableConv(3, num_channels, stride=1),
            _DepthwiseSeparableConv(num_channels, num_channels * 2, stride=2),
            _DepthwiseSeparableConv(num_channels * 2, num_channels * 4, stride=2),
        )
        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            _DepthwiseSeparableConv(num_channels * 4, num_channels * 2, stride=1),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            _DepthwiseSeparableConv(num_channels * 2, num_channels, stride=1),
        )
        self.output_conv = nn.Conv2d(num_channels, 3, kernel_size=3, padding=1)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        """Enhance an underwater image.

        Args:
            x (Tensor): Input RGB image tensor of shape (N, 3, H, W),
                values expected in [0, 1].

        Returns:
            Tensor: Enhanced image of the same shape as ``x``, clipped to [0, 1].
        """
        feat = self.encoder(x)
        feat = self.decoder(feat)
        residual = torch.tanh(self.output_conv(feat))
        return (x + residual).clamp(0.0, 1.0)


class _DepthwiseSeparableConv(nn.Module):
    """Depthwise separable convolution: depthwise conv + pointwise conv."""

    def __init__(self, in_channels, out_channels, stride=1):
        super(_DepthwiseSeparableConv, self).__init__()
        self.dw = nn.Conv2d(
            in_channels, in_channels, kernel_size=3, stride=stride,
            padding=1, groups=in_channels, bias=False
        )
        self.pw = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU6(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.pw(self.dw(x))))
