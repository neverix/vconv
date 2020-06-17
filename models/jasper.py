"""
A (hopefully) generalizable speech recognition/synthesis model based on Jasper 5x5.
"""

from hparams import Map
import torch.nn as nn


class JasperModel(nn.Module):
    """
    Jasper model builder.
    Hint: look at hparams to understand more.
    """
    def __init__(self, hp: Map):
        super(JasperModel, self).__init__()
        param_names = hp.model_block_params
        params_default = hp.block_params_default
        blocks = hp.model_blocks
        self.blocks = nn.ModuleList()
        prev_channels = hp.in_channels
        for params in blocks:
            params = params + params_default[len(params):]
            hparams = Map(hp)
            for name, value in zip(param_names, params):
                hparams[name] = value
            hparams.in_channels = prev_channels
            block = ResBlock(hparams)
            self.blocks.append(block)
            prev_channels = hparams.out_channels

    def forward(self, x):
        y = x
        for block in self.blocks:
            y = block(y)
        return y


class ResBlock(nn.Module):
    """
    Jasper residual block.
    """
    def __init__(self, hp: Map):
        super(ResBlock, self).__init__()
        self.first = ConvBNRelu(Map(
            hp, in_channels=hp.in_channels, out_channels=hp.out_channels  # expanded for clarity's sake
        ))
        self.subs = nn.ModuleList()
        for i in range(hp.n_sub - 1):
            layer = ConvBNRelu(Map(
                hp, in_channels=hp.out_channels, out_channels=hp.out_channels  # same here
            ))
            self.subs.append(layer)

    def forward(self, x):
        f = self.first(x)
        y = f
        for i, sub in self.subs:
            last = i == len(self.subs) - 1
            y = sub(y, r=f if last else None)
        return y


class ConvBNRelu(nn.Module):
    """
    A single 1-D convolution + batch normalization + relu layer.
    """
    def __init__(self, hp: Map):
        super(ConvBNRelu, self).__init__()
        self.hp = hp
        padding = int(hp.dilation * (hp.kernel_size - 1) / 2)
        self.conv = nn.Conv1d(hp.in_channels, hp.out_channels,
                              kernel_size=hp.kernel_size,
                              padding=padding, bias=True,
                              stride=hp.stride, dilation=hp.dilation)
        self.bn = nn.BatchNorm1d(hp.out_channels)
        self.activation = hp.activation
        self.dropout = nn.Dropout(hp.dropout_prob)

    def forward(self, x, r=None):
        y = self.conv(x)
        y = self.bn(y)
        if r is not None:
            y += r
        y = self.activation(y)
        y = self.dropout(y)
        return y
