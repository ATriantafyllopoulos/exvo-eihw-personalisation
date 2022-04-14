# The MIT License

# Copyright (c) 2018-2020 Qiuqiang Kong

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

import audobject
import torch
import torch.nn.functional as F


class config:
    r"""Get/set defaults for the :mod:`audfoo` module."""

    transforms = {
        16000: {
            'n_fft': 512,
            'win_length': 512,
            'hop_length': 160,
            'n_mels': 64,
            'f_min': 50,
            'f_max': 8000,
        }
    }


def init_layer(layer):
    """Initialize a Linear or Convolutional layer. """
    torch.nn.init.xavier_uniform_(layer.weight)

    if hasattr(layer, 'bias'):
        if layer.bias is not None:
            layer.bias.data.fill_(0.)


def init_bn(bn):
    """Initialize a Batchnorm layer. """
    bn.bias.data.fill_(0.)
    bn.weight.data.fill_(1.)


class ConvBlock(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int):

        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.conv1 = torch.nn.Conv2d(in_channels=self.in_channels,
                                     out_channels=self.out_channels,
                                     kernel_size=(3, 3), stride=(1, 1),
                                     padding=(1, 1), bias=False)

        self.conv2 = torch.nn.Conv2d(in_channels=self.out_channels,
                                     out_channels=self.out_channels,
                                     kernel_size=(3, 3), stride=(1, 1),
                                     padding=(1, 1), bias=False)

        self.bn1 = torch.nn.BatchNorm2d(self.out_channels)
        self.bn2 = torch.nn.BatchNorm2d(self.out_channels)

        self.init_weight()

    def init_weight(self):
        init_layer(self.conv1)
        init_layer(self.conv2)
        init_bn(self.bn1)
        init_bn(self.bn2)

    def forward(self, x, pool_size=(2, 2), pool_type='avg'):

        x = F.relu_(self.bn1(self.conv1(x)))
        x = F.relu_(self.bn2(self.conv2(x)))
        if pool_type == 'max':
            x = F.max_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'avg':
            x = F.avg_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'avg+max':
            x1 = F.avg_pool2d(x, kernel_size=pool_size)
            x2 = F.max_pool2d(x, kernel_size=pool_size)
            x = x1 + x2
        else:
            raise Exception('Incorrect argument!')

        return x


class Cnn14(torch.nn.Module, audobject.Object):
    r"""Cnn14 model architecture.

    Args:
        sampling_rate: feature extraction is configurable
            based on sampling rate
        output_dim: number of output classes to be used
        sigmoid_output: whether output should be passed through
            a sigmoid. Useful for multi-label problems
        segmentwise: whether output should be returned per-segment
            or aggregated over the entire clip
        in_channels: number of input channels
    """

    def __init__(
        self,
        output_dim: int,
        sigmoid_output: bool = False,
        segmentwise: bool = False,
        in_channels: int = 1
    ):

        super().__init__()

        self.output_dim = output_dim
        self.sigmoid_output = sigmoid_output
        self.segmentwise = segmentwise
        self.in_channels = in_channels

        self.bn0 = torch.nn.BatchNorm2d(64)
        self.conv_block1 = ConvBlock(in_channels=in_channels, out_channels=64)
        self.conv_block2 = ConvBlock(in_channels=64, out_channels=128)
        self.conv_block3 = ConvBlock(in_channels=128, out_channels=256)
        self.conv_block4 = ConvBlock(in_channels=256, out_channels=512)
        self.conv_block5 = ConvBlock(in_channels=512, out_channels=1024)
        self.conv_block6 = ConvBlock(in_channels=1024, out_channels=2048)

        self.fc1 = torch.nn.Linear(2048, 2048, bias=True)
        self.out = torch.nn.Linear(2048, self.output_dim, bias=True)

        self.init_weight()

    def init_weight(self):
        init_bn(self.bn0)
        init_layer(self.fc1)
        init_layer(self.out)

    def get_embedding(self, x):
        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)

        x = self.conv_block1(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block2(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block3(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block4(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block5(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block6(x, pool_size=(1, 1), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = torch.mean(x, dim=3)

        if self.segmentwise:
            return self.segmentwise_path(x)
        else:
            return self.clipwise_path(x)

    def segmentwise_path(self, x):
        x1 = F.max_pool1d(x, kernel_size=3, stride=1, padding=1)
        x2 = F.avg_pool1d(x, kernel_size=3, stride=1, padding=1)
        x = x1 + x2
        x = F.dropout(x, p=0.5, training=self.training)
        x = x.transpose(1, 2)
        x = F.relu_(self.fc1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        return x

    def clipwise_path(self, x):
        (x1, _) = torch.max(x, dim=2)
        x2 = torch.mean(x, dim=2)
        x = x1 + x2

        x = F.relu_(self.fc1(x))
        return x

    def forward(self, x):
        x = self.get_embedding(x)
        x = self.out(x)
        if self.sigmoid_output:
            x = torch.sigmoid(x)
        return x


class Cnn10(torch.nn.Module, audobject.Object):
    def __init__(
        self,
        output_dim: int,
        sigmoid_output: bool = False,
        segmentwise: bool = False
    ):

        super().__init__()
        self.output_dim = output_dim
        self.sigmoid_output = sigmoid_output
        self.segmentwise = segmentwise

        self.bn0 = torch.nn.BatchNorm2d(64)

        self.conv_block1 = ConvBlock(in_channels=1, out_channels=64)
        self.conv_block2 = ConvBlock(in_channels=64, out_channels=128)
        self.conv_block3 = ConvBlock(in_channels=128, out_channels=256)
        self.conv_block4 = ConvBlock(in_channels=256, out_channels=512)

        self.fc1 = torch.nn.Linear(512, 512, bias=True)
        self.out = torch.nn.Linear(512, output_dim, bias=True)

        self.init_weight()

    def init_weight(self):
        init_bn(self.bn0)
        init_layer(self.fc1)
        init_layer(self.out)

    def get_embedding(self, x):
        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)

        x = self.conv_block1(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block2(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block3(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block4(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = torch.mean(x, dim=3)

        if self.segmentwise:
            return self.segmentwise_path(x)
        else:
            return self.clipwise_path(x)

    def segmentwise_path(self, x):
        x1 = F.max_pool1d(x, kernel_size=3, stride=1, padding=1)
        x2 = F.avg_pool1d(x, kernel_size=3, stride=1, padding=1)
        x = x1 + x2
        x = F.dropout(x, p=0.5, training=self.training)
        x = x.transpose(1, 2)
        x = F.relu_(self.fc1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        return x

    def clipwise_path(self, x):
        (x1, _) = torch.max(x, dim=2)
        x2 = torch.mean(x, dim=2)
        x = x1 + x2

        x = F.relu_(self.fc1(x))
        return x

    def forward(self, x):
        x = self.get_embedding(x)
        x = self.out(x)
        if self.sigmoid_output:
            x = torch.sigmoid(x)
        return x
