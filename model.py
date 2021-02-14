import copy
from collections import OrderedDict

import torch
from torch import nn

from utils import round_filters, round_repeats
from efficientnet_pytorch import EfficientNet as org_eff
import random


class MBConvConfig:
    def __init__(self, in_channels, out_channels, expansion_ratio, stride, kernel_size, se_ratio, dropout_rate):
        self.in_channels, self.out_channels, self.expansion_ratio, self.stride, self.kernel_size, self.se_ratio, self.dropout_rate=\
            in_channels, out_channels, expansion_ratio, stride, kernel_size, se_ratio, dropout_rate
        self.padding = (kernel_size-1)//2

    def get_zero(self):
        self.in_channels = self.out_channels
        self.stride = 1


class EfficientNetConfig:
    def __init__(self, depth=1.0, width=1.0, resolution=1.0, dropout_rate=0.2,
                 batch_norm_momentum=0.99, num_classes=1000):
        self.depth, self.width, self.resolution, self.dropout_rate, self.batch_norm_momentum, self.num_classes = \
            depth, width, resolution, dropout_rate, batch_norm_momentum, num_classes


class MBConv(nn.Module):
    def __init__(self, block_args:MBConvConfig, config:EfficientNetConfig):
        super(MBConv, self).__init__()
        # SE + MobileConv2
        mid_size = block_args.in_channels*block_args.expansion_ratio

        # Expanded channel size
        if block_args.expansion_ratio != 1:
            self.expand = \
                nn.Sequential(nn.Conv2d(in_channels=block_args.in_channels,
                                        out_channels=mid_size,
                                        kernel_size=1,
                                        bias=False),
                              nn.BatchNorm2d(num_features=mid_size, momentum=config.batch_norm_momentum),
                              nn.SiLU()
                )
        else:
            self.expand = None
        self.dwconv = nn.Sequential(
                        nn.Conv2d(in_channels=mid_size,
                                  out_channels=mid_size,
                                  groups=mid_size,
                                  kernel_size=block_args.kernel_size,
                                  stride=block_args.stride,
                                  padding=block_args.padding,
                                  bias=False
                                ),
                        nn.BatchNorm2d(num_features=mid_size,
                                       momentum=config.batch_norm_momentum),
                        nn.SiLU()
                    )

        # SE Layer
        reduced_filters = max(1, int(block_args.in_channels * block_args.se_ratio))
        self.se_layers = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=mid_size,
                      out_channels=reduced_filters,
                      kernel_size=1),
            nn.SiLU(),
            nn.Conv2d(in_channels=reduced_filters,
                      out_channels=mid_size,
                      kernel_size=1),
            nn.Sigmoid()
        )

        self.last_conv = nn.Sequential(
            nn.Conv2d(in_channels=mid_size,
                      out_channels=block_args.out_channels,
                      kernel_size=1,
                      bias=False),
            nn.BatchNorm2d(num_features=block_args.out_channels, momentum=config.batch_norm_momentum)
        )

        self.dropout_phase = nn.Dropout(block_args.dropout_rate)
        self.is_same = (block_args.in_channels == block_args.out_channels) and block_args.stride == 1

    def forward(self, inputs):
        if self.training and random.random() >= 0.8:
            return inputs
        x = inputs
        if self.expand is not None:
            x = self.expand(x)
        x = self.dwconv(x)
        x = x * self.se_layers(x)
        x = self.last_conv(x)
        if self.is_same:
            x = self.dropout_phase(x)
            x = x + inputs
        return x


class EfficientNet(nn.Module):
    baseline = [[16, 1, 3, 1, 1],
                [24, 2, 3, 6, 2],
                [40, 2, 5, 6, 2],
                [80, 2, 3, 6, 3],
                [112, 1, 5, 6, 3],
                [192, 2, 5, 6, 4],
                [320, 1, 3, 6, 1]] # output_channel, stride, kernel_size, expansion_rate, num_layers

    def __init__(self, config:EfficientNetConfig):
        super(EfficientNet, self).__init__()
        bef_filter_num = round_filters(32, config.width, 8)
        self.modules = [nn.Sequential(nn.Conv2d(in_channels=3,
                                                out_channels=bef_filter_num,
                                                kernel_size=3,
                                                padding=1,
                                                stride=2,
                                                bias=False),
                                      nn.BatchNorm2d(num_features=bef_filter_num, momentum=config.batch_norm_momentum),
                                      nn.SiLU()
                                     )
                        ]
        sum_layers = 0
        for k in self.baseline:
            k[0] = round_filters(k[0], config.width, 8)
            k[4] = round_repeats(k[4], config.depth)
            sum_layers += k[4]

        cnt_layers = 0
        for k in self.baseline:
            channels, stride, kernel_size, expansion_ratio, num_layers = k
            cur_dropout = (cnt_layers / sum_layers) * config.dropout_rate
            block_conf = MBConvConfig(in_channels=bef_filter_num, out_channels=channels,
                                      kernel_size=kernel_size, stride=stride, expansion_ratio=expansion_ratio, se_ratio=0.25, dropout_rate=cur_dropout)
            self.modules.append(MBConv(block_conf, config))
            cnt_layers += 1
            if num_layers > 1:
                new_conf = copy.deepcopy(block_conf)
                new_conf.get_zero()
                for _ in range(num_layers-1):
                    cur_dropout = (cnt_layers / sum_layers) * config.dropout_rate
                    new_conf.dropout_rate = cur_dropout
                    self.modules.append(MBConv(new_conf, config))
                    cnt_layers += 1
            bef_filter_num = channels

        # End of Conv
        final_channel = round_filters(1280, width=config.width, depth_divisor=8)
        self.modules.append(
                                nn.Sequential(
                                    nn.Conv2d(in_channels=bef_filter_num,
                                              out_channels=final_channel,
                                              kernel_size=1,
                                              bias=False),
                                    nn.BatchNorm2d(num_features=final_channel,
                                                   momentum=config.batch_norm_momentum),
                                    nn.SiLU(),
                                    nn.AdaptiveAvgPool2d(1),
                                    nn.Dropout(config.dropout_rate),
                                    nn.Flatten(),
                                    nn.Linear(final_channel, config.num_classes)
                                )
        )
        self.model = nn.Sequential(*self.modules)

    def forward(self, x):
        return self.model(x)


def convert_state_dict():
    asdf = org_eff.from_pretrained('efficientnet-b0')
    effnet = EfficientNet(EfficientNetConfig(depth=1.0, width=1.0, resolution=1.15))

    org_state = asdf.state_dict()
    to_change = effnet.state_dict()
    new_state = OrderedDict()
    for org, cur in zip(org_state.items(), to_change.items()):
        new_state[cur[0]] = org[1]
    return new_state


if __name__ == '__main__':
    new_state = convert_state_dict()
    import time
    img = torch.randn([1, 3, 244, 244])
    times = {'org':[], 'my':[]}

    effnet = EfficientNet(EfficientNetConfig(depth=1.0, width=1.0, resolution=1.15))
    effnet.load_state_dict(new_state)

    torch.save(new_state, 'MyEfficientNet-b0.pth')
    print(sum(times['my']))
    print(sum(times['org']))
