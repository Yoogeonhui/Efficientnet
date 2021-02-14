from torch import nn
import torch
import math


#가장 헷갈렸던 round filters, 가장 가까운 depth_divisor 배수로 채널 크기를 설정함. 단 0.9배 미만이면 올려준다.
def round_filters(filters, width, depth_divisor):
    filters *= width
    new_filters = int(filters+depth_divisor/2) // depth_divisor * depth_divisor
    new_filters = max(new_filters, depth_divisor)

    if new_filters < 0.9 * filters:
        new_filters += depth_divisor
    return int(new_filters)


def round_repeats(repeats, depth):
    return int(math.ceil(depth * repeats))

