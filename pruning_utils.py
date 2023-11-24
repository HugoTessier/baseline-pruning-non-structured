import torch
import math
import numpy as np


class MaskedConv(torch.nn.Module):
    def __init__(self, conv: torch.nn.Conv2d, mask):
        super().__init__()
        self.conv = conv
        self.mask = mask

    def forward(self, x):
        return torch.nn.functional.conv2d(input=x,
                                          weight=self.conv.weight * self.mask,
                                          bias=self.conv.bias,
                                          stride=self.conv.stride,
                                          padding=self.conv.padding,
                                          dilation=self.conv.dilation,
                                          groups=self.conv.groups)


class MaskedLinear(torch.nn.Module):
    def __init__(self, linear: torch.nn.Linear, mask):
        super().__init__()
        self.linear = linear
        self.mask = mask

    def forward(self, x):
        return torch.nn.functional.linear(input=x,
                                          weight=self.linear.weight * self.mask,
                                          bias=self.linear.bias)


def create_regular_mask(parameters: torch.Tensor, rate: float):
    period = round(1 / (1 - rate))
    squeezed_parameters = parameters.squeeze()
    prime = math.gcd(squeezed_parameters.shape[-1], period) == 1
    shape = list(squeezed_parameters.shape)
    if not prime:
        shape[-1] += 1
    mask = torch.zeros(math.prod(shape), device=squeezed_parameters.device)
    mask[::period] = 1
    mask = mask.view(shape)
    if not prime:
        mask = mask.view(-1, mask.shape[-1])[:, :-1].view(squeezed_parameters.shape)
    mask = mask.view(parameters.shape)
    return mask


def create_random_mask(parameters: torch.Tensor, rate: float):
    mask = torch.zeros(parameters.numel(), device=parameters.device)
    indices = np.random.choice(np.arange(len(mask)), int(len(mask) * rate), replace=False)
    mask[indices] = 1
    mask = mask.view(parameters.shape)
    return mask


def create_mask(m: torch.nn.Module, pruning_rate: float, pruning_type: str):
    if pruning_type == 'regular':
        return create_regular_mask(m.weight, pruning_rate)
    elif pruning_type == 'random':
        return create_random_mask(m.weight, pruning_rate)
    else:
        raise ValueError


def mask_network(net: torch.nn.Module, pruning_rate, pruning_type):
    for n, m in net.named_children():
        if isinstance(m, torch.nn.Conv2d):
            mask = create_mask(m, pruning_rate, pruning_type)
            setattr(net, n, MaskedConv(m, mask))
        elif isinstance(m, torch.nn.Linear):
            mask = create_mask(m, pruning_rate, pruning_type)
            setattr(net, n, MaskedLinear(m, mask))
        else:
            mask_network(m, pruning_rate, pruning_type)


def apply_pruning_and_unmask_network(net: torch.nn.Module):
    for n, m in net.named_children():
        if isinstance(m, MaskedConv):
            conv = m.conv
            conv.weight.data *= m.mask
            setattr(net, n, conv)
        elif isinstance(m, MaskedLinear):
            linear = m.linear
            linear.weight.data *= m.mask
            setattr(net, n, linear)
        else:
            apply_pruning_and_unmask_network(m)
