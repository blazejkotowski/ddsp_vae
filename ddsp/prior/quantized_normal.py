import torch
from torch import nn
import math

class QuantizedNormal(nn.Module):
    def __init__(self, resolution, dither=True):
        super().__init__()
        self.resolution = resolution
        self.dither = dither
        self.clamp = 4

    def from_normal(self, x):
        return .5 * (1 + torch.erf(x / math.sqrt(2)))

    def to_normal(self, x):
        x = torch.erfinv(2 * x - 1) * math.sqrt(2)
        return torch.clamp(x, -self.clamp, self.clamp)

    def encode(self, x):
        x = self.from_normal(x)
        x = torch.floor(x * self.resolution)
        x = torch.clamp(x, 0, self.resolution - 1)
        return x.long()

    def to_stack_one_hot(self, x):
        x = nn.functional.one_hot(x, self.resolution)
        x = x.permute(0, 2, 1, 3)
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1).float()
        return x

    def decode(self, x):
        x = x / self.resolution
        if self.dither:
            x = x + torch.rand_like(x) / self.resolution
        x = self.to_normal(x)
        return x