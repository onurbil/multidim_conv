import torch
import torch.nn.functional as F
from torch import nn
from models.attention_augmented_conv import AugmentedConv
from models.layers import DepthwiseSeparableConv
from einops import rearrange


class MultidimConv(nn.Module):
    def __init__(self, channels, height, width, kernel_size=3, kernels_per_layer=16, padding=1):
        super(MultidimConv, self).__init__()
        self.normal = DepthwiseSeparableConv(channels, channels, kernels_per_layer=kernels_per_layer,
                                             kernel_size=kernel_size, padding=padding)
        self.horizontal = DepthwiseSeparableConv(height, height, kernels_per_layer=kernels_per_layer,
                                                 kernel_size=kernel_size, padding=padding)
        self.vertical = DepthwiseSeparableConv(width, width, kernels_per_layer=kernels_per_layer,
                                               kernel_size=kernel_size, padding=padding)
        self.bn_normal = nn.BatchNorm2d(channels)
        self.bn_horizontal = nn.BatchNorm2d(height)
        self.bn_vertical = nn.BatchNorm2d(width)

    def forward(self, x):
        x_normal = self.normal(x)

        # r_re = rearrange(x, "b c h w -> b h c w")
        r_re = rearrange(x, "b c h w -> b w h c")
        x_horizontal = self.horizontal(r_re)  # x.permute(0,2,1,3)

        # v_re = rearrange(x, "b c h w -> b w c h")
        v_re = rearrange(x, "b c h w -> b h c w")
        x_vertical = self.vertical(v_re)  # x.permute(0,3,1,2)

        x_normal = F.relu(self.bn_normal(x_normal))
        x_horizontal = F.relu(self.bn_horizontal(x_horizontal))
        x_vertical = F.relu(self.bn_vertical(x_vertical))

        output = torch.cat([rearrange(x_normal, "b c h w -> b (c h w)"),
                            rearrange(x_horizontal, "b w h c -> b (c h w)"),
                            rearrange(x_vertical, "b h c w -> b (c h w)")
                            ], dim=1)
        return output


class MultidimConvNetwork(nn.Module):
    def __init__(self, channels, height, width, output_channels, kernel_size=3, kernels_per_layer=16, padding=1,
                 hidden_neurons=128):
        super(MultidimConvNetwork, self).__init__()
        self.multidim = MultidimConv(channels, height, width, kernel_size=kernel_size,
                                     kernels_per_layer=kernels_per_layer, padding=padding)

        self.merge = nn.Linear(3 * channels * width * height, hidden_neurons)
        self.output = nn.Linear(hidden_neurons, output_channels)

    def forward(self, x):
        output = self.multidim(x)
        output = F.relu(self.merge(output))
        output = self.output(output)
        return output