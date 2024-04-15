import torch
import torch.nn as nn

class DownBlock(nn.Module):
    r"""
    
        A downblock's job is to convert a tensor of shape (B, C, H, W) to a tensor of shape (B, 2C, H/2, W/2)


    """
    def __init__(self, in_channels, out_channels, norm_channels, num_layers):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.norm_channels = norm_channels
        self.num_layers = num_layers
        self.resnet = nn.ModuleList(
            [
                nn.Sequential(
                    nn.GroupNorm(norm_channels, in_channels if i == 0 else out_channels),
                    nn.SiLU(),
                    nn.Conv2d(in_channels if i == 0 else out_channels, out_channels, kernel_size=3, stride = 1, padding=1),
                )
            ] for i in range(num_layers)
        )

        self.resnet_1d_conv = nn.Conv1d(in_channels if i == 0 else out_channels, out_channels, kernel_size=1)
        

    def forward(self, x):
        out = x
        for i in range(self.num_layers):
            resnet_block_input = out
            out = self.resnet[i](out)
            out = out + self.resnet_1d_conv(resnet_block_input)



        return out
