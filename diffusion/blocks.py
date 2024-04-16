import torch
import torch.nn as nn

class DownBlock(nn.Module):
    r"""
    
        A downblock converts a tensor of shape (B, C, H, W) into 

    """
    def __init__(self, in_channels, out_channels, norm_channels, num_layers, t_embedding_dim):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.norm_channels = norm_channels
        self.num_layers = num_layers
        self.t_embedding_dim = t_embedding_dim
        self.resnet = nn.ModuleList(
            [
                nn.Sequential(
                    nn.GroupNorm(norm_channels, in_channels if i == 0 else out_channels),
                    nn.SiLU(),
                    nn.Conv2d(in_channels if i == 0 else out_channels, out_channels, kernel_size=3, stride = 1, padding=1),
                )
            ] for i in range(num_layers)
        )
        if t_embedding_dim is not None:
            self.t_embedding = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.SiLU(),
                        nn.Linear(t_embedding_dim, out_channels)
                    )
                ]
            )

        self.resnet_1d_conv = nn.ModuleList(
            [
                nn.Conv1d(in_channels if i == 0 else out_channels, out_channels, kernel_size=1)
            ] for i in range(num_layers)
        )
            

    def forward(self, x, t_embedding=None):
        out = x
        for i in range(self.num_layers):
            resnet_block_input = out
            out = out + self.t_embedding[i]
        out = self.resnet[i](out)
        out = out + self.resnet_1d_conv(resnet_block_input)
        return out

