import torch
from torch import nn
from torch.nn import functional as F
from attention import SelfAttention

'''Layer Normalization: makes sure each layer always see the same distribution of the data, 
distributed around 0 with variance 1. Especially when certain layers have a completely outlier distribution
that can heavily delay the training. Normalizes across all features/channels for each individual data sample,
which include Channels, Height, and Width

Group Normalization: Normalizes over groups of channels within each sample, Similar to LayerNor but applied per group
So divide channels into G groups. Better because its often that pixels closer have a much stronger relation, while pixels
way far apart may not, so group norm takes the win.

Just dont want the layers to osciallte too much, otherwise the loss function will oscillate too'''

class VAE_AttentionBlock(nn.Module):

    def __init__(self, channels: int):
        super().__init__()
        self.groupnorm = nn.GroupNorm(32, channels)
        self.attention = SelfAttention(1, channels)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor: 
        # x: (Batch_Size, Features, Height, Width)

        residue = x
        n, c, h, w = x.shape

        #self attention between all the pixels (own embedding with representation in features) in input image
        # (Batch_Size, Features, Height, Width) -> (Batch_Size, Features, Height * Width)
        x = x.view(n, c, h*w)

        # (Batch_Size, Features, Height * Width) -> (Batch_Size, Height * Width, Features)
        x = x.transpose(-1, -2)

        # (Batch_Size, Height * Width, Features) -> (Batch_Size, Height * Width, Features)
        x = self.attention(x)

        # (Batch_Size, Height * Width, Features) -> (Batch_Size, Features, Height * Width)
        x = self.transpose(-1, -2)

        # (Batch_Size, Features, Height * Width) -> (Batch_Size, Features, Height, Width)
        x = x.view((n, c, h, w))

        x += residue

        return x #tensor of the shape above
class VAE_ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.groupnorm_1 = nn.GroupNorm(32, in_channels)
        self.conv_1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        self.groupnorm_2 * nn.GroupNorm(32, out_channels)
        self.conv_2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        #skip connection where we skip some layers and connect to last output of the layer

        if in_channels == out_channels:
            self.residual_layer = nn.Identity()
        else:
            self.residual_layer = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (Batch_Size, In_Channels, Height, Width)


        residue = x

        x = self.groupnorm_1(x)

        x = F.silu(x)

        x = self.conv_1(x)

        x = self.groupnorm_2(x)

        x = F.silu(x)

        x = self.conv_2(x)

        # if we do x + residue it will only works if the number of in channels is the same as out channels
        return x + self.residual_layer(residue) #allows different in channel and out channels

class VAE_Decoder(nn.Sequential):

    def __init__(self):
        super().__init__(
            nn.Conv2d(4, 4, kernel_size=1, padding=0),

            nn.Conv2d(4, 512, kernel_size=3, padding=1),

            VAE_ResidualBlock(512, 512),

            VAE_AttentionBlock(512),

            VAE_ResidualBlock(512, 512),

            VAE_ResidualBlock(512, 512),

            # (Batch_Size, 512, Height / 8, Width / 8) -> (Batch_Size, 512, Height / 8, Width / 8)
            VAE_ResidualBlock(512, 512),

            # (Batch_Size, 512, Height / 8, Width / 8) -> (Batch_Size, 512, Height / 4, Width / 4)
            nn.Upsample(scale_factor=2), #resize an image, replicates the pixels along the dimensions by that factor given

            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            
            VAE_ResidualBlock(512, 512),
            VAE_ResidualBlock(512, 512),
            VAE_ResidualBlock(512, 512),

            # (Batch_Size, 512, Height / 4, Width / 4) -> (Batch_Size, 512, Height / 2, Width / 2)
            nn.Upsample(scale_factor=2),

            nn.Conv2d(512, 512, kernel_size=3, padding=1),

            VAE_ResidualBlock(512, 256),
            VAE_ResidualBlock(256, 256),
            VAE_ResidualBlock(256, 256),

            # (Batch_Size, 256, Height / 2, Width / 2) -> (Batch_Size, 256, Height, Width)
            nn.Upsample(scale_factor=2),

            nn.Conv2d(256, 256, kernel_size=3, padding=1),

            VAE_ResidualBlock(256, 128),
            VAE_ResidualBlock(256, 128),
            VAE_ResidualBlock(256, 128),

            nn.GroupNorm(32, 128),

            nn.SiLU(),

            # (Batch_Size, 128, Height, Width) -> (Batch_Size, 3, Height, Width)
            nn.Conv2d(128, 3, kernel_size=3, padding=1)

        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (Batch_Size, 4, Height / 8, Width / 8)

        x /= 0.18215

        for module in self:
            x = module(x)

        # (Batch_Size, 3, Height, Width)
        return x