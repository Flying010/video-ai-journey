#Variational Encoder
import torch
from torch import nn
from torch.nn import functional as F
from decoder import VAE_AttentionBlock, VAE_ResidualBlock

class VAE_Encoder(nn.Sequential): #encoder is a sequence of submodels
    def __init__(self):
        super().__init__( #each model reduces the dimension of data and increases the number of features
            #convert from 3 to 128 channels
            #(Batch_Size, Channel, Height, Width) -> (Batch_Size, 128, Height, Width)
            nn.Conv2d(3, 128, kernel_size=3, padding=1),

            #Residual block, combination of convolutions and normalization
            # (Batch_Size, 128, Height, Width) -> (Batch_Size, 128, Height, Width)
            VAE_ResidualBlock(128, 128),

            # (Batch_Size, 128, Height, Width) -> (Batch_Size, 128, Height, Width)
            VAE_ResidualBlock(128, 128),

            # (Batch_Size, 128, Height, Width) -> (Batch_Size, 128, Height/2, Width/2)
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=0),
            
            # (Batch_Size, 128, Height/2, Width/2) -> (Batch_Size, 256, Height/2, Width/2)
            VAE_ResidualBlock(128, 256),

            # (Batch_Size, 256, Height/2, Width/2) -> (Batch_Size, 256, Height/2, Width/2)
            VAE_ResidualBlock(256, 256),

            # (Batch_Size, 256, Height/2, Width/2) -> (Batch_Size, 256, Height/4, Width/4)
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=0),

            # (Batch_Size, 256, Height/4, Width/4) -> (Batch_Size, 512, Height/4, Width/4)
            VAE_ResidualBlock(256, 512),

            # (Batch_Size, 512, Height/4, Width/4) -> (Batch_Size, 512, Height/4, Width/4)
            VAE_ResidualBlock(512, 512),

            # (Batch_Size, 512, Height/4, Width/4) -> (Batch_Size, 512, Height/8, Width/8)
            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=0),

            # (Batch_Size, 512, Height/4, Width/4) -> (Batch_Size, 512, Height/8, Width/8)
            VAE_ResidualBlock(512, 512),

            VAE_ResidualBlock(512, 512),

            # (Batch_Size, 512, Height/8, Width/8) -> (Batch_Size, 512, Height/8, Width/8)
            VAE_ResidualBlock(512, 512),

            #self attention over each pixel, relate pixels to each other
            # (Batch_Size, 512, Height/8, Width/8) -> (Batch_Size, 512, Height/8, Width/8)
            VAE_AttentionBlock(512),

            # (Batch_Size, 512, Height/8, Width/8) -> (Batch_Size, 512, Height/8, Width/8)
            VAE_ResidualBlock(512, 512),

            #Group normalization (Batch_Size, 512, Height/8, Width/8) -> (Batch_Size, 512, Height/8, Width/8)
            nn.GroupNorm(32, 512), #32 groups, 512 channels

            nn.SiLU(), #activation function, sigmoid linear unit, similar to ReLU

            # (Batch_Size, 512, Height/8, Width/8) -> (Batch_Size, 512, Height/8, Width/8)           
            nn.Conv2d(512, 8, kernel_size=3, padding=1),

            # (Batch_Size, 8, Height/8, Width/8) -> (Batch_Size, 8, Height/8, Width/8)           
            nn.Conv2d(8, 8, kernel_size=1, padding=0)
        )
    def forward(self, x: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        # x: (Batch_Size, Channel, Height, Width)
        # noise: (Batch_Size, Out_Channels, Height / 8, Width / 8)

        #VAE learns the mu and the sigma which is the mean and variance of the distribution
        for module in self:
            if getattr(module, 'stride', None) == (2, 2): #apply to convolutions that only have stride 2
                # (Padding_Left, Padding_Right, Padding-Top, Padding_Bottom)
                x = F.pad(x, (0, 1, 0, 1))
            x = module(x)
        # (Batch_Size, 8, Height, Height / 8, Width / 8) -> two tensors of shape (Batch_Size, 4, Height / 8, Width / 8)
        mean, log_variance = torch.chunk(x, 2, dim=1)

        log_variance = torch.clamp(log_variance, -30, 20)

        variance = log_variance.exp()

        stdev = variance.sqrt()

        # Z -> N(0,1) -> N(mean, variance)?
        #X = mean + stdev*Z
        x = mean + stdev * noise
        
        #Scale the output by a constant
        x *= 0.18215

        return x
        

