import torch.nn as nn
import torch
import torchvision.models as models
# import torchvision.transforms as transforms
# from dataset import GAN_Data

class RWMAB(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, (3, 3), stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels, in_channels, (3, 3), stride=1, padding=1),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, (1, 1), stride=1, padding=0),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # print('In RWMAB')
        x_ = self.layer1(x)
        x__ = self.layer2(x_)

        x = x__ * x_ + x

        return x


class ShortResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()

        self.layers = nn.ModuleList([RWMAB(in_channels) for _ in range(16)])

    def forward(self, x):
        # print('In Short Residual Block')
        x_ = x.clone()

        for layer in self.layers:
            x_ = layer(x_)

        return x_ + x


class Generator(nn.Module):
    def __init__(self, in_channels=3, blocks=8, scale = 4):
        super().__init__()

        # Add the noise part here

        self.conv = nn.Conv2d(in_channels, 64, (3, 3), stride=1, padding=1)

        self.short_blocks = nn.ModuleList(
            [ShortResidualBlock(64) for _ in range(blocks)]
        )

        self.conv2 = nn.Conv2d(64, 64, (1, 1), stride=1, padding=0)

        # if(scale == 4):
        #     upsample_blocks = [Upsampler(channel = 64, kernel_size = 3, scale = 2, act = nn.PReLU()) for _ in range(2)]
        # else:
        #     upsample_blocks = [Upsampler(channel = 64, kernel_size = 3, scale = scale, act = nn.PReLU())]

        # self.tail = nn.Sequential(*upsample_blocks)
        
        # self.last_conv = conv(in_channel = 64, out_channel = 3, kernel_size = 3, BN = False, act = nn.Tanh())
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, (3, 3), stride=1, padding=1),
            nn.PixelShuffle(2),
            nn.LeakyReLU(),
            nn.Conv2d(64, 256, (3, 3), stride=1, padding=1),
            nn.PixelShuffle(2),  # Remove if output is 2x the input
            nn.LeakyReLU(),
            nn.Conv2d(64, 3, (1, 1), stride=1, padding=0),  # Change 64 -> 256
            nn.Tanh(),
            # nn.Sigmoid(),
        )

    def forward(self, x):
        # print('In Generator')
        x = self.conv(x)
        x_ = x.clone()

        for layer in self.short_blocks:
            x_ = layer(x_)

        x = torch.cat([self.conv2(x_), x], dim=1)

        x = self.conv3(x)

        return x

# import torch
# import torch.nn as nn
# from ops import *


# class Generator(nn.Module):
    
#     def __init__(self, img_feat = 3, n_feats = 64, kernel_size = 3, num_block = 16, act = nn.PReLU(), scale=4):
#         super(Generator, self).__init__()
        
#         self.conv01 = conv(in_channel = img_feat, out_channel = n_feats, kernel_size = 9, BN = False, act = act)
        
#         resblocks = [ResBlock(channels = n_feats, kernel_size = 3, act = act) for _ in range(num_block)]
#         self.body = nn.Sequential(*resblocks)
        
#         self.conv02 = conv(in_channel = n_feats, out_channel = n_feats, kernel_size = 3, BN = True, act = None)
        
#         if(scale == 4):
#             upsample_blocks = [Upsampler(channel = n_feats, kernel_size = 3, scale = 2, act = act) for _ in range(2)]
#         else:
#             upsample_blocks = [Upsampler(channel = n_feats, kernel_size = 3, scale = scale, act = act)]

#         self.tail = nn.Sequential(*upsample_blocks)
        
#         self.last_conv = conv(in_channel = n_feats, out_channel = img_feat, kernel_size = 3, BN = False, act = nn.Tanh())
        
#     def forward(self, x):
        
#         x = self.conv01(x)
#         _skip_connection = x
        
#         x = self.body(x)
#         x = self.conv02(x)
#         feat = x + _skip_connection
        
#         x = self.tail(feat)
#         x = self.last_conv(x)
        
#         return x, feat
