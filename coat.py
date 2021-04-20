import torch
from torch import nn, einsum
import torch.nn.functional as F
import math
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from module import FeedForward, ConvAttention, PreNorm
import numpy as np


class Transformer(nn.Module):

    def __init__(self, depth, dim, heads, dim_head, scale, dropout):
        super(Transformer, self).__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList([
                    PreNorm(dim, ConvAttention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                    PreNorm(dim, FeedForward(dim, dim*scale, dropout=dropout))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


class SerialBlock(nn.Module):
    
    def __init__(self, feature_size, in_channels, out_channels, depth=2, nheads=8, scale=8,
                 conv_kernel=7, stride=2, dropout=0.):
        super(SerialBlock, self).__init__()
        self.cls_embed = nn.Linear(in_channels, out_channels)
        padding = (conv_kernel -1)//2
        self.conv_embed = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, conv_kernel, stride, padding),
            Rearrange('b c h w -> b (h w) c', h = feature_size, w = feature_size),
            nn.LayerNorm(out_channels)
        )


        self.transformer = Transformer(depth=depth, dim=out_channels, heads=nheads, dim_head=out_channels//nheads,
                                       scale=scale, dropout=dropout)

    def forward(self, x, cls_tokens):
        '''

        :param x: [B C H W]
        :return: [B (H W) C]
        '''
        x = self.conv_embed(x)
        cls_tokens = self.cls_embed(cls_tokens)
        x = torch.cat((cls_tokens, x), dim=1)
        x = self.transformer(x)
        return x

class ParallelBlock(nn.Module):

    def __init__(self, in_channels, nheads=8, dropout=0.):
        super(ParallelBlock, self).__init__()


        self.p1 = PreNorm(in_channels, ConvAttention(in_channels,
                                                             heads=nheads,
                                                             dim_head=in_channels//nheads,
                                                             dropout=dropout))
        self.p2 = PreNorm(in_channels, ConvAttention(in_channels,
                                                     heads=nheads,
                                                     dim_head=in_channels // nheads,
                                                     dropout=dropout))
        self.p3 = PreNorm(in_channels, ConvAttention(in_channels,
                                                     heads=nheads,
                                                     dim_head=in_channels // nheads,
                                                     dropout=dropout))
    def forward(self, x1, x2, x3):
        '''

        :param x: [B C H W]
        :return: [B (H W) C]
        '''
        return self.p1(x1), self.p2(x2), self.p3(x3)


        

class CoaT(nn.Module):
    
    def __init__(self, in_channels, image_size, num_classes, out_channels=[64, 128, 256, 320], depths=[2, 2, 2, 2],
                 heads=8, scales=[8, 8, 4, 4], downscales=[4, 2, 2, 2], kernels=[7, 3, 3, 3], use_parallel=False,
                 parallel_depth = 6, parallel_channels=152, dropout=0.):
        super(CoaT, self).__init__()

        assert len(out_channels) == len(depths) == len(scales) == len(downscales) == len(kernels)
        feature_size = image_size
        self.cls_token = nn.Parameter(torch.randn(1, 1, in_channels))
        self.serial_layers = nn.ModuleList([])
        for out_channel, depth, scale, downscale, kernel in zip(out_channels, depths, scales, downscales, kernels):
            feature_size = feature_size // downscale
            self.serial_layers.append(
                SerialBlock(feature_size, in_channels, out_channel, depth, heads, scale, kernel, downscale, dropout)
            )
            in_channels = out_channel


        self.use_parallel = use_parallel
        if use_parallel:
            self.parallel_conv_attn = nn.ModuleList([])
            self.parallel_ffn = nn.ModuleList([])
            for _ in range(parallel_depth):
                self.parallel_conv_attn.append(ParallelBlock(parallel_channels, heads, dropout)
                )
                self.parallel_ffn.append(
                        PreNorm(parallel_channels, FeedForward(parallel_channels, parallel_channels * 4, dropout=dropout))
                        )

            self.parallel_mlp_head = nn.Sequential(
                nn.LayerNorm(in_channels*3),
                nn.Linear(in_channels*3, num_classes)
            )



        self.serial_mlp_head = nn.Sequential(
            nn.LayerNorm(in_channels),
            nn.Linear(in_channels, num_classes)
        )

    def forward(self, x):
        b, c, _, _ = x.shape
        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b)
        serial_outputs = []
        for serial_block in self.serial_layers:
            x = serial_block(x, cls_tokens)
            serial_outputs.append(x)
            cls_tokens = x[:, :1]
            l = w = int(math.sqrt(x[:, 1:].shape[1]))
            x = rearrange(x[:, 1:], 'b (l w) c -> b c l w', l=l, w=w)

        s2 = serial_outputs[1]
        s3 = serial_outputs[2]
        s4 = serial_outputs[3]
        if self.use_parallel:
            for attn, ffn in zip(self.parallel_conv_attn, self.parallel_ffn):
                s2, s3, s4 = attn(s2, s3, s4)
                cls_s2 = s2[:, :1]
                cls_s3 = s3[:, :1]
                cls_s4 = s4[:, :1]
                s2 = rearrange(s2[:,1:], 'b (l w) d -> b d l w', l=28, w=28)
                s3 = rearrange(s3[:, 1:], 'b (l w) d -> b d l w', l=14, w=14)
                s4 = rearrange(s4[:, 1:], 'b (l w) d -> b d l w', l=7, w=7)

                s2 = s2 + F.interpolate(s3, (28, 28), mode='bilinear') + F.interpolate(s4, (28, 28), mode='bilinear')
                s3 = s3 + F.interpolate(s2, (14, 14), mode='bilinear') + F.interpolate(s4, (14, 14), mode='bilinear')
                s4 = s4 + F.interpolate(s2, (7, 7), mode='bilinear') + F.interpolate(s3, (7, 7), mode='bilinear')

                s2 = rearrange(s2, 'b d l w -> b (l w) d')
                s3 = rearrange(s3, 'b d l w -> b (l w) d')
                s4 = rearrange(s4, 'b d l w -> b (l w) d')

                s2 = ffn(torch.cat([cls_s2, s2], dim=1))
                s3 = ffn(torch.cat([cls_s3, s3], dim=1))
                s4 = ffn(torch.cat([cls_s4, s4], dim=1))

            cls_tokens = torch.cat([s2[:,0], s3[:,0], s4[:,0]], dim=1)
            return self.parallel_mlp_head(cls_tokens)
        else:
            return self.serial_mlp_head(cls_tokens.squeeze(1))


if __name__ == "__main__":
    img = torch.ones([1, 3, 224, 224])

    model = CoaT(3, 224, 1000, out_channels=[152, 152, 152, 152], scales=[4, 4, 4, 4], use_parallel=True)

    out = model(img)

    print("Shape of out :", out.shape)  # [B, num_classes]


    parameters = filter(lambda p: p.requires_grad, model.parameters())
    parameters = sum([np.prod(p.size()) for p in parameters]) / 1_000_000
    print('Trainable Parameters: %.3fM' % parameters)







