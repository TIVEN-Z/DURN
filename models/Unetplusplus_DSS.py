from os import name
import torch.nn.functional as F
import torch.nn as nn
import torch


class Attention_dss(nn.Module):
    def __init__(self, attention_type, in_channels=None):
        super().__init__()
        self.attention_type = attention_type


    def forward(self, x, skip):
        if self.attention_type is None:
            return torch.cat([x, skip], dim=1)
        else:
            x_mean_sig = torch.sigmoid(torch.mean(x, dim=1, keepdim=True))
            skip = skip * x_mean_sig
            return torch.cat([x, skip], dim=1)

class Attention(nn.Module):
    def __init__(self, name, **params):
        super().__init__()

        if name is None:
            self.attention = nn.Identity(**params)

    def forward(self, x):
        return self.attention(x)


class Conv2dReLU(nn.Sequential):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            padding=0,
            stride=1,
            use_batchnorm=True, 
    ):
        conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=not (use_batchnorm),  
        )
        relu = nn.ReLU(inplace=True)

        bn = nn.BatchNorm2d(out_channels)

        super(Conv2dReLU, self).__init__(conv, bn, relu)



class DecoderBlock(nn.Module):
    def __init__(
            self,
            in_channels,
            skip_channels,  
            out_channels,
            use_batchnorm=True,
            attention_type=None,
    ):
        super().__init__()
        self.conv1 = Conv2dReLU(
            in_channels + skip_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.attention_dss = Attention_dss(attention_type)
        self.conv2 = Conv2dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )


    def forward(self, x, skip=None):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        if skip is not None:
            x = self.attention_dss(x, skip)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class UnetPlusPlusDecoder(nn.Module):
    def __init__(
            self,
            encoder_channels, 
            decoder_channels,  
            n_blocks=5,
            use_batchnorm=True,
            attention_type=None,
            center=False,
    ):
        super().__init__()

        if n_blocks != len(decoder_channels):
            raise ValueError(
                "Model depth is {}, but you provide `decoder_channels` for {} blocks.".format(
                    n_blocks, len(decoder_channels)
                )
            )

        # remove first skip with same spatial resolution
        encoder_channels = encoder_channels[1:]
        # reverse channels to start from head of encoder
        encoder_channels = encoder_channels[::-1]

        # computing blocks input and output channels
        head_channels = encoder_channels[0]
        self.in_channels = [head_channels] + list(decoder_channels[:-1])
        self.skip_channels = list(encoder_channels[1:]) + [0]
        self.out_channels = decoder_channels  # 256, 128, 64, 32, 16

        self.center = nn.Identity()

        # combine decoder keyword arguments
        kwargs = dict(use_batchnorm=use_batchnorm, attention_type=attention_type)

        blocks = {}
        for layer_idx in range(len(self.in_channels) - 1):  # 0 1 2 3
            for depth_idx in range(layer_idx + 1):
                if depth_idx == 0:
                    in_ch = self.in_channels[layer_idx]
                    skip_ch = self.skip_channels[layer_idx] * (layer_idx + 1)
                    out_ch = self.out_channels[layer_idx]
                else:
                    out_ch = self.skip_channels[layer_idx]
                    skip_ch = self.skip_channels[layer_idx] * (layer_idx + 1 - depth_idx)
                    in_ch = self.skip_channels[layer_idx - 1]

                if depth_idx == 0 and layer_idx in [0, 1, 2, 3]:
                    blocks[f"x_{depth_idx}_{layer_idx}"] = DecoderBlock(in_ch, skip_ch, out_ch, attention_type="dss")
                else:
                    blocks[f"x_{depth_idx}_{layer_idx}"] = DecoderBlock(in_ch, skip_ch, out_ch, **kwargs)

                # blocks[f"x_{depth_idx}_{layer_idx}"] = DecoderBlock(in_ch, skip_ch, out_ch, **kwargs)
        blocks[f"x_{0}_{len(self.in_channels) - 1}"] = DecoderBlock(
            self.in_channels[-1], 0, self.out_channels[-1], **kwargs
        )
        self.blocks = nn.ModuleDict(blocks)
        self.depth = len(self.in_channels) - 1  # 4

    def forward(self, *features):
        # features = 3, 24, 48, 64, 160, 256
        # remove first skip with same spatial resolution
        features = features[1:]  # 24, 48, 64, 160, 256
        # reverse channels to start from head of encode
        features = features[::-1]  # 256, 160, 64, 48, 24
        # start building dense connections
        dense_x = {}
        for layer_idx in range(len(self.in_channels) - 1):
            for depth_idx in range(self.depth - layer_idx):
                if layer_idx == 0:
                    output = self.blocks[f"x_{depth_idx}_{depth_idx}"](features[depth_idx], features[depth_idx + 1])
                    dense_x[f"x_{depth_idx}_{depth_idx}"] = output
                else:
                    dense_l_i = depth_idx + layer_idx
                    cat_features = [dense_x[f"x_{idx}_{dense_l_i}"] for idx in range(depth_idx + 1, dense_l_i + 1)]
                    cat_features = torch.cat(cat_features + [features[dense_l_i + 1]], dim=1)
                    dense_x[f"x_{depth_idx}_{dense_l_i}"] = self.blocks[f"x_{depth_idx}_{dense_l_i}"](
                        dense_x[f"x_{depth_idx}_{dense_l_i - 1}"], cat_features
                    )
        dense_x[f"x_{0}_{self.depth}"] = self.blocks[f"x_{0}_{self.depth}"](dense_x[f"x_{0}_{self.depth - 1}"])
        return dense_x[f"x_{0}_{self.depth}"]
