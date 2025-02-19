import torch.nn as nn
import torch
from models.EfficientnetV2 import efficientnetv2_s
from models.Unetplusplus_DSS import UnetPlusPlusDecoder as UnetPlusPlusDecoderDSS

class SegmentationHead(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, upsampling=1):
        conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        upsampling = nn.UpsamplingBilinear2d(scale_factor=upsampling) if upsampling > 1 else nn.Identity()
        super().__init__(conv2d, upsampling)


class AFFE(nn.Module):  # Scale Enhancement Module
    def __init__(self, c_in, c_out, rate=4):
        super(AFFE, self).__init__()
        self.rate = rate

        self.m_channel = c_in // 4

        self.conv = nn.Conv2d(c_in, self.m_channel, 3, stride=1, padding=1)
        self.relu = nn.ReLU(inplace=True)

        self.conv1_1 = nn.Conv2d(self.m_channel, 1, 1, stride=1, padding=0)
        self.conv1_2 = nn.Conv2d(1, 1, 3, stride=1, padding=1)
        self.sigmoid_1 = nn.Sigmoid()

        dilation = self.rate * 1 if self.rate >= 1 else 1
        self.conv2 = nn.Conv2d(self.m_channel, self.m_channel, 3, stride=1, dilation=dilation, padding=dilation)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2_1 = nn.Conv2d(self.m_channel, 1, 1, stride=1, padding=0)
        self.conv2_2 = nn.Conv2d(1, 1, 3, stride=1, padding=1)
        self.sigmoid_2 = nn.Sigmoid()

        dilation = self.rate * 2 if self.rate >= 1 else 1
        self.conv3 = nn.Conv2d(self.m_channel, self.m_channel, 3, stride=1, dilation=dilation, padding=dilation)
        self.relu3 = nn.ReLU(inplace=True)
        self.conv3_1 = nn.Conv2d(self.m_channel, 1, 1, stride=1, padding=0)
        self.conv3_2 = nn.Conv2d(1, 1, 3, stride=1, padding=1)
        self.sigmoid_3 = nn.Sigmoid()

        dilation = self.rate * 3 if self.rate >= 1 else 1
        self.conv4 = nn.Conv2d(self.m_channel, self.m_channel, 3, stride=1, dilation=dilation, padding=dilation)
        self.relu4 = nn.ReLU(inplace=True)
        self.conv4_1 = nn.Conv2d(self.m_channel, 1, 1, stride=1, padding=0)
        self.conv4_2 = nn.Conv2d(1, 1, 3, stride=1, padding=1)
        self.sigmoid_4 = nn.Sigmoid()

        self.conv5 = nn.Conv2d(self.m_channel * 4, c_out, 3, stride=1, padding=1)
        self.relu5 = nn.ReLU(inplace=True)

        self._initialize_weights()

    def forward(self, x):
        o = self.relu(self.conv(x))  # c=32
        o1 = o  # c=32
        o2 = self.relu2(self.conv2(o))  # c=32
        o3 = self.relu3(self.conv3(o))  # c=32
        o4 = self.relu4(self.conv4(o))  # c=32

        o1_w = self.sigmoid_1(self.conv1_2(self.conv1_1(o1)))
        o2_w = self.sigmoid_2(self.conv2_2(self.conv2_1(o2)))
        o3_w = self.sigmoid_3(self.conv3_2(self.conv3_1(o3)))
        o4_w = self.sigmoid_4(self.conv4_2(self.conv4_1(o4)))

        o1_w = o1_w / (o1_w + o2_w + o3_w + o4_w)
        o2_w = o2_w / (o1_w + o2_w + o3_w + o4_w)
        o3_w = o3_w / (o1_w + o2_w + o3_w + o4_w)
        o4_w = o4_w / (o1_w + o2_w + o3_w + o4_w)

        o1 = o1 * o1_w
        o2 = o2 * o2_w
        o3 = o3 * o3_w
        o4 = o4 * o4_w

        out = torch.cat([o1, o2, o3, o4], dim=1)
        out = out + x
        out = self.relu5(self.conv5(out))
        return out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.01)
                if m.bias is not None: 
                    m.bias.data.zero_()


class Features_AFFE(nn.Module):
    def __init__(self, in_channels):
        super(Features_AFFE, self).__init__()
        self.sem1 = AFFE(c_in=in_channels[0], c_out=in_channels[0])
        self.sem2 = AFFE(c_in=in_channels[1], c_out=in_channels[1])
        self.sem3 = AFFE(c_in=in_channels[2], c_out=in_channels[2])
        self.sem4 = AFFE(c_in=in_channels[3], c_out=in_channels[3])
        self.sem5 = AFFE(c_in=in_channels[4], c_out=in_channels[4])

    def forward(self, x):
        features = []
        x0, x1, x2, x3, x4, x5 = x
        features.append(x0)

        x1_sem = self.sem1(x1)
        features.append(x1_sem)

        x2_sem = self.sem2(x2)
        features.append(x2_sem)

        x3_sem = self.sem3(x3)
        features.append(x3_sem)

        x4_sem = self.sem4(x4)
        features.append(x4_sem)

        x5_sem = self.sem5(x5)
        features.append(x5_sem)
        return features


class DURN(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder_depth = 5
        self.encoder_channels = [3, 24, 48, 64, 160, 256]  # [3, 24, 48, 64, 160, 256]
        self.decoder_use_batchnorm = True,
        self.decoder_channels = (256, 128, 64, 32, 16)
        self.decoder_attention_type = None

        self.encoder = efficientnetv2_s(stage=5)

        self.decoder = UnetPlusPlusDecoderDSS(
            encoder_channels=self.encoder_channels,
            decoder_channels=self.decoder_channels,  # 256, 128, 64, 32, 16
            n_blocks=self.encoder_depth,
            use_batchnorm=True,
            center=False,
            attention_type=None,
        )
        self.segmentation_head = SegmentationHead(
            in_channels=self.decoder_channels[-1],
            out_channels=1,
            kernel_size=3, )

        self.decoder_u = UnetPlusPlusDecoderDSS(
            encoder_channels=self.encoder_channels,
            decoder_channels=self.decoder_channels,  # 256, 128, 64, 32, 16
            n_blocks=self.encoder_depth,  # 5
            use_batchnorm=self.decoder_use_batchnorm,
            center=False,
            attention_type=self.decoder_attention_type,
        )

        self.segmentation_head_u = SegmentationHead(
            in_channels=self.decoder_channels[-1],
            out_channels=1,
            kernel_size=3, )

        self.features_affe = Features_AFFE(in_channels=[24, 48, 64, 160, 256])

    def forward(self, x):
        features = self.encoder(x)
        # features = self.features_dss(features)
        features = self.features_affe(features)
        decoder_output = self.decoder(*features)
        mean = self.segmentation_head(decoder_output)
        decoder_output_u = self.decoder_u(*features)
        std = self.segmentation_head_u(decoder_output_u)
        std = nn.Softplus()(std)
        return mean, std