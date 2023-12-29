# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
from toolbox.models.backbone.mobilenetv2 import mobilenet_v2
import collections


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, dilation=1, groups=1):
        padding = ((kernel_size - 1) * dilation + 1) // 2
        # padding = (kernel_size - 1) // 2
        super(ConvBNReLU, self).__init__(collections.OrderedDict([
            ('conv', nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, dilation=dilation, groups=groups,
                               bias=False)),
            ('bn', nn.BatchNorm2d(out_planes)),
            ('relu', nn.ReLU6(inplace=True))]))


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()

        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = max_out
        att = self.sigmoid(out)
        out = torch.mul(x, att)
        return out


class Fusion(nn.Module):
    def __init__(self, in_dim, kernel_size=7):
        super(Fusion, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(1, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.tp = nn.Conv2d(in_dim, 64, kernel_size=1)
        self.ca = ChannelAttention(64)

    def forward(self, x1, x2):
        max_out, _ = torch.max(x2, dim=1, keepdim=True)
        x2 = max_out
        x2 = self.conv1(x2)
        att2 = self.sigmoid(x2+x1)
        out = torch.mul(x1, att2) + x2
        tp = self.tp(out)
        fuseout = self.ca(tp)

        return fuseout

class Interv2(nn.Module):
    def __init__(self, in_channels):
        super(Interv2, self).__init__()
        self.N = in_channels // 4
        self.S = in_channels // 2

        self.theta = nn.Conv2d(in_channels, self.N, 1, 1, 0, bias=False)
        self.phi = nn.Conv2d(in_channels, self.S, 1, 1, 0, bias=False)

        self.relu = nn.ReLU()

        self.node_conv = nn.Conv1d(self.N, self.N, 1, 1, 0, bias=False)
        self.channel_conv = nn.Conv1d(self.S, self.S, 1, 1, 0, bias=False)

        self.conv_2 = nn.Conv2d(self.S, in_channels, 1, 1, 0, bias=False)

    def forward(self, x):
        batch, C, H, W = x.size()
        L = H * W

        B = self.theta(x).view(-1, self.N, L) #N*L

        phi = self.phi(x).view(-1, self.S, L)  #S*L
        phi = torch.transpose(phi, 1, 2)       #L*S

        V = torch.bmm(B, phi) / L  #  #N*S

        AV = self.relu(self.node_conv(V))  #N*S
        IV = V+AV
        IAVW = self.relu(self.channel_conv(torch.transpose(IV, 1, 2)))  #S*N

        y = torch.bmm(torch.transpose(B, 1, 2), torch.transpose(IAVW, 1, 2))
        y = y.view(-1, self.S, H, W)
        y = self.conv_2(y)

        return x + y


class CRRM(nn.Module):
    def __init__(self, in_channels):
        super(CRRM, self).__init__()
        self.C = 128
        self.omg = nn.Conv2d(in_channels, self.C, 1, 1, 0, bias=False)  #C*H*W
        self.theta = nn.Conv2d(in_channels, self.C, 1, 1, 0, bias=False)  #C*H*W
        self.ori = nn.Conv2d(in_channels, self.C, 1, 1, 0, bias=False)   #C*H*W

        self.relu = nn.ReLU()

        self.node_conv = nn.Conv1d(self.C, self.C, 1, 1, 0, bias=False)
        self.channel_conv = nn.Conv1d(self.C, self.C, 1, 1, 0, bias=False)


    def forward(self, x):
        batch, C, H, W = x.size()
        L = H * W
        o = self.omg(x)
        omg = o.view(-1, self.C, L) #C*L
        theta = self.theta(x).view(-1, self.C, L) #C*L
        ori = self.ori(x).view(-1, self.C, L)  #C*L

        B = torch.transpose(ori, 1, 2)       #L*C

        V = torch.bmm(theta, B)  #
        A = torch.nn.functional.softmax(V, dim=-1) #CC
        AV = self.node_conv(A)  #C*C
        IV = A+AV
        IAV = self.relu(self.channel_conv(torch.transpose(IV, 1, 2)))  #C*C

        y = torch.bmm(IAV, omg)
        y = y.view(-1, self.C, H, W)

        return o + y

class ARM_I(nn.Module):
    def __init__(self, low_channel, high_channel, middle):
        super(ARM_I, self).__init__()

        self.dilation_conv1 = ConvBNReLU(low_channel+high_channel+middle, low_channel, dilation=1)
        self.dilation_conv2 = ConvBNReLU(low_channel+high_channel+middle, low_channel, dilation=2)
        self.dilation_conv3 = ConvBNReLU(low_channel+high_channel+middle, low_channel, dilation=4)
        self.conv1 = ConvBNReLU(low_channel*3, low_channel, kernel_size=1)
        self.upx2 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.high_out = ConvBNReLU(middle, low_channel, kernel_size=1)

    def forward(self, low, previous_arm, high):

        if previous_arm.size()[2] != low.size()[2]:
            previous_arm = self.upx2(previous_arm)
            high = self.upx2(high)
        x = torch.cat((low, previous_arm, high), dim=1)
        x1 = self.dilation_conv1(x)
        x2 = self.dilation_conv2(x)
        x3 = self.dilation_conv3(x)
        low = self.conv1(torch.cat((x1, x2, x3), dim=1))
        high = self.high_out(high)
        out1 = low*high
        out2 = out1+high
        return out1, out2


class ARM_II(nn.Module):
    def __init__(self, low_channel, high_channel):
        super(ARM_II, self).__init__()

        self.dilation_conv1 = ConvBNReLU(low_channel+2*high_channel, low_channel, dilation=1)
        self.dilation_conv2 = ConvBNReLU(low_channel+2*high_channel, low_channel, dilation=2)
        self.dilation_conv3 = ConvBNReLU(low_channel+2*high_channel, low_channel, dilation=4)
        self.conv1 = ConvBNReLU(low_channel*3, low_channel, kernel_size=1)
        self.upx2 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.high_out = ConvBNReLU(high_channel, low_channel, kernel_size=1)

    def forward(self, low, previous_arm, high):

        if previous_arm.size()[2] != low.size()[2]:
            previous_arm = self.upx2(previous_arm)
            high = self.upx2(high)
        x = torch.cat((low, previous_arm, high), dim=1)
        x1 = self.dilation_conv1(x)
        x2 = self.dilation_conv2(x)
        x3 = self.dilation_conv3(x)
        low = self.conv1(torch.cat((x1, x2, x3), dim=1))
        high = self.high_out(high)
        out1 = low*high
        out2 = out1+high
        return out1, out2


class mobilenetGloRe3_CRRM_dule_arm_bou_att(nn.Module):

    def __init__(self, n_classes):
        super(mobilenetGloRe3_CRRM_dule_arm_bou_att, self).__init__()

        self.mobile_rgb = mobilenet_v2(pretrained=True)
        self.mobile_dep = mobilenet_v2(pretrained=True)

        self.Glore5 = Interv2(160*2)
        self.Glore4 = Interv2(96*2)
        self.Glore3 = Interv2(64*2)
        self.lfe2 = Fusion(32)
        self.lfe1 = Fusion(24)

        self.CRRM = CRRM((160+96+64)*2)

        self.arm1 = ARM_II(64, 64)
        self.arm2 = ARM_II(32*2, 64*2)
        self.arm3 = ARM_II(64*2, 96*2)
        self.arm4 = ARM_I(96*2, 160*2, 128)
        self.upx2 = nn.UpsamplingBilinear2d(scale_factor=2)

        self.classifier_Z1 = nn.Sequential(
            ConvBNReLU(64, 64, kernel_size=3, stride=1),
            self.upx2,
            nn.Conv2d(64, n_classes, kernel_size=1),
            self.upx2,

        )




    def forward(self, rgb, dep):
        dep1 = self.mobile_dep.features[0:4](dep)
        dep2 = self.mobile_dep.features[4:7](dep1)
        dep3 = self.mobile_dep.features[7:11](dep2)
        dep4 = self.mobile_dep.features[11:14](dep3)
        dep5 = self.mobile_dep.features[14:17](dep4)

        rgb1 = self.mobile_rgb.features[0:4](rgb)
        rgb2 = self.mobile_rgb.features[4:7](rgb1)
        rgb3 = self.mobile_rgb.features[7:11](rgb2)
        rgb4 = self.mobile_rgb.features[11:14](rgb3)
        rgb5 = self.mobile_rgb.features[14:17](rgb4)

        fuse5 = self.Glore5(torch.cat([rgb5, dep5], dim=1))
        fuse4 = self.Glore4(torch.cat([rgb4, dep4], dim=1))
        fuse3 = self.Glore3(torch.cat([rgb3, dep3], dim=1))
        fuse_mid = self.CRRM(torch.cat([fuse5, fuse4, fuse3], dim=1))

        l2 = self.lfe2(rgb2, dep2)
        l1 = self.lfe1(rgb1, dep1)


        out4_lv, out4_zi = self.arm4(fuse4, fuse5, fuse_mid)
        out3_lv, out3_zi = self.arm3(fuse3, out4_lv, out4_zi)
        out2_lv, out2_zi = self.arm2(l2, out3_lv, out3_zi)
        out1_lv, out1_zi = self.arm1(l1, out2_lv, out2_zi)

        out = self.classifier_Z1(out1_zi)
        return out
