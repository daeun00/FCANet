from pickletools import uint8

import torch
import torch.nn as nn
import torch.nn.functional as F
import SWT

###tensor
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()
###


class UL(nn.Module):
    def __init__(self, inchannel):
        super(UL, self).__init__()

        # Encoder
        self.enc1_1 = ConvBlock(input_size=inchannel, output_size=64, kernel_size=3, stride=1, padding='same')
        self.enc1_2 = ConvBlock(input_size=64, output_size=64, kernel_size=3, stride=1, padding='same')
        self.pool1 = nn.MaxPool2d(kernel_size=2)

        self.enc2_1 = ConvBlock(input_size=64, output_size=128, kernel_size=3, stride=1, padding='same')
        self.enc2_2 = ConvBlock(input_size=128, output_size=128, kernel_size=3, stride=1, padding='same')
        self.pool2 = nn.MaxPool2d(kernel_size=2)

        self.enc3_1 = ConvBlock(input_size=128, output_size=256, kernel_size=3, stride=1, padding='same')
        self.enc3_2 = ConvBlock(input_size=256, output_size=256, kernel_size=3, stride=1, padding='same')
        self.max3 = nn.AdaptiveMaxPool2d((1, 1))
        self.convmax3 = ConvBlock(input_size=256, output_size=256, kernel_size=1, stride=1, padding='same')

        self.dec4_1 = ConvBlock(input_size=256, output_size=256, kernel_size=3, stride=1, padding='same')
        self.unpool4 = upsample(input_size=256, output_size=128, kernel_size=2, stride=2, padding=0)

        self.conv5 = ConvBlock(input_size=256, output_size=256, kernel_size=3, stride=1, padding='same')
        self.enc5_1 = ConvBlock(input_size=256, output_size=128, kernel_size=3, stride=1, padding='same')
        self.enc5_2 = ConvBlock(input_size=128, output_size=128, kernel_size=3, stride=1, padding='same')
        self.pool5 = nn.MaxPool2d(kernel_size=2)

        self.enc6_1 = ConvBlock(input_size=128, output_size=256, kernel_size=3, stride=1, padding='same')
        self.enc6_2 = ConvBlock(input_size=256, output_size=256, kernel_size=3, stride=1, padding='same')
        self.max6 = nn.AdaptiveMaxPool2d((1, 1))
        self.convmax6 = ConvBlock(input_size=256, output_size=256, kernel_size=1, stride=1, padding='same')

        self.dec7_1 = ConvBlock(input_size=256, output_size=256, kernel_size=3, stride=1, padding='same')
        self.unpool7 = upsample(input_size=256, output_size=128, kernel_size=2, stride=2, padding=0)

        self.dec8_1 = ConvBlock(input_size=256, output_size=128, kernel_size=3, stride=1, padding='same')
        self.dec8_2 = ConvBlock(input_size=128, output_size=128, kernel_size=3, stride=1, padding='same')
        self.unpool8 = upsample(input_size=128, output_size=64, kernel_size=2, stride=2, padding=0)

        self.dec9_1 = ConvBlock(input_size=128, output_size=64, kernel_size=3, stride=1, padding='same')
        self.dec9_2 = ConvBlock(input_size=64, output_size=64, kernel_size=3, stride=1, padding='same')

        self.max9 = nn.AdaptiveMaxPool2d((1, 1))
        self.convmax9 = ConvBlock(input_size=64, output_size=64, kernel_size=1, stride=1, padding='same')


        self.no = NoiseBlock(input_size=64, output_size=64, kernel_size=3)
        self.out = nn.Conv2d(in_channels=64, out_channels=3, kernel_size=1)  # 수정

    def forward(self, x):
        # ULBP

        feature1 = self.enc1_1(x)  # 3 64
        feature1 = self.enc1_2(feature1) # 64 64
        pool_feature1 = self.pool1(feature1) # 1/2

        feature2 = self.enc2_1(pool_feature1)  # 64 128
        feature2 = self.enc2_2(feature2) # 128 128
        pool_feature2 = self.pool2(feature2) # 1/4

        feature3 = self.enc3_1(pool_feature2)  # 128 256
        feature3 = self.enc3_2(feature3) # 256 256
        max_feature3 = self.max3(feature3)
        max_feature3 = self.convmax3(max_feature3)

        feature4 = feature3 + max_feature3
        feature4 = self.dec4_1(feature4) # 256 256
        feature4 = self.unpool4(feature4)  # 256 128

        feature5= torch.cat((feature2, feature4), 1)  # 256
        # feature5 = self.conv5(feature5) # 256 256
        feature5 = self.enc5_1(feature5) # 256 128
        feature5 = self.enc5_2(feature5) # 128 128
        pool_feature5 = self.pool5(feature5)

        feature6 = self.enc6_1(pool_feature5)  # 128 256
        feature6 = self.enc6_2(feature6) # 256 256
        max_feature6 = self.max6(feature6)
        max_feature6 = self.convmax6(max_feature6)

        feature7 = feature6 + max_feature6
        feature7 = self.dec7_1(feature7) # 256 256
        feature7 = self.unpool7(feature7)  # 256 128

        feature8 = torch.cat((feature5, feature7), 1)  # 256
        feature8 = self.dec8_1(feature8) # 256 128
        feature8 = self.dec8_2(feature8) # 128 128
        feature8 = self.unpool8(feature8)  # 128 64

        feature9 = torch.cat((feature1, feature8), 1)  # 128
        feature9 = self.dec9_1(feature9)  # 128 64
        feature9 = self.dec9_2(feature9)  # 64 64

        max_feature9 = self.max9(feature9)
        max_feature9 = self.convmax9(max_feature9)

        feature10 = feature9 + max_feature9

        NO_feature10 = self.no(feature10) # 64 64
        ULout = self.out(NO_feature10) # 64 3

        return ULout




class FCANet(nn.Module):
    def __init__(self, input_dim=3):
        super(FCANet, self).__init__()

        self.swt = SWT.SWTForward(J=1, wave='haar', mode='zero')
        self.iswt = SWT.SWTInverse(wave='haar', mode='zero')

        self.convl = torch.nn.Conv2d(3, 3, kernel_size=3, stride=1, padding='same')
        self.convh = torch.nn.Conv2d(9, 9, kernel_size=3, stride=1, padding='same')
        self.convs = torch.nn.Conv2d(1, 3, kernel_size=3, stride=1, padding='same')

        ### SF block_1
        self.slConv6to3 = torch.nn.Conv2d(6, 3, kernel_size=3, stride=1, padding='same')
        ######

        self.ul1 = UL(inchannel = 3)
        self.ul2 = UL(inchannel=3)

        self.max = nn.AdaptiveMaxPool2d((1, 1))
        self.avg = nn.AdaptiveAvgPool2d((1, 1))


        ### MSS block
        self.hconv3 = torch.nn.Conv2d(in_channels=9, out_channels=9, kernel_size=3, stride=1, padding='same')
        self.hconv5 = torch.nn.Conv2d(in_channels=9, out_channels=9, kernel_size=5, stride=1, padding='same')
        self.hconv7 = torch.nn.Conv2d(in_channels=9, out_channels=9, kernel_size=7, stride=1, padding='same')
        self.h_finalConv = torch.nn.Conv2d(in_channels=27, out_channels=9, kernel_size=3, stride=1, padding='same')
        ######

        ### FCA block
        self.lmaxConv = torch.nn.Conv2d(in_channels=3, out_channels=3, kernel_size=1, stride=1, padding='same')
        self.hAvgConv = torch.nn.Conv2d(in_channels=9, out_channels=3, kernel_size=1, stride=1, padding='same')
        self.conv18to9 = torch.nn.Conv2d(in_channels=18, out_channels=9, kernel_size=3, stride=1, padding='same')
        ######

        ### SF block_2
        self.orgConv = torch.nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, stride=1, padding='same')
        self.soConv6to3 = torch.nn.Conv2d(6, 3, kernel_size=3, stride=1, padding='same')
        ######
        self.finalout = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=1, stride=1)


        for m in self.modules():
            classname = m.__class__.__name__
            if classname.find('Conv2d') != -1:
                torch.nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif classname.find('ConvTranspose2d') != -1:
                torch.nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()

        ##########################################



    def get_list(self, out):
        SWT_list = [out]

        return SWT_list

    def waveL(self, t1, t2, t3):
        #Separate

        t1_0, t1_1, t1_2, t1_3 = torch.chunk(t1[0], 4, dim=1)
        t2_0, t2_1, t2_2, t2_3 = torch.chunk(t2[0], 4, dim=1)
        t3_0, t3_1, t3_2, t3_3 = torch.chunk(t3[0], 4, dim=1)
        ll = torch.cat((t1_0, t2_0, t3_0), dim=1)

        return ll

    def waveH(self, t1, t2, t3):
        #Separate

        t1_0, t1_1, t1_2, t1_3 = torch.chunk(t1[0], 4, dim=1)
        t2_0, t2_1, t2_2, t2_3 = torch.chunk(t2[0], 4, dim=1)
        t3_0, t3_1, t3_2, t3_3 = torch.chunk(t3[0], 4, dim=1)
        lh = torch.cat((t1_1, t2_1, t3_1), dim=1)
        hl = torch.cat((t1_2, t2_2, t3_2), dim=1)
        hh = torch.cat((t1_3, t2_3, t3_3), dim=1)
        total = torch.cat((lh, hl, hh), dim=1)

        return total

    def rgb2hsv(self, rgb: torch.Tensor) -> torch.Tensor:
        cmax, cmax_idx = torch.max(rgb, dim=1, keepdim=True)
        cmin = torch.min(rgb, dim=1, keepdim=True)[0]
        delta = cmax - cmin
        hsv_s = torch.where(cmax == 0, torch.tensor(0.).type_as(rgb), delta / cmax)

        return hsv_s

    def forward(self, x_ori, tar=None):
        # data gate

        r, g, b = torch.split(x_ori, 1, dim=1)
        red = self.swt(r)
        green = self.swt(g)
        blue = self.swt(b)

        x_low = self.waveL(red, green, blue)
        x_high = self.waveH(red, green, blue)

        x_sat = self.rgb2hsv(x_ori)
        x_sat = self.convs(x_sat)  # 3 #Fs

        l_in = self.convl(x_low)  # Fl
        h_in = self.convh(x_high)  # Fh

        ##### Saturation Fusion Block (Fs & Fl) #####
        sat_low = torch.cat((l_in, x_sat), dim=1)  # 6
        sat_low_max = self.max(sat_low)
        sat_low_weighted6 = sat_low * sat_low_max
        sat_low_weighted3 = self.slConv6to3(sat_low_weighted6)
        sat_low_out = l_in * sat_low_weighted3  # Fl-SF
        ######

        ##### U-Shaped Lightening (Fl-SF) #####
        final_l = self.ul1(sat_low_out)  # Fl-L
        ######

        ### Multi Scale Sharpening Block ###
        h_c3 = self.hconv3(h_in)
        h_c5 = self.hconv5(h_in)
        h_c7 = self.hconv7(h_in)

        h_max3 = self.max(h_c3)
        h_max5 = self.max(h_c5)
        h_max7 = self.max(h_c7)

        h_w3 = h_c3 * h_max3
        h_w5 = h_c5 * h_max5
        h_w7 = h_c7 * h_max7

        h_out27 = torch.cat((h_w3, h_w5, h_w7), dim=1)

        h_out9 = self.h_finalConv(h_out27)

        h_out_max = self.max(h_out9)
        final_h = h_out_max * h_in  # Fh-SS
        ######

        ### Frequency Cross Attention Block ###

        h_lh = final_h[:, 0:3]
        # print(h_lh.shape)
        h_hl = final_h[:, 3:6]
        h_hh = final_h[:, 6:9]

        l_max = self.max(final_l)
        l_max_out = self.lmaxConv(l_max)

        h_lh_lmax = h_lh * l_max_out
        h_hl_lmax = h_hl * l_max_out
        h_hh_lmax = h_hh * l_max_out

        h_final_out = torch.cat((h_lh_lmax, h_hl_lmax, h_hh_lmax), dim=1)  # Fh-L

        h_lh_avg = self.avg(h_lh)
        h_hl_avg = self.avg(h_hl)
        h_hh_avg = self.avg(h_hh)

        h_avg = torch.cat((h_lh_avg, h_hl_avg, h_hh_avg), dim=1)
        h_avg_out = self.hAvgConv(h_avg)

        l_hAvg = final_l * h_avg_out  # Fl-H
        ######

        ### Residual ###
        l_added = x_low + l_hAvg  # Fl-E

        h_cat = torch.cat((x_high, h_final_out), dim=1)
        h_cat_out = self.conv18to9(h_cat)  # Fh-E
        ######

        ### ISWT ###
        swt_out = torch.cat((l_added, h_cat_out), dim=1)
        swt_out = self.get_list(swt_out)
        swt_out = self.iswt(swt_out)
        ######

        ###### Refinement ######

        ### Saturation Fusion Block (Fs & FI) ###
        org_feature = self.orgConv(x_ori)  # FI

        sat_org = torch.cat((org_feature, x_sat), dim=1)  # 6
        sat_org_max = self.max(sat_org)
        sat_org_weighted6 = sat_org_max * sat_org  # 6
        sat_org_weighted3 = self.soConv6to3(sat_org_weighted6)  # 3
        sat_org_out = org_feature * sat_org_weighted3  # FI-SF
        ######

        ### U-Shaped Lightening (FI-SF) ###
        org_ul_out = self.ul2(sat_org_out)  # FI-L
        ######

        weighted_res = org_ul_out * swt_out

        ############

        pred = self.finalout(weighted_res)

        return pred


# LBP


class LightenBlock(torch.nn.Module):
    def __init__(self, input_size, output_size, kernel_size, stride, padding, bias=True):
        super(LightenBlock, self).__init__()
        self.conv_Encoder = ASB(input_size, output_size, kernel_size=3)
        self.conv_Offset = ASB(input_size, output_size, kernel_size=3)
        self.conv_Decoder = ASB(input_size, output_size, kernel_size=3)

    def forward(self, x):
        offset = self.conv_Offset(x)
        code_lighten = x + offset
        out = self.conv_Decoder(code_lighten)
        return out


class DarkenBlock(torch.nn.Module):
    def __init__(self, input_size, output_size, kernel_size, stride, padding, bias=True):
        super(DarkenBlock, self).__init__()
        self.conv_Encoder = ASB(input_size, output_size, kernel_size=3)
        self.conv_Offset = ASB(input_size, output_size, kernel_size=3)
        self.conv_Decoder = ASB(input_size, output_size, kernel_size=3)

    def forward(self, x):
        offset = self.conv_Offset(x)
        code_lighten = x - offset
        out = self.conv_Decoder(code_lighten)
        return out


class FusionLayer(nn.Module):
    def __init__(self, inchannel, outchannel):
        super(FusionLayer, self).__init__()

        self.MASE = MASEblock(inchannel)
        self.PASB = ASB(inchannel, outchannel, kernel_size=3)

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.MASE(x).view(b, c, 1, 1)
        y = x * y.expand_as(x)
        y = y + x
        y = self.PASB(y)
        return y


class LBP(torch.nn.Module):
    def __init__(self, input_size, output_size, kernel_size, stride, padding):
        super(LBP, self).__init__()
        self.fusion = FusionLayer(input_size, output_size)
        self.conv1 = LightenBlock(output_size, output_size, kernel_size, stride, padding, bias=True)
        self.conv2 = DarkenBlock(output_size, output_size, kernel_size, stride, padding, bias=True)
        self.conv3 = LightenBlock(output_size, output_size, kernel_size, stride, padding, bias=True)
        self.local_weight1_1 = weASB(input_size, output_size, kernel_size=1)
        self.local_weight2_1 = weASB(input_size, output_size, kernel_size=1)

    def forward(self, x):
        x = self.fusion(x)
        hr = self.conv1(x)
        lr = self.conv2(hr)
        residue = self.local_weight1_1(x) - lr
        h_residue = self.conv3(residue)
        hr_weight = self.local_weight2_1(hr)
        return hr_weight + h_residue


# Multi Scale / Noise Block


class MSBlock(torch.nn.Module):
    def __init__(self, input_size, output_size, kernel_size):
        super(MSBlock, self).__init__()
        self.ds1 = nn.Conv2d(input_size, output_size, kernel_size, stride=2, padding=1)
        self.ds2 = nn.Conv2d(input_size, output_size, kernel_size, stride=2, padding=1)
        self.conv1 = nn.Conv2d(input_size, output_size, kernel_size, stride=1, padding='same', bias=True)
        self.conv2 = nn.Conv2d(input_size, output_size, kernel_size, stride=1, padding='same', bias=True)
        self.conv3 = nn.Conv2d(input_size, output_size, kernel_size, stride=1, padding='same', bias=True)
        self.conv4 = nn.Conv2d(input_size * 4, output_size, kernel_size, stride=1, padding='same', bias=True)
        self.bn1 = nn.BatchNorm2d(input_size)
        self.bn2 = nn.BatchNorm2d(input_size)
        self.bn3 = nn.BatchNorm2d(input_size)
        self.act1 = nn.ReLU()

    def forward(self, x):
        out1 = self.conv1(x)
        out1 = self.bn1(out1)
        out1 = self.act1(out1)  # CONV BLOCK

        out2 = self.ds1(x)
        out2 = self.conv2(out2)
        out2 = self.bn2(out2)
        out2 = self.act1(out2)
        out2 = F.interpolate(out2, size=(out1.size()[2], out1.size()[3]))

        out3 = self.ds1(x)
        out3 = self.ds2(out3)
        out3 = self.conv3(out3)
        out3 = self.bn3(out3)
        out3 = self.act1(out3)
        out3 = F.interpolate(out3, size=(out2.size()[2], out2.size()[3]))
        out3 = F.interpolate(out3, size=(out1.size()[2], out1.size()[3]))
        out = torch.cat([x, out1, out2, out3], dim=1)
        out = self.conv4(out)

        return out


class NoiseBlock(torch.nn.Module):
    def __init__(self, input_size, output_size, kernel_size):
        super(NoiseBlock, self).__init__()
        self.conv1 = nn.Conv2d(input_size, output_size, kernel_size, stride=1, padding='same', bias=True)
        self.conv2 = nn.Conv2d(input_size, output_size, kernel_size, stride=1, padding='same', bias=True)
        self.conv3 = nn.Conv2d(input_size, output_size, kernel_size, stride=1, padding='same', bias=True)
        self.act1 = nn.ReLU()

    def forward(self, x):
        out1 = self.conv1(x)
        out1 = self.act1(out1)
        out2 = self.conv2(out1)
        out2 = self.act1(out2)
        out3 = self.conv3(out2)

        out = torch.tanh(out3)

        return out


# Base modules


class ConvBlock(torch.nn.Module):
    def __init__(self, input_size, output_size, kernel_size, stride, padding, bias=True, isuseBN=False):
        super(ConvBlock, self).__init__()
        self.isuseBN = isuseBN
        self.conv = torch.nn.Conv2d(input_size, output_size, kernel_size, stride, padding, bias=bias)
        if self.isuseBN:
            self.bn = nn.BatchNorm2d(output_size)
        self.act = torch.nn.ReLU()

    def forward(self, x):
        out = self.conv(x)
        if self.isuseBN:
            out = self.bn(out)
        out = self.act(out)
        return out


class upsample(torch.nn.Module):
    def __init__(self, input_size, output_size, kernel_size=3, stride=1, padding='same'):
        super(upsample, self).__init__()
        self.up = nn.ConvTranspose2d(input_size, output_size, kernel_size=2, stride=2, padding=0, bias=True)

    def forward(self, x):  # skip 빈거 넣어줄때
        x = self.up(x)

        return x


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


# Stretching Block(ASB/ANB)


class MASEblock(nn.Module):
    def __init__(self, in_channels, r=16):
        super().__init__()
        self.squeeze = nn.AdaptiveMaxPool2d((1, 1))
        self.excitation = nn.Sequential(
            nn.Linear(in_channels, in_channels // r),
            nn.ReLU(),
            nn.Linear(in_channels // r, in_channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.squeeze(x)
        x = x.view(x.size(0), -1)
        x = self.excitation(x)
        x = x.view(x.size(0), x.size(1), 1, 1)

        return x


class MISEblock(nn.Module):
    def __init__(self, in_channels, r=16):
        super().__init__()
        self.squeeze = nn.AdaptiveMaxPool2d((1, 1))
        self.excitation = nn.Sequential(
            nn.Linear(in_channels, in_channels // r),
            nn.ReLU(),
            nn.Linear(in_channels // r, in_channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = -self.squeeze(-x)
        x = x.view(x.size(0), -1)
        x = self.excitation(x)
        x = x.view(x.size(0), x.size(1), 1, 1)

        return x


class ANB(nn.Module):
    def __init__(self, in_channels):
        super().__init__()

        self.maseblock = MASEblock(in_channels)
        self.miseblock = MISEblock(in_channels)

    def forward(self, x):
        im_h = self.maseblock(x)
        im_l = self.miseblock(x)

        me = torch.tensor(0.00001, dtype=torch.float32).cuda()

        x = (x - im_l) / torch.maximum(im_h - im_l, me)
        x = torch.clip(x, 0.0, 1.0)

        return x


class ASB(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=1, padding='same'),
            nn.BatchNorm2d(out_channels),
            nn.PReLU(),
        )

    def forward(self, x):
        x = self.conv(x)

        return x


class weASB(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=1),
            nn.BatchNorm2d(out_channels),
            nn.PReLU(),
        )

    def forward(self, x):
        x = self.conv(x)

        return x
