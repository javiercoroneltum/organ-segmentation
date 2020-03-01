import torch
import torch.nn as nn


class ConvBnRelu2d(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, output_padding=1, dilation=1, stride=1, groups=1, is_bn=True, is_relu=True, is_decoder=False):
        super(ConvBnRelu2d, self).__init__()
        if is_decoder:
            self.transpConv = torch.nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, output_padding=output_padding, stride=stride, dilation=dilation, groups=groups, bias=False)
            self.conv = None
        else:
            self.transpConv = None
            self.conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation, groups=groups, bias=False)
        self.bn = torch.nn.BatchNorm2d(out_channels, eps=1e-4)
        self.relu = torch.nn.ReLU(inplace=True)
        if is_bn is False: self.bn = None
        if is_relu is False: self.relu = None

    def forward(self, x):
        if self.conv is None:
            x = self.transpConv(x)
        elif self.transpConv is None:
            x = self.conv(x)
            
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class StackEncoder(torch.nn.Module):
    def __init__(self, x_channels, y_channels, kernel_size=3, stride=1):
        super(StackEncoder, self).__init__()
        padding = (kernel_size - 1) // 2
        self.encode = torch.nn.Sequential(
            ConvBnRelu2d(x_channels, y_channels, kernel_size=kernel_size, padding=padding, dilation=1, stride=stride, groups=1),
            ConvBnRelu2d(y_channels, y_channels, kernel_size=kernel_size, padding=padding, dilation=1, stride=stride, groups=1),
        )

    def forward(self, x):
        y = self.encode(x)
        y_small = torch.nn.functional.max_pool2d(y, kernel_size=2, stride=2)
        return y, y_small


class StackDecoder(torch.nn.Module):
    def __init__(self, x_big_channels, x_channels, y_channels, kernel_size=3, stride=1):
        super(StackDecoder, self).__init__()
        padding = (kernel_size - 1) // 2

        self.decode = torch.nn.Sequential(
            ConvBnRelu2d(x_big_channels + x_channels, y_channels, kernel_size=kernel_size, padding=padding, dilation=1, stride=stride, groups=1),
            ConvBnRelu2d(y_channels,                  y_channels, kernel_size=kernel_size, padding=padding, dilation=1, stride=stride, groups=1),
            ConvBnRelu2d(y_channels,                  y_channels, kernel_size=kernel_size, padding=padding, dilation=1, stride=stride, groups=1),
        )

    def forward(self, x_big, x):
        N, C, H, W = x_big.size()
        y = torch.nn.functional.interpolate(x, size=(H, W), mode='bilinear', align_corners=True)
        y = torch.cat([y, x_big], 1)
        y = self.decode(y)
        return y


class StackDecoderTranspose(torch.nn.Module):
    def __init__(self, x_big_channels, x_channels, y_channels, kernel_size=3, stride=1, stride_transpose=2, padding=1, padding_transpose=1, output_padding=0):
        super(StackDecoderTranspose, self).__init__()
        padding = (kernel_size - 1) // 2

        self.decode = torch.nn.Sequential(
            ConvBnRelu2d(x_big_channels + x_channels, y_channels, kernel_size=kernel_size, padding=padding, dilation=1, stride=stride, groups=1),
            ConvBnRelu2d(y_channels,                  y_channels, kernel_size=kernel_size, padding=padding, dilation=1, stride=stride, groups=1),
            ConvBnRelu2d(y_channels,                  y_channels, kernel_size=kernel_size, padding=padding_transpose, output_padding=output_padding, dilation=1, stride=stride_transpose, groups=1, is_decoder=True),
        )

    def forward(self, x_big, x):
        y = torch.cat([x, x_big], 1)
        y = self.decode(y)
        return y


class UNet1024(torch.nn.Module):
    def __init__(self):
        super(UNet1024, self).__init__()
        #C, H, W = in_shape
        # assert(C==3)

        # 1024
        self.down1 = StackEncoder(1, 64, kernel_size=3)    # Channels: 1 in,   64 out;  Image size: 300 in, 150 out
        self.down2 = StackEncoder(64, 128, kernel_size=3)   # Channels: 64 in,  128 out;  Image size: 150 in, 75 out
        self.down3 = StackEncoder(128, 256, kernel_size=3)  # Channels: 128 in,  256 out; Image size: 75 in,  38 out
        self.down4 = StackEncoder(256, 512, kernel_size=3) # Channels: 256 in, 512 out; Image size: 38 in,  19 out
        
        self.center = torch.nn.Sequential(
            ConvBnRelu2d(512, 1024, kernel_size=3, padding=1, stride=1), # Channels: 512 in, 1024 out; Image size: 5 in,  5 out
            ConvBnRelu2d(1024, 1024, kernel_size=3, padding=1, stride=1), # Channels: 1024 in, 1024 out; Image size: 5 in,  5 out
        )

        # x_big_channels, x_channels, y_channels
        self.up4 = StackDecoder(512, 1024, 512, kernel_size=3) # Channels: 768+768 = 1536 in, 512 out; Image size: 5 in,   10 out
        self.up3 = StackDecoder(256, 512, 256, kernel_size=3) # Channels: 512+512 = 1024 in, 256 out; Image size: 10 in,  19 out
        self.up2 = StackDecoder(128, 256, 128, kernel_size=3) # Channels: 256+256 = 512  in, 128 out; Image size: 19 in,  38 out
        self.up1 = StackDecoder(64, 128, 64, kernel_size=3)  # Channels: 128+128 = 256  in, 64  out; Image size: 38 in,  75 out
        self.classify = torch.nn.Conv2d(64, 5, kernel_size=1, padding=0, stride=1, bias=True) # Channels: 64 in, 1 out; Image size: 300 in, 300 out

        # Official init from torch repo.
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        out = x  # ;print('x    ',x.size())
        #
        down1, out = self.down1(out)  ##;print('down1',down1.size())  #256
        down2, out = self.down2(out)  # ;print('down2',down2.size())  #128
        down3, out = self.down3(out)  # ;print('down3',down3.size())  #64
        down4, out = self.down4(out)  # ;print('down4',down4.size())  #32
        pass  # ;print('out  ',out.size())

        out = self.center(out)
        out = self.up4(down4, out)
        out = self.up3(down3, out)
        out = self.up2(down2, out)
        out = self.up1(down1, out)
        # 1024

        out = self.classify(out)
        out = torch.squeeze(out, dim=1)
        return out


class UNet768(torch.nn.Module):
    def __init__(self):
        super(UNet768, self).__init__()
        self.down1 = StackEncoder(1, 24, kernel_size=3)    # Channels: 1 in,   24 out;  Image size: 300 in, 150 out
        self.down2 = StackEncoder(24, 64, kernel_size=3)   # Channels: 24 in,  64 out;  Image size: 150 in, 75 out
        self.down3 = StackEncoder(64, 128, kernel_size=3)  # Channels: 64 in,  128 out; Image size: 75 in,  38 out
        self.down4 = StackEncoder(128, 256, kernel_size=3) # Channels: 128 in, 256 out; Image size: 38 in,  19 out
        self.down5 = StackEncoder(256, 512, kernel_size=3) # Channels: 256 in, 512 out; Image size: 19 in,  10 out
        self.down6 = StackEncoder(512, 768, kernel_size=3) # Channels: 512 in, 768 out; Image size: 10 in,  5 out

        self.center = torch.nn.Sequential(
            ConvBnRelu2d(768, 768, kernel_size=3, padding=1, stride=1), # Channels: 768 in, 768 out; Image size: 5 in,  5 out
        )

        # x_big_channels, x_channels, y_channels
        self.up6 = StackDecoder(768, 768, 512, kernel_size=3) # Channels: 768+768 = 1536 in, 512 out; Image size: 5 in,   10 out
        self.up5 = StackDecoder(512, 512, 256, kernel_size=3) # Channels: 512+512 = 1024 in, 256 out; Image size: 10 in,  19 out
        self.up4 = StackDecoder(256, 256, 128, kernel_size=3) # Channels: 256+256 = 512  in, 128 out; Image size: 19 in,  38 out
        self.up3 = StackDecoder(128, 128, 64, kernel_size=3)  # Channels: 128+128 = 256  in, 64  out; Image size: 38 in,  75 out
        self.up2 = StackDecoder(64, 64, 24, kernel_size=3)    # Channels: 64+64   = 128  in, 24  out; Image size: 75 in,  150 out
        self.up1 = StackDecoder(24, 24, 24, kernel_size=3)    # Channels: 24+24   = 48   in, 24  out; Image size: 150 in, 300 out
        self.classify = torch.nn.Conv2d(24, 5, kernel_size=1, padding=0, stride=1, bias=True) # Channels: 24 in, 1 out; Image size: 300 in, 300 out

    def forward(self, x):
        out = x  # ;print('x    ',x.size())
        down1, out = self.down1(out)  #;print('down1',down1.size())  #256
        down2, out = self.down2(out)  #;print('down2',down2.size())  #128
        down3, out = self.down3(out)  #;print('down3',down3.size())  #64
        down4, out = self.down4(out)  #;print('down4',down4.size())  #32
        down5, out = self.down5(out)  #;print('down5',down5.size())  #16
        down6, out = self.down6(out)  #;print('down6',down6.size())  #8
        pass  # ;print('out  ',out.size())

        out = self.center(out)#;print('center',out.size())  #256
        out = self.up6(down6, out)#;print('up6',out.size())  #256
        out = self.up5(down5, out)#;print('up5',out.size())  #256
        out = self.up4(down4, out)#;print('up4',out.size())  #256
        out = self.up3(down3, out)#;print('up3',out.size())  #256
        out = self.up2(down2, out)#;print('up2',out.size())  #256
        out = self.up1(down1, out)#;print('up1',out.size())  #256

        out = self.classify(out)#;print('classify',out.size())  #256
        return out

        out = x  # ;print('x    ',x.size())
        #
        down1, out = self.down1(out)  #; print('down1',down1.size())  #256
        down2, out = self.down2(out)  #; print('down2',down2.size())  #128
        down3, out = self.down3(out)  #; print('down3',down3.size())  #64
        pass  # ;print('out  ',out.size())
        out = self.center(out) #; print('center',out.size())  #64
        out = self.up3(down3, out) #; print('up3',out.size())  #64
        out = self.up2(down2, out) #; print('up2',out.size())  #64
        out = self.up1(down1, out) #; print('up1',out.size())  #64
        #out = torch.nn.functional.upsample(out, size=(128, 128), mode='bilinear', align_corners=True)#; print('resample',out.size())  #64
        #out = self.extra(out); print('up1',out.size())  #64
        # 1024
        out = self.classify(out); #print('classify',out.size())  #64
        out = torch.squeeze(out, dim=1) #; print('classifier',out.size())  #64

        return out                         