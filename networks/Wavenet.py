import torch
import torch.nn as nn
from networks.wavemlp import WaveMLP_S
from timm.models.layers import DropPath
from pytorch_wavelets import DTCWTForward, DTCWTInverse
from torch.nn.functional import kl_div

class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class TransBasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=2, stride=2, padding=0, dilation=1,output_padding=0, bias=False):
        super(TransBasicConv2d, self).__init__()
        self.Deconv = nn.ConvTranspose2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding,output_padding= output_padding, dilation=dilation, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)
        self.inch = in_planes
    def forward(self, x):

        x = self.Deconv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class Mlp(nn.Module):
    def __init__(self, in_features=64, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        # print('x',x.shape)
        x = self.fc1(x)
        # print('fc',x.shape)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Mlp_wave(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()

        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.act = act_layer()
        self.drop = nn.Dropout(drop)
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1, 1)
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class BAB_Decoder(nn.Module):
    def __init__(self, channel_1=1024, channel_2=512, channel_3=256, dilation_1=3, dilation_2=2):
        super(BAB_Decoder, self).__init__()

        self.conv1 = BasicConv2d(channel_1, channel_2, 3, padding=1)
        self.conv1_Dila = BasicConv2d(channel_2, channel_2, 3, padding=dilation_1, dilation=dilation_1)
        self.conv2 = BasicConv2d(channel_2, channel_2, 3, padding=1)
        self.conv2_Dila = BasicConv2d(channel_2, channel_2, 3, padding=dilation_2, dilation=dilation_2)
        self.conv3 = BasicConv2d(channel_2, channel_2, 3, padding=1)
        self.conv_fuse = BasicConv2d(channel_2*3, channel_3, 3, padding=1)

    def forward(self, x):
        x1 = self.conv1(x)
        x1_dila = self.conv1_Dila(x1)

        x2 = self.conv2(x1)
        x2_dila = self.conv2_Dila(x2)

        x3 = self.conv3(x2)

        x_fuse = self.conv_fuse(torch.cat((x1_dila, x2_dila, x3), 1))

        return x_fuse

class FFT(nn.Module):
    def __init__(self,inchannel,outchannel):
        super().__init__()
        self.DWT = DTCWTForward(J=3, biort='near_sym_b', qshift='qshift_b')
        # self.DWT =DTCWTForward(J=3, include_scale=True)
        self.IWT = DTCWTInverse(biort='near_sym_b', qshift='qshift_b')
        self.conv1 = BasicConv2d(outchannel, outchannel)
        self.conv2 = BasicConv2d(inchannel, outchannel)
        self.conv3 = BasicConv2d(outchannel, outchannel)
        self.change = TransBasicConv2d(outchannel, outchannel)

    def forward(self, x, y):
        y = self.conv2(y)
        Xl, Xh = self.DWT(x)
        Yl, Yh = self.DWT(y)
        x_y = self.conv1(Xl) + self.conv1(Yl)

        x_m = self.IWT((x_y, Xh))
        y_m = self.IWT((x_y, Yh))

        out = self.conv3(x_m + y_m)
        return out

class PATM_BAB(nn.Module):
    def __init__(self, channel_1=1024, channel_2=512, channel_3=256, dilation_1=3, dilation_2=2):
        super().__init__()
        self.conv1 = BasicConv2d(channel_1, channel_2, 3, padding=1)
        self.conv1_Dila = BasicConv2d(channel_2, channel_2, 3, padding=dilation_1, dilation=dilation_1)
        self.conv2 = BasicConv2d(channel_2, channel_2, 3, padding=1)
        self.conv2_Dila = BasicConv2d(channel_2, channel_2, 3, padding=dilation_2, dilation=dilation_2)
        self.conv3 = BasicConv2d(channel_2, channel_2, 3, padding=1)
        self.conv_fuse = BasicConv2d(channel_2 *2, channel_3, 3, padding=1)
        self.drop = nn.Dropout(0.5)
        self.conv_last=TransBasicConv2d(channel_3, channel_3,  kernel_size=2, stride=2,
                             padding=0, dilation=1, bias=False)
    def forward(self, x):
        x1 = self.conv1(x)
        x1_dila = self.conv1_Dila(x1)

        x2 = self.conv2(x1)
        x2_dila = self.conv2_Dila(x2)

        x3 = self.conv3(x2)
        x1_dila = torch.cat([x1_dila * torch.cos(x1_dila), x1_dila * torch.sin(x1_dila)], dim=1)
        x2_dila = torch.cat([x2_dila * torch.cos(x2_dila), x2_dila * torch.sin(x2_dila)], dim=1)
        x3 = torch.cat([x3 * torch.cos(x3), x3 * torch.sin(x3)], dim=1)
        # print('x1_dila + x2_dila+x3',x1_dila.shape)
        x_fuse = self.conv_fuse(x1_dila + x2_dila +x3)
        # x_fuse = self.conv_fuse(torch.cat((x1_dila, x2_dila, x3), 1))
        # print('x_f',x_fuse.shape)
        x_fuse= self.drop(x_fuse)
        # print()
        x_fuse = self.conv_last(x_fuse)
        return x_fuse

class DWT(nn.Module):
    def __init__(self, inchannel,outchannel):
        super(DWT, self).__init__()
        self.DWT = DTCWTForward(J=3, biort='near_sym_b', qshift='qshift_b')
        self.IWT = DTCWTInverse(biort='near_sym_b', qshift='qshift_b')
        self.conv1 = BasicConv2d(outchannel,outchannel)
        self.conv2 = BasicConv2d(inchannel, outchannel)
        self.conv3 = BasicConv2d(outchannel, outchannel)
        self.change = TransBasicConv2d(outchannel,outchannel)
    def forward(self, x, y):
        # print('x',x.shape)
        y = self.change(self.conv2(y))
        # print('y',y.shape)
        Xl, Xh = self.DWT(x)
        Yl, Yh = self.DWT(y)
        # print('Xl',Xl.shape)
        x_y = self.conv1(Xl)+self.conv1(Yl)
        # print('x_y',x_y.shape)
        # print('Xh',Xh.shape)
        # print('Yh',Yh.shape)
        x_m = self.IWT((x_y,Xh))
        y_m = self.IWT((x_y,Yh))
        # print('x_m',x_m.shape)
        # print('y_m',y_m.shape)
        out = self.conv3(x_m + y_m)
        return out
class Edge_Aware(nn.Module):
    def __init__(self, ):
        super(Edge_Aware, self).__init__()
        self.conv1 = TransBasicConv2d(512, 64,kernel_size=4,stride=8,padding=0,dilation=2,output_padding=1)
        self.conv2 = TransBasicConv2d(320, 64,kernel_size=2,stride=4,padding=0,dilation=2,output_padding=1)
        self.conv3 = TransBasicConv2d(128, 64,kernel_size=2,stride=2,padding=1,dilation=2,output_padding=1)
        self.pos_embed = BasicConv2d(64, 64 )
        self.pos_embed3 = BasicConv2d(64, 64)
        self.conv31 = nn.Conv2d(64,1, kernel_size=1)
        self.conv512_64 = TransBasicConv2d(512,64)
        self.conv320_64 = TransBasicConv2d(320, 64)
        self.conv128_64 = TransBasicConv2d(128, 64)
        self.up = nn.Upsample(56)
        self.up2 = nn.Upsample(384)
        self.norm1 = nn.LayerNorm(64)
        self.norm2 = nn.BatchNorm2d(64)
        self.drop_path = DropPath(0.3)
        self.maxpool =nn.AdaptiveMaxPool2d(1)
        # self.qkv = nn.Linear(64, 64 * 3, bias=False)
        self.num_heads = 8
        self.mlp1 = Mlp(in_features=64, out_features=64)
        self.mlp2 = Mlp(in_features=64, out_features=64)
        self.mlp3 = Mlp(in_features=64, out_features=64)
    def forward(self, x, y, z, v):


        # v = self.conv1(v)
        # z = self.conv2(z)
        # y = self.conv3(y)
        # print('v',v)
        v = self.up(self.conv512_64(v))
        z = self.up(self.conv320_64(z))
        y = self.up(self.conv128_64(y))
        x = self.up(x)

        x_max = self.maxpool(x)
        # print('x_max',x_max.shape)
        b,_,_,_ = x_max.shape
        x_max = x_max.reshape(b, -1)
        x_y = self.mlp1(x_max)
        # print('s',x_y.shape)
        x_z = self.mlp2(x_max)
        x_v = self.mlp3(x_max)

        x_y = x_y.reshape(b,64,1,1)
        x_z = x_z.reshape(b, 64, 1, 1)
        x_v = x_v.reshape(b, 64, 1, 1)
        x_y = torch.mul(x_y, y)
        x_z = torch.mul(x_z, z)
        x_v = torch.mul(x_v, v)


        # x_mix_1 = torch.cat((x_y,x_z,x_v),dim=1)
        x_mix_1 = x_y+ x_z+ x_v
        # print('sd',x_mix_1.shape)
        x_mix_1 =  self.norm2(x_mix_1)
        # print('x_mix_1',x_mix_1.shape)
        x_mix_1= self.pos_embed3(x_mix_1)
        x_mix = self.drop_path(x_mix_1)
        x_mix = x_mix_1 + self. pos_embed3(x_mix)
        x_mix = self.up2(self.conv31(x_mix))
        return x_mix

class Mutual_info_reg(nn.Module):
    def __init__(self, input_channels=64, channels=64, latent_size=6):
        super(Mutual_info_reg, self).__init__()
        self.soft = torch.nn.Softmax(dim=1)
    def forward(self, rgb_feat, depth_feat):

        # print('rgb_feat',rgb_feat.shape)
        # print('depth_feat', depth_feat.shape)
        rgb_feat = self.soft(rgb_feat)
        depth_feat = self.soft(depth_feat)
        #
        # print('rgb_feat',rgb_feat.shape)
        # print('depth_feat', depth_feat.shape)
        return  kl_div(rgb_feat.log(), depth_feat)

class WaveNet(nn.Module):
    def __init__(self, channel=32):
        super(WaveNet, self).__init__()


        self.encoderR = WaveMLP_S()
        # Lateral layers
        self.lateral_conv0 = BasicConv2d(64, 64, 3, stride=1, padding=1)
        self.lateral_conv1 = BasicConv2d(128, 64, 3, stride=1, padding=1)
        self.lateral_conv2 = BasicConv2d(320, 128, 3, stride=1, padding=1)
        self.lateral_conv3 = BasicConv2d(512, 320, 3, stride=1, padding=1)


        self.FFT1 = FFT(64,64)
        self.FFT2 = FFT(128,128)
        self.FFT3 = FFT(320,320)
        self.FFT4 = FFT(512,512)



        self.conv512_64 = BasicConv2d(512, 64)
        self.conv320_64 = BasicConv2d(320, 64)
        self.conv128_64 = BasicConv2d(128, 64)
        self.sigmoid = nn.Sigmoid()
        self.S4 = nn.Conv2d(512, 1, 3, stride=1, padding=1)
        self.S3 = nn.Conv2d(320, 1, 3, stride=1, padding=1)
        self.S2 = nn.Conv2d(128, 1, 3, stride=1, padding=1)
        self.S1 = nn.Conv2d(64, 1, 3, stride=1, padding=1)
        self.up1 = nn.Upsample(384)
        self.up2 = nn.Upsample(384)
        self.up3 = nn.Upsample(384)
        self.up_loss = nn.Upsample(92)
        # Mutual_info_reg1
        self.mi_level1 = Mutual_info_reg(64, 64, 6)
        self.mi_level2 = Mutual_info_reg(64, 64, 6)
        self.mi_level3 = Mutual_info_reg(64, 64, 6)
        self.mi_level4 = Mutual_info_reg(64, 64, 6)

        self.edge = Edge_Aware()
        self.PATM4 = PATM_BAB(512, 512, 512, 3, 2)
        self.PATM3 = PATM_BAB(832, 512, 320, 3, 2)
        self.PATM2 = PATM_BAB(448, 256, 128, 5, 3)
        self.PATM1 = PATM_BAB(192, 128, 64, 5, 3)

    def forward(self, x_rgb,x_thermal):
        x0,x1,x2,x3 = self.encoderR(x_rgb)
        y0, y1, y2, y3 = self.encoderR(x_thermal)

        x2_ACCoM = self.FFT1(x0, y0)
        x3_ACCoM = self.FFT2(x1, y1)
        x4_ACCoM = self.FFT3(x2, y2)
        x5_ACCoM = self.FFT4(x3, y3)

        edge = self.edge(x2_ACCoM, x3_ACCoM, x4_ACCoM, x5_ACCoM)

        mer_cros4 = self.PATM4(x5_ACCoM)
        m4 = torch.cat((mer_cros4,x4_ACCoM),dim=1)
        mer_cros3 = self.PATM3(m4)
        m3 = torch.cat((mer_cros3, x3_ACCoM), dim=1)
        mer_cros2 = self.PATM2(m3)
        m2 = torch.cat((mer_cros2, x2_ACCoM), dim=1)
        mer_cros1 = self.PATM1(m2)

        s1 = self.up1(self.S1(mer_cros1))
        s2 = self.up2(self.S2(mer_cros2))
        s3 = self.up3(self.S3(mer_cros3))
        s4 = self.up3(self.S4(mer_cros4))

        x_loss0 = x0
        y_loss0 = y0
        x_loss1 = self.up_loss(self.conv128_64(x1))
        y_loss1 = self.up_loss(self.conv128_64(y1))
        x_loss2 = self.up_loss(self.conv320_64(x2))
        y_loss2 = self.up_loss(self.conv320_64(y2))
        x_loss3 = self.up_loss(self.conv512_64(x3))
        y_loss3 = self.up_loss(self.conv512_64(y3))

        lat_loss0 = self.mi_level1(x_loss0, y_loss0)
        lat_loss1 = self.mi_level2(x_loss1, y_loss1)
        lat_loss2 = self.mi_level3(x_loss2, y_loss2)
        lat_loss3 = self.mi_level4(x_loss3, y_loss3)
        lat_loss = lat_loss0 + lat_loss1 + lat_loss2 + lat_loss3
        return s1, s2, s3, s4, self.sigmoid(s1), self.sigmoid(s2), self.sigmoid(s3), self.sigmoid(s4),edge,lat_loss

if __name__=='__main__':
    image = torch.randn(1, 3, 384, 384).cuda(0)
    ndsm = torch.randn(1, 64, 56, 56)
    ndsm1 = torch.randn(1, 128, 28, 28)
    ndsm2 = torch.randn(1, 320, 14, 14)
    ndsm3 = torch.randn(1, 512, 7, 7)

    net = WaveNet().cuda()

    # """修改权重
    from config import opt
    save_path = opt.save_path
    net = WaveNet().cuda(0)
    pth = torch.load('/home/hjk/桌面/代码/Wave_Swin_384_Stu_val1000_RGBT_share.pth')
    net.load_state_dict(
        torch.load('/home/hjk/桌面/代码/Wave_Swin_384_Stu_val1000_RGBT_share.pth'),strict=False)
    print()
    net.eval()
    with torch.no_grad():
        torch.save(net.state_dict(), '/home/hjk/桌面/代码/RES34_1_best_mae_test.pth')

    # """
