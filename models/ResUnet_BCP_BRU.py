import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from .resnet import conv1x1, conv3x3, resnet18, resnet50
sys.path.append('..')
from detect_occ.utils.utility import kaiming_init, init_weights
import pdb


def ResUnet_BCP_BRU(config, weights=None):
    """create UNet with given config"""
    model = ResUnetModel(config, encoder_arch='resnet50', pretrained=config.network.pretrained)

    if config.network.init_type == 'kaiming':
        model.apply(kaiming_init)
        print('=> use kaiming init')

    if weights is not None:
        model.load_state_dict(weights['state_dict'])

    return model



# /////////////////////////////////////////////
#           define the whole network
# \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
class ResUnetModel(nn.Module):
    def __init__(self, config, encoder_arch='resnet50', pretrained=True):
        """
        Unet arch with resnet encoder
        :param inplanes: num of encoder first conv out channel
        """
        super(ResUnetModel, self).__init__()
        self.config = config
        if encoder_arch == 'resnet50':
            self.encoder_out_ch = [64, 256, 512, 1024]  # to conv4
            # self.decoder_out_ch = [1024, 512, 256, 64]
        elif encoder_arch == 'resnet18':
            self.encoder_out_ch = [64, 64, 128, 256]
            self.decoder_out_ch = [256, 128, 64, 64]

        # create encoder and load pretrained weights if needed
        self.inconv = DoubleConvBlock(in_nc=config.network.in_channels, mid_nc=64, out_nc=64, use_bias=True)

        # custom resnet encoder conv1
        if config.network.in_channels != 3:
            self.encode1 = nn.Sequential(
                nn.Conv2d(config.network.in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),
                nn.BatchNorm2d(64), nn.ReLU(inplace=True))

        if encoder_arch == 'resnet50':
            self.encoder = resnet50(pretrained=pretrained)
        elif encoder_arch == 'resnet18':
            self.encoder = resnet18(pretrained=pretrained)
        if pretrained:
            print('=> load imagenet pretrained weights')

        # bottleneck
        # self.middle = DoubleConvBlock(in_nc=1024, mid_nc=512, out_nc=512, use_bias=True)

        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        # !!!!!!!!!! remember to adjust !!!!!!!!!!!!!!!!
        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        self.channel1 = ChannelConV(1024, 512)
        self.channel2 = ChannelConV(512, 256)
        self.channel3 = ChannelConV(256, 128)
        self.channel4 = ChannelConV(64, 64)

        # bottleneck -> BCP module
        self.bcp = BCP(1024, 512)

        # decoder
        # self.deconv3 = UnetUpBlock(in_ch=1024, mid_ch=256, out_ch=256, config=config)
        # self.deconv2 = UnetUpBlock(in_ch=512, mid_ch=64, out_ch=64, config=config)
        # self.deconv1 = UnetUpBlock(in_ch=128, mid_ch=64, out_ch=64, config=config)
        # self.deconv0 = UnetUpBlock(in_ch=128, mid_ch=64, out_ch=64, config=config)

        # decoder -> BRU module
        self.bru1 = BRU(512, 512, 256, 256)
        self.bru2 = BRU(256, 256, 128, 128)
        self.bru3 = BRU(128, 128, 64, 64)
        self.bru4 = BRU(64, 64, 64, 64)

        self.crop = Crop(2)

        ## output layers of occlusion edge
        self.out0_edge = OutputBlock(64, 8, 4, 1)
        self.out0_ori = OutputBlock_ori(64, 1, kernel_size=1, use_bias=True, activation='None')

    def forward(self, input0):
        """layer index indicate spatial size"""
        conv0 = self.inconv(input0)  # in_ch=>64
        if self.config.network.in_channels == 3:
            conv1, conv2, conv3, conv4 = self.encoder.forward(input0)  # 3=>64,256,512,1024,2048
        else:
            conv1 = self.encode1(input0)  # in_ch=>64
            conv2, conv3, conv4 = self.encoder.forward_2to4(conv1)  # 64=>256,512,1024

        middle = self.bcp(conv4)

        # deconv3 = self.deconv3.forward(middle, conv3)  # (512+512)=>256
        # deconv2 = self.deconv2.forward(deconv3, conv2)  # (256+256)=>64
        # deconv1 = self.deconv1.forward(deconv2, conv1)  # (64+64)=>64
        # deconv0 = self.deconv0.forward(deconv1, conv0)  # (64+64)=>64
        
        conv4 = self.channel1(conv4)
        conv3 = self.channel2(conv3)
        conv2 = self.channel3(conv2)
        conv1 = self.channel4(conv1)

        deconv4 = self.bru1(middle, conv4)
        deconv3 = self.bru2(deconv4, conv3)
        deconv2 = self.bru3(deconv3, conv2)
        deconv1 = self.bru4(deconv2, conv1)

        deconv0 = self.crop(deconv1, conv0)

        out0_edge = self.out0_edge(deconv0)
        out0_ori = self.out0_ori(deconv0)

        return out0_edge, out0_ori


class OutputBlock(nn.Module):
    def __init__(self, input_nc, mid_nc=8, output_nc=4, final_nc=1, use_bias=True):
        super(OutputBlock, self).__init__()
        self.model = nn.Sequential(
                nn.Conv2d(input_nc, mid_nc, kernel_size=3, padding=1, stride=1, bias=use_bias),
                nn.BatchNorm2d(mid_nc),
                nn.ReLU(inplace=True),

                nn.Conv2d(mid_nc, output_nc, kernel_size=3, padding=1, stride=1, bias=use_bias),
                nn.BatchNorm2d(output_nc),
                nn.ReLU(inplace=True),
            )

        self.edge = nn.Sequential(
                nn.Conv2d(output_nc, final_nc, padding=0, stride=1, kernel_size=1, bias=use_bias)
            )

    def forward(self, x):
        edge = self.model(x)
        return self.edge(edge)


class OutputBlock_ori(nn.Module):
    def __init__(self, input_nc, output_nc, kernel_size=1, use_bias=True, activation='none'):
        super(OutputBlock_ori, self).__init__()

        model = [
            nn.Conv2d(input_nc, output_nc, kernel_size=kernel_size, padding=0, bias=use_bias),
        ]

        if activation == 'tanh':
            model.append(nn.Tanh())  # [-1, 1]
        elif activation == 'sigmoid':
            model.append(nn.Sigmoid())  # [0, 1]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


class DoubleConvBlock(nn.Module):
    def __init__(self, in_nc, mid_nc, out_nc, use_bias=True):
        super(DoubleConvBlock, self).__init__()

        self.model = nn.Sequential(
            nn.Conv2d(in_nc, mid_nc, kernel_size=3, stride=1, padding=1, bias=use_bias),
            nn.BatchNorm2d(mid_nc),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_nc, out_nc, kernel_size=3, stride=1, padding=1, bias=use_bias),
            nn.BatchNorm2d(out_nc),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.model(x)




class BCP(nn.Module):

    '''
        @ input :   input_nc    ->  input channel
                    output_nc   ->  output channel
    '''
    def __init__(self, input_nc=2048, output_nc=256):
        super(BCP, self).__init__()
        self.result = None
        self.plain6 = nn.Sequential(
            # plain6a
                nn.Conv2d(input_nc, output_nc, kernel_size=3, stride=1, padding=1), # lr_mult?????
                nn.BatchNorm2d(output_nc),
                nn.ReLU(inplace=True),

            # plain6b
                nn.Conv2d(output_nc, output_nc, kernel_size=3, stride=1, padding=1), # lr_mult?????
                nn.BatchNorm2d(output_nc),
                nn.ReLU(inplace=True)
            )

        self.onePoneConV = nn.Conv2d(output_nc, output_nc, kernel_size=1, stride=1, padding=0) # lr_mult & bias_term & bias_filler ??????

        self.W = nn.Sequential(
                nn.Conv2d(output_nc, output_nc, kernel_size=1, stride=1, padding=0), # lr_mult & bias_term & bias_filler ??????
                nn.BatchNorm2d(output_nc)
            )


    def forward(self, x):
        x = self.plain6(x)
        x = self.onePoneConV(x)

        n, c, w, h = x.shape

        x = torch.reshape(x, (n, c, -1))

        self.phi = x                                    # phi_reshape
        self.theta = torch.transpose(x, 1, 2)           # theta_transpose
        self.G = torch.transpose(x, 1, 2)               # G_transpose

        self.ajacent = torch.bmm(self.phi, self.theta)  # ajacent
        self.ajacent = F.softmax(self.ajacent, dim=2)   # ajacent_softmax

        self.G = torch.bmm(self.G, self.ajacent)        # G_round
        self.G = torch.transpose(self.G, 1, 2)          # G_round_transpose
        self.G = torch.reshape(self.G, (n, -1, w, h))    # G_reshape2 

        # self.result = self.W(self.G).cpu().numpy()
        
        return self.W(self.G)



'''
    define the "Crop" Layer in CAFFE via PyTorch
'''
class Crop(nn.Module):
    def __init__(self, axis, offset=0):
        super(Crop, self).__init__()
        self.axis = axis
        self.offset = offset

    def forward(self, x, ref):
        for axis in range(self.axis, x.dim()):
            ref_size = ref.size(axis)
            indices = torch.arange(self.offset, self.offset + ref_size).long()
            indices = x.data.new().resize_(indices.size()).copy_(indices)
            x = x.index_select(axis, indices.to(torch.int64))
        return x


'''
    define the "Concat" Layer in CAFFE via PyTorch
'''
class Concat(nn.Module):
    def __init__(self, axis):
        super(Concat, self).__init__()
        self.axis = axis

    def forward(self, *inputs):
        return torch.cat(inputs, self.axis)



'''
    define the "Eltwise" Layer in CAFFE via PyTorch
'''
class Eltwise(nn.Module):
    def __init__(self, operation='+'):
        super(Eltwise, self).__init__()
        self.operation = operation

    def forward(self, *inputs):
        if self.operation == '+' or self.operation == 'SUM':
            x = inputs[0]
            for i in range(1,len(inputs)):
                x = x + inputs[i]
        elif self.operation == '*' or self.operation == 'MUL':
            x = inputs[0]
            for i in range(1,len(inputs)):
                x = x * inputs[i]
        elif self.operation == '/' or self.operation == 'DIV':
            x = inputs[0]
            for i in range(1,len(inputs)):
                x = x / inputs[i]
        elif self.operation == 'MAX':
            x = inputs[0]
            for i in range(1,len(inputs)):
                x =torch.max(x, inputs[i])
        else:
            print('forward Eltwise, unknown operator')
        return x



'''
    Boundary Refinement Unit
'''
class BRU(nn.Module):

    '''
        @ input :   input_nc    ->  input channel
                    mid_nc      ->  mid layer's channel
                    output_nc   ->  output channel
                    groups      ->  <DeconV> param
                    axis        ->  <Crop> param
                    offset      ->  <Crop> param
    '''
    def __init__(self, input_nc, mid_nc, output_nc, groups, axis=2, offset=0):
        super(BRU, self).__init__()
        self.result = None
        self.side_output = None
        self.mul = None
        self.crop = Crop(axis, offset)
        self.eltwise_add = Eltwise('+')
        self.eltwise_mul = Eltwise('*')

        self.resres_fb1 = nn.Sequential(
            # resres_f_b1
            nn.Conv2d(input_nc, mid_nc, kernel_size=1, stride=1, padding=0),  # lr_mult?????
            nn.BatchNorm2d(mid_nc)
        )

        self.resres_fb2 = nn.Sequential(
            # resres_f_b2a
            nn.Conv2d(mid_nc, mid_nc, kernel_size=1, stride=1, padding=0),  # lr_mult?????
            nn.BatchNorm2d(mid_nc),
            nn.ReLU(inplace=True),

            # resres_f_b2b
            nn.Conv2d(mid_nc, mid_nc, kernel_size=3, stride=1, padding=1),  # lr_mult?????
            nn.BatchNorm2d(mid_nc),
            nn.ReLU(inplace=True),

            # resres_f_b2c
            nn.Conv2d(mid_nc, mid_nc, kernel_size=1, stride=1, padding=0),  # lr_mult?????
            nn.BatchNorm2d(mid_nc)
        )

        self.res_f_refine = nn.Sequential(
            # res_f_fuse
            nn.Conv2d(mid_nc, mid_nc, kernel_size=3, stride=1, padding=1),  # lr_mult?????
            nn.BatchNorm2d(mid_nc),
            nn.ReLU(inplace=True)
        )

        self.res_f_fuse = nn.Sequential(
            nn.Conv2d(mid_nc, output_nc, kernel_size=3, stride=1, padding=1),  # lr_mult?????
            nn.BatchNorm2d(output_nc),
            nn.ReLU(inplace=True)
        )

        self.deconv = nn.ConvTranspose2d(output_nc, output_nc, kernel_size=4, stride=2, padding=0, output_padding=0,
                                         groups=groups, bias=False)

    '''
        @ input :   last_BRU    ->  former BRU output
                    side_output ->  backbone's output

        ! notice:   last_BRU's W & H >= side_output's W & H   
    '''
    def forward(self, last_BRU, side_output):

        # self.side_output = side_output.cpu().numpy()

        self.last_BRU = self.crop(last_BRU, side_output)  # crop last_BRU in shape of side_output

        self.x1 = self.resres_fb1(self.last_BRU)
        self.x2 = self.resres_fb2(self.last_BRU)

        self.resres_f = F.relu(self.eltwise_add(self.x1, self.x2))
        self.res_f_mul = self.eltwise_mul(side_output, self.resres_f)

        # self.mul = self.res_f_mul.cpu().numpy()

        self.res_f_mul = self.res_f_refine(self.res_f_mul)

        self.res_f_mul_add = self.eltwise_add(self.last_BRU, self.res_f_mul)

        self.res_f_mul_add_fuse = self.res_f_fuse(self.res_f_mul_add)
        
        # self.result = self.res_f_mul_add_fuse.cpu().numpy()

        return self.deconv(self.res_f_mul_add_fuse)


class ChannelConV(nn.Module):
	def __init__(self, input_nc, output_nc, bias_use = False):
		super(ChannelConV, self).__init__()
		self.conv = nn.Sequential(
				nn.Conv2d(input_nc, output_nc, stride=1, padding=0, kernel_size=1, bias=bias_use),
				nn.BatchNorm2d(output_nc),
				nn.ReLU(inplace=True)
			)

	def forward(self, x):
		return self.conv(x)
