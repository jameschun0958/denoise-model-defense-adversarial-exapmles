import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class comdefend(nn.Module):
    def __init__(self, noise=True, inference=False):
        super(comdefend, self).__init__()
        self.noise = noise
        self.inference = inference
        self.comcnn_layer = comCNN()
        self.recCNN_layer = resCNN()
    
    def gaussian(self, x, stddev=20):
        noise = Variable(x.data.new(x.size())).normal_(0.0, stddev)
        return x - noise

    def forward(self, x):

        linear_code = self.comcnn_layer(x)
        
        if(self.noise):
            nosiy_code = self.gaussian(linear_code)
        else:
            nosiy_code = linear_code
        
        binary_code = nn.Sigmoid()(nosiy_code)

        if(self.inference == True):
            binary_code = binary_code > 0.5
            binary_code = binary_code.float()
        
        out = self.recCNN_layer(binary_code)

        return linear_code, binary_code, out

class comCNN(nn.Module):
    def __init__(self):
        super(comCNN, self).__init__()
        self.com_conv1 = self.conv2d(3, 16)
        self.com_conv2 = self.conv2d(16, 32)
        self.com_conv3 = self.conv2d(32, 64)
        self.com_conv4 = self.conv2d(64, 128)
        self.com_conv5 = self.conv2d(128, 256)
        self.com_conv6 = self.conv2d(256, 128)
        self.com_conv7 = self.conv2d(128, 64)
        self.com_conv8 = self.conv2d(64, 32)
        self.com_out = self.conv2d(32, 12, use_elu=False)

    def conv2d(self, in_channel, out_channel, use_elu=True):
        if(use_elu):
            x = nn.Sequential(
                    nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=True),
                    nn.ELU()
            )
        else:
            x = nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=True)
            
        return x

    def forward(self, x):
        x = x - 0.5
        x = self.com_conv1(x)
        x = self.com_conv2(x)
        x = self.com_conv3(x)
        x = self.com_conv4(x)
        x = self.com_conv5(x)
        x = self.com_conv6(x)
        x = self.com_conv7(x)
        x = self.com_conv8(x)
        x = self.com_out(x)

        return x

class resCNN(nn.Module):
    def __init__(self):
        super(resCNN, self).__init__()
        self.rec_conv1 = self.conv2d(12, 32)
        self.rec_conv2 = self.conv2d(32, 64)
        self.rec_conv3 = self.conv2d(64, 128)
        self.rec_conv4 = self.conv2d(128, 256)
        self.rec_conv5 = self.conv2d(256, 128)
        self.rec_conv6 = self.conv2d(128, 64)
        self.rec_conv7 = self.conv2d(64, 32)
        self.rec_conv8 = self.conv2d(32, 16)
        self.rec_conv9 = self.conv2d(16, 3, use_elu=False)
        self.rec_out = nn.Sigmoid()    
        
    def conv2d(self, in_channel, out_channel, use_elu=True):
        if(use_elu):
            x = nn.Sequential(
                    nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=True),
                    nn.ELU()
            )
        else:
            x = nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=True)

        return x

    def forward(self, x):
        x = self.rec_conv1(x)
        x = self.rec_conv2(x)
        x = self.rec_conv3(x)
        x = self.rec_conv4(x)
        x = self.rec_conv5(x)
        x = self.rec_conv6(x)
        x = self.rec_conv7(x)
        x = self.rec_conv8(x)
        x = self.rec_conv9(x)
        x = self.rec_out(x)

        return x   
