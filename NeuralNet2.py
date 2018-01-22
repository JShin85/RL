import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F



class ResidualBlock(nn.Module):
 expansion = 1    
     def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
         

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out    


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv3 = nn.Conv2d(19, 19, kernel_size=3, stride=1)
        self.bn3 = nn.BatchNorm2d(19)    
        self.conv_p = nn.Conv2d(2, 1, kernel_size=1, stride=1)
        self.bn_p = nn.BatchNorm2d(1)
        self.conv_v = nn.Conv2d(2, 1, kernel_size=1, stride=1)
        self.bn_v = nn.BatchNorm2d(1)
        self.f1 = nn.Linear(362, 2)
        self.f2 = nn.Linear(256, 2)
        self.f3 = nn.Linear

    def forward(self):
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn_p(self.conv_p(x)))
        x = F.self.f1(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.f2(x))
        x = x.view(x.size(0), -1)
        x = F.tanh(self.f3(x))

        return x

   

