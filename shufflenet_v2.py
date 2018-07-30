
'''
	ShuffleNet_v2

Author: Zhengwei Li
Data: July 30 2018
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from collections import OrderedDict
from torch.nn import init




def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.data.size()

    channels_per_group = num_channels // groups
    
    # reshape
    x = x.view(batchsize, groups, 
        channels_per_group, height, width)

    # transpose
    # - contiguous() required if transpose() is used before view().
    #   See https://github.com/pytorch/pytorch/issues/764
    x = torch.transpose(x, 1, 2).contiguous()

    # flatten
    x = x.view(batchsize, -1, height, width)

    return x


class ShuffleUnit_v2(nn.Module):
    def __init__(self, in_channels, channels_split=True, split=2):
        """channels_split : True / False;  
            split         : channel split 
                            2  : 1:1
                            3  : 1:2
                            7  : 1:6
                            13 : 1:12
        """
        
        super(ShuffleUnit_v2, self).__init__()

        self.in_channels = in_channels
        self.channels_split = channels_split
        self.split = split


        if self.channels_split:
            # channel spilt 
            # no downsample

            split_channel = self.in_channels - int(self.in_channels // self.split)

            self.residual_branch = nn.Sequential(
                # 1x1 conv
                nn.Conv2d(split_channel, split_channel, 1, 1, 0, bias=False),
                nn.BatchNorm2d(split_channel),
                nn.ReLU(inplace=True),
                # DW conv stride = 1
                nn.Conv2d(split_channel, split_channel, 3, 1, 1, groups=split_channel, bias=False),
                nn.BatchNorm2d(split_channel),
    
                # 1x1 conv
                nn.Conv2d(split_channel, split_channel, 1, 1, 0, bias=False),
                nn.BatchNorm2d(split_channel),
                nn.ReLU(inplace=True)
                )
        else:
            # no channel spilt 
            # downsample
            channels = self.in_channels
            self.residual_branch = nn.Sequential(
                # 1x1 conv
                nn.Conv2d(channels, channels, 1, 1, 0, bias=False),
                nn.BatchNorm2d(channels),
                nn.ReLU(inplace=True),
                # DW conv stride = 2
                nn.Conv2d(channels, channels, 3, 2, 1, groups=channels, bias=False),
                nn.BatchNorm2d(channels),
    
                # 1x1 conv
                nn.Conv2d(channels, channels, 1, 1, 0, bias=False),
                nn.BatchNorm2d(channels),
                nn.ReLU(inplace=True)
                )

            self.primary_branch = nn.Sequential(

                # DW conv stride = 2
                nn.Conv2d(channels, channels, 3, 2, 1, groups=channels, bias=False),
                nn.BatchNorm2d(channels),
                # 1x1 conv
                nn.Conv2d(channels, channels, 1, 1, 0, bias=False),
                nn.BatchNorm2d(channels),
                nn.ReLU(inplace=True),
            )

    def forward(self, x):

        if self.channels_split:

            x_primary = x[:,0:int(self.in_channels // self.split),:,:]
            x_residual = x[:,int(self.in_channels // self.split):self.in_channels,:,:]
            x_residual = self.residual_branch(x_residual)
            x = torch.cat((x_primary, x_residual), dim=1)
            x = channel_shuffle(x, groups=2)

        else:
            x_primary = self.primary_branch(x)
            x_residual = self.residual_branch(x)
            x = torch.cat((x_primary, x_residual), dim=1)
            x = channel_shuffle(x, groups=2)

        return x

class ShuffleNet_v2(nn.Module):
    """ShuffleNet_v2 implementation.
    """

    def __init__(self, expand=1, in_channels=3, n_classes=1000):
        """ShuffleNet_v2.

        Arguments:
            in_channels (int, optional): number of channels in the input tensor.
                Default is 3 for RGB image inputs.
            expand: channels 0.5x, 1x, 1.5x, 2x
            
        """
        super(ShuffleNet_v2, self).__init__()

        self.expand = expand
        self.stage_repeats = [3, 7, 3]
        self.in_channels =  in_channels
        self.num_classes = n_classes


        if self.expand == 0:
            self.stage_out_channels = [3, 24, 48, 96, 192, 1024]
        elif self.expand == 1:
            self.stage_out_channels = [3, 58, 116, 232, 464, 1024]
        elif self.expand == 2:
            self.stage_out_channels = [3, 88, 176, 352, 704, 1024]
        elif self.expand == 3:
            self.stage_out_channels = [3, 122, 244, 488, 976, 2048]

        
        # Stage 1 always has 24 output channels
        self.conv1 = nn.Conv2d(self.in_channels,
                             self.stage_out_channels[1], # stage 1
                             kernel_size=3,
                             stride=2,
                             padding=1)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Stage 2
        self.stage2 = self._make_stage(2)
        # Stage 3
        self.stage3 = self._make_stage(3)
        # Stage 4
        self.stage4 = self._make_stage(4)
        self.conv5 = nn.Conv2d(self.stage_out_channels[4],
                             self.stage_out_channels[5], # stage 1
                             kernel_size=1,
                             stride=1,
                             padding=0)

        num_inputs = self.stage_out_channels[-1]
        self.fc = nn.Linear(num_inputs, self.num_classes)        

        self.init_params()


    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)


    def _make_stage(self, stage):
        modules = OrderedDict()
        stage_name = "ShuffleUnit_Stage{}".format(stage)
        
        # First ShuffleUnit in the stage
        first_module = ShuffleUnit_v2(
            self.stage_out_channels[stage-1],
            channels_split=False
            )
        modules[stage_name+"_0"] = first_module

        # add more ShuffleUnits depending on pre-defined number of repeats
        for i in range(self.stage_repeats[stage-2]):
            name = stage_name + "_{}".format(i+1)
            module = ShuffleUnit_v2(
                self.stage_out_channels[stage],
                channels_split=True,
                split=2
                )
            modules[name] = module

        return nn.Sequential(modules)


    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)

        x = self.stage2(x)     
        x = self.stage3(x)
        x = self.stage4(x)

        x = self.conv5(x)
        # global average pooling layer
        x = F.avg_pool2d(x, x.data.size()[-2:])
        
        # flatten for input to fully-connected layer
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return F.log_softmax(x, dim=1)

        


if __name__ == "__main__":
    """Testing
    """
    model = ShuffleNet_v2()
    x = torch.randn(1,3,224,224)
    y = model(x)
    print(model)
    print(y.size())
