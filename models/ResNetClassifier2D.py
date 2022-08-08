#!/usr/bin/env python

"""
Implimentation of the paper: Fei Wang et. al. "Residual Attention Network for Image Classification: - CVPR 2017 (https://arxiv.org/pdf/1704.06904)
With modifications: dynamic choice of the blocks
"""

import math
import torch.nn as nn
import torch.nn.functional as f
import torch


__author__ = "Soumick Chatterjee"
__copyright__ = "Copyright 2018, Soumick Chatterjee & OvGU:ESF:MEMoRIAL"
__credits__ = ["Soumick Chatterjee"]

__license__ = "GPL"
__version__ = "1.0.0"
__email__ = "soumick.chatterjee@ovgu.de"
__status__ = "Under Testing"

class AttentionModule_Adv(nn.Module):
    def __init__(self, in_channels, out_channels, ResidualBlock, n_stage, current_stage, drop_prob=0.0):
        super(AttentionModule_Adv, self).__init__()

        #Actual stages can be upto 3, as per the paper
        stages_per_actualstage = math.ceil(n_stage/3)
        self.actual_stage = current_stage//stages_per_actualstage

        #Initial Blocks, present in all actual stages
        self.first_residual_blocks = ResidualBlock(in_channels, out_channels, drop_prob=drop_prob)
        self.trunk_branches = nn.Sequential(
            ResidualBlock(in_channels, out_channels, drop_prob=drop_prob),
            ResidualBlock(in_channels, out_channels, drop_prob=drop_prob)
         )
        self.mpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.softmax1_blocks = nn.Sequential(
            ResidualBlock(in_channels, out_channels, drop_prob=drop_prob),
            ResidualBlock(in_channels, out_channels, drop_prob=drop_prob)
        )

        if self.actual_stage < 2: #For 1st and 2nd Stage
            self.skip1_connection_residual_block = ResidualBlock(in_channels, out_channels, drop_prob=drop_prob)
            self.mpool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        if self.actual_stage == 1:
            self.softmax2_blocks = nn.Sequential(
                ResidualBlock(in_channels, out_channels, drop_prob=drop_prob),
                ResidualBlock(in_channels, out_channels, drop_prob=drop_prob)
            )
            self.softmax3_blocks = ResidualBlock(in_channels, out_channels, drop_prob=drop_prob)
        elif self.actual_stage == 0:
            self.softmax2_blocks = ResidualBlock(in_channels, out_channels, drop_prob=drop_prob)
            self.skip2_connection_residual_block = ResidualBlock(in_channels, out_channels, drop_prob=drop_prob)
            self.mpool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            self.softmax3_blocks = nn.Sequential(
                ResidualBlock(in_channels, out_channels, drop_prob=drop_prob),
                ResidualBlock(in_channels, out_channels, drop_prob=drop_prob)
            )
            self.softmax4_blocks = ResidualBlock(in_channels, out_channels, drop_prob=drop_prob)
            self.softmax5_blocks = ResidualBlock(in_channels, out_channels, drop_prob=drop_prob)

        #Final Blocks, present in all actual stages
        self.softmaxN_blocks = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, bias=False),
            nn.Sigmoid()
        )
        self.last_blocks = ResidualBlock(in_channels, out_channels, drop_prob=drop_prob)

    def forward(self, x):
        size = x.shape[2:]
        
        #Initial Blocks, present in all actual stages
        x = self.first_residual_blocks(x)
        out_trunk = self.trunk_branches(x)
        out_mpool1 = self.mpool1(x)
        out_softmax1 = self.softmax1_blocks(out_mpool1)

        if self.actual_stage == 2: #3rd Stage
            out_softmaxNm1 = out_softmax1
        else:
            out_skip1_connection = self.skip1_connection_residual_block(out_softmax1)
            out_mpool2 = self.mpool2(out_softmax1)
            out_softmax2 = self.softmax2_blocks(out_mpool2)
            if self.actual_stage == 1: #2nd Stage                            
                out_softmax2_up = f.interpolate(out_softmax2, size=[x // 2 for x in size], mode='bilinear', align_corners=False)
                out_interp2 = out_softmax2_up + out_softmax1
                out = out_interp2 + out_skip1_connection
                out_softmaxNm1 = self.softmax3_blocks(out)
            elif self.actual_stage == 0: #1st Stage
                out_skip2_connection = self.skip2_connection_residual_block(out_softmax2)
                out_mpool3 = self.mpool3(out_softmax2)
                out_softmax3 = self.softmax3_blocks(out_mpool3)
                out_softmax3_up = f.interpolate(out_softmax3, size=[x // 4 for x in size], mode='bilinear', align_corners=False)
                out_interp3 = out_softmax3_up + out_softmax2
                out = out_interp3 + out_skip2_connection
                out_softmax4 = self.softmax4_blocks(out)
                out_softmax4_up = f.interpolate(out_softmax4, size=[x // 2 for x in size], mode='bilinear', align_corners=False)
                out_interp2 = out_softmax4_up + out_softmax1
                out = out_interp2 + out_skip1_connection
                out_softmaxNm1 = self.softmax5_blocks(out)

        #Final Blocks, present in all actual stages
        out_softmaxNm1_up = f.interpolate(out_softmaxNm1, size=size, mode='bilinear', align_corners=False)
        out_interpN = out_softmaxNm1_up + out_trunk
        out_softmaxN = self.softmaxN_blocks(out_interpN)
        out = (1 + out_softmaxN) * out_trunk
        return self.last_blocks(out)


class ResidualBottleneckBlock(nn.Module):
    ## This is a residual block with bottleneck, and with pre-activation
    def __init__(self, input_channels, output_channels, relu=nn.ReLU, norm=nn.BatchNorm2d, stride=1, drop_prob=0.0):
        super(ResidualBottleneckBlock, self).__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.stride = stride
        self.bn1 = norm(input_channels)
        self.relu = relu()
        self.conv1 = nn.Conv2d(input_channels, int(output_channels/4), 1, 1, bias = False)
        self.do1 = nn.Dropout2d(drop_prob)
        self.bn2 = norm(int(output_channels/4))
        self.relu = relu()
        self.conv2 = nn.Conv2d(int(output_channels/4), int(output_channels/4), 3, stride, padding = 1, bias = False)
        self.do2 = nn.Dropout2d(drop_prob)
        self.bn3 = norm(int(output_channels/4))
        self.relu = relu()
        self.conv3 = nn.Conv2d(int(output_channels/4), output_channels, 1, 1, bias = False)
        self.conv4 = nn.Conv2d(input_channels, output_channels , 1, stride, bias = False)
        
    def forward(self, x):
        residual = x
        out = self.bn1(x)
        out1 = self.relu(out)
        out = self.do1(self.conv1(out1))
        out = self.bn2(out)
        out = self.relu(out)
        out = self.do2(self.conv2(out))
        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)
        if (self.input_channels != self.output_channels) or (self.stride !=1 ):
            residual = self.conv4(out1)
        out += residual
        return out

class ResidualBasicBlock(nn.Module):
    #This is a Residual Basic block, with activation in middle and a added dropout from MRI Recon ResNet
    def __init__(self, in_features, out_features, relu=nn.ReLU, norm=nn.BatchNorm2d, stride=1, drop_prob=0.0):
        super(ResidualBasicBlock, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.stride = stride

        conv_block = [  nn.ReflectionPad2d(1),
                        nn.Conv2d(in_features, out_features//2, 3),
                        norm(out_features//2),
                        relu(),
                        nn.Dropout2d(drop_prob),
                        nn.ReflectionPad2d(1),
                        nn.Conv2d(out_features//2, out_features, 3, stride),
                        norm(out_features)  ]

        self.additonal_conv = nn.Conv2d(in_features, out_features , 1, stride)

        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        residual = x
        if (self.in_features != self.out_features) or (self.stride !=1 ):
            residual = self.additonal_conv(residual)
        return residual + self.conv_block(x)

class ResNetClassifier(nn.Module):
    def __init__(self, n_channels=1, n_class=10, starting_n_features=16, input_size=128, is_relu_leaky=True, is_bottleneck_residual=True, pool=None, norm=None, drop_prob=0.0, droppercent4attention=0.0): 
        super(ResNetClassifier, self).__init__()

        if is_relu_leaky:
            relu = nn.PReLU
        else:
            relu = nn.ReLU
        
        if norm is None:
            norm = nn.BatchNorm2d
        if pool is None:
            pool = nn.MaxPool2d
        if is_bottleneck_residual:
            resBlock = ResidualBottleneckBlock
        else:
            resBlock = ResidualBasicBlock

        # Calculations Zone
        total_divisions_needed = math.ceil(math.log2(input_size))
        n_stages = total_divisions_needed - 3 #once downsampled in initial pool, once downsampled in final residual and once downsampled at the end for final pooling

        # Stage 0
        self.intialConv = nn.Sequential(
                    nn.ReflectionPad2d(3),
                    nn.Conv2d(n_channels, starting_n_features, 7),
                    norm(starting_n_features),
                    relu() 
                    )
        self.initialPool = pool(kernel_size=3, stride=2, padding=1)
        self.initialResBlock = resBlock(starting_n_features, starting_n_features*2, relu, norm, drop_prob=drop_prob)
        self.initialAttention = AttentionModule_Adv(starting_n_features*2, starting_n_features*2, resBlock, n_stages, 0, drop_prob=droppercent4attention*drop_prob)
                
        # Stage 1 to n
        n_stage_res = []
        n_stage_attention = [] 
        in_channels = starting_n_features*2
        for i in range(n_stages):
            n_stage_res.append(resBlock(in_channels, in_channels*2, relu, norm, stride=2, drop_prob=drop_prob))
            stage_attention = []
            for _ in range(i+1):
                stage_attention.append(AttentionModule_Adv(in_channels*2, in_channels*2, resBlock, n_stages, i, drop_prob=droppercent4attention*drop_prob))
            n_stage_attention.append(nn.ModuleList(stage_attention))
            in_channels *= 2

        self.n_stage_res = nn.ModuleList(n_stage_res)
        self.n_stage_attention = nn.ModuleList(n_stage_attention)

        # Final residuals
        self.finalResBlock0 = resBlock(in_channels, in_channels*2, relu, norm, stride=2, drop_prob=drop_prob)
        in_channels *= 2
        self.finalResBlock1 = resBlock(in_channels, in_channels, relu, norm, drop_prob=drop_prob)
        self.finalResBlock2 = resBlock(in_channels, in_channels, relu, norm, drop_prob=drop_prob)

        # Final pooling
        self.finalpool = nn.Sequential(
            norm(in_channels),
            relu(),
            nn.AvgPool2d(kernel_size=2, stride=1)
        )
        self.fc = nn.Linear(in_channels,n_class)

    def forward(self, x):
        # Stage 0
        x = self.intialConv(x)
        x = self.initialPool(x)
        x = self.initialResBlock(x)
        x = self.initialAttention(x)

        # Stage 1 to N
        for i in range(len(self.n_stage_res)):
            x = self.n_stage_res[i](x)
            for j in range(len(self.n_stage_attention[i])):
                x = self.n_stage_attention[i][j](x)

        # Final Stage
        x = self.finalResBlock0(x)
        x = self.finalResBlock1(x)
        x = self.finalResBlock2(x)
        x = self.finalpool(x)

        # Fully connected layer
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

        
# m=ResNetClassifier()
# z=torch.zeros(5,1,128,128)
# o=m(z)
# print('test')
