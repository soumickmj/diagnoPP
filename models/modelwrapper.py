import torch
import torch.nn as nn
from torchvision import models
import pretrainedmodels #get it from https://github.com/Cadene/pretrained-models.pytorch.git

class ModelWrapper(nn.Module):

    def __init__(self, in_channels=1, n_class=14, model_class=models.resnet18, repeatgray=True, pretrained_modelpath=None, usePretrainWeights=True, isMultiLabel=True):
        super(ModelWrapper, self).__init__()

        if usePretrainWeights:
            if 'pretrainedmodels.models' in model_class.__module__:
                # self.net = model_class(num_classes=1000, pretrained='imagenet')
                self.net = model_class(num_classes=1001,pretrained=None)
                self.net.load_state_dict(torch.load(pretrained_modelpath))
                noPYTModel = True
            else:
                self.net = model_class(pretrained=True)
                noPYTModel = False

            if noPYTModel:
                num_ftrs = self.net.last_linear.in_features
                self.net.last_linear = nn.Linear(num_ftrs, n_class)
            elif type(self.net) is models.densenet.DenseNet:
                num_ftrs = self.net.classifier.in_features
                self.net.classifier = nn.Linear(num_ftrs, n_class)
            elif type(self.net) is models.vgg.VGG or type(self.net) is models.AlexNet:
                num_ftrs = self.net.classifier[-1].in_features
                self.net.classifier[-1] = nn.Linear(num_ftrs, n_class)
            else:
                num_ftrs = self.net.fc.in_features
                self.net.fc = nn.Linear(num_ftrs, n_class)

            if model_class is models.inception_v3:
                self.net.AuxLogits.fc = nn.Linear(self.net.AuxLogits.fc.in_features, n_class)
        else:
            print('Initializing Fresh Network')
            self.net = model_class(num_classes=n_class,pretrained=None)

        if in_channels != 3 and not repeatgray:
            self.net = nn.Sequential(
                nn.Conv2d(in_channels, 3, kernel_size=3),
                self.net
            )       

        if not isMultiLabel and model_class is not models.inception_v3:
            self.net = nn.Sequential(
                self.net,
                nn.Softmax()
            ) 
        elif not isMultiLabel and model_class is models.inception_v3:
            self.net.fc = nn.Sequential(
                self.net.fc,
                nn.Softmax()
            ) 
            self.net.AuxLogits.fc = nn.Sequential(
                self.net.AuxLogits.fc,
                nn.Softmax()
            ) 

    def get_device(self):
        return next(self.parameters()).device

    def forward(self, x):
        return self.net(x)

class ModelWrapper4Pretrained(nn.Module):
    def __init__(self, chk_pnt, act=None):
        super(ModelWrapper4Pretrained, self).__init__()

        self.net = chk_pnt['model']
        self.net.load_state_dict(chk_pnt['state_dict'])
        if act is not None:
            self.net = nn.Sequential(
                self.net,
                act()
            )

    def get_device(self):
        return next(self.parameters()).device

    def forward(self, x):
        return self.net(x)
