import torch
import torch.nn as nn
from torchvision import models
import pretrainedmodels

class ModelEnsamble(nn.Module):

    def __init__(self, model_checkpoints=[], sigmoidOnOut=True):
        super(ModelEnsamble, self).__init__()
        self.sigmoidOnOut = sigmoidOnOut
        self.models = nn.ModuleList()
        for checkpoint in model_checkpoints:
            chk = torch.load(checkpoint)
            model=chk['model']
            model.load_state_dict(chk['state_dict'])
            self.models.append(model)

    def forward(self, x):
        result = None
        for model in self.models:
            out = model(x)
            if self.sigmoidOnOut:
                out = torch.sigmoid(out)
            if result is None:
                result = out
            else:
                result += out
        return result/len(self.models)