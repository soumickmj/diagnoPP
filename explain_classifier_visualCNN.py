import os
import pickle
import json
import random
import numpy as np
from itertools import chain
from sklearn.metrics import multilabel_confusion_matrix
import torch
import torch.nn as nn
import torchvision
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
# import torchxrayvision as xrv
from utils.dataloader_pyt_CXR import COVID19_Dataset
from tqdm import tqdm
import sys
import matplotlib.pyplot as plt
from models.modelwrapper import ModelWrapper
from torchvision import models
from torch.utils.tensorboard import SummaryWriter
from torch.optim import Adam
from apex import amp
from utils.confusion_plotting import cm_as_img_pyt
from utils.helpers import result_analyze_multilabelclassify as result_analyze
from utils.captum.visualizer import visualize_model
from utils.visualCNN.calller import call_visualCNN

gpuID="1"
seed = 1701
batch_size = 8
num_epochs = 250
lr = 0.001
num_workers=4
repeatgray=True
useClassWeight=False
confidence=0.51
foldJSONpath = r"S:\MEMoRIAL_SharedStorage_M1.2+4+7\Data\PublicDSs\Covid-19\IEEE8023_COVID_CHESTXRay\IEEE8023V2_60_40_00fold0.txt"
imgfolder = r"S:\MEMoRIAL_SharedStorage_M1.2+4+7\Data\PublicDSs\Covid-19\IEEE8023_COVID_CHESTXRay\imagesV2"
metapath = r"S:\MEMoRIAL_SharedStorage_M1.2+4+7\Data\PublicDSs\Covid-19\IEEE8023_COVID_CHESTXRay\metadataV2.csv"
log_path = r'S:\MEMoRIAL_SharedStorage_M1.2+4+7\Data\PublicDSs\Covid-19\IEEE8023_COVID_CHESTXRay\TBLogs'
save_path = r'S:\MEMoRIAL_SharedStorage_M1.2+4+7\Data\PublicDSs\Covid-19\IEEE8023_COVID_CHESTXRay\Output'
checkpoint2load = r"S:\MEMoRIAL_SharedStorage_M1.2+4+7\Data\PublicDSs\Covid-19\IEEE8023_COVID_CHESTXRay\Output\IncepResV2_intencrop_IEEE8023V2_60_40_00fold0_noCW_repeatGray_best.pth.tar" #set to None for fresh
views=["PA", "AP", "AP Supine"]
norm_type = 'intencrop'
equal_size = 512
unique_patients=False
subgroup_diseases=True
useCuda=True
test_patients=None

#####Visualization parameters
##Captum
layerID=None #used for guidedGradCam. If None, then guidedGradCam won't be executed
layer_name='' #used for guidedGradCam
feature_mask=None #used in featureAblation and shapleyValues
plt_fig_axis=None
show_plt=True
show_original=True

#
filter_position = 0

pretrainedModel=models.resnext50_32x4d
# import pretrainedmodels
# pretrainedModel = pretrainedmodels.polynet

trainID="ResNext50_intencrop_IEEE8023V2_60_40_00fold0_noCW_repeatGray"
n_class=14
log_freq = 10
intenper_upper=207.0 #95 percentile
intenper_lower=0.0

os.environ["CUDA_VISIBLE_DEVICES"] = gpuID
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

if __name__ == "__main__" :
    device = torch.device("cuda:0" if torch.cuda.is_available() and useCuda else "cpu")    
    testset = COVID19_Dataset(imgpath=imgfolder, 
                    csvpath=metapath, 
                    views=views,
                    norm_type = norm_type, #'max1' or 'normalize1024' (between -1024, +1024) or 'minmax (between -1, +1)' or 'minmaxPOS (between 0, +1)'
                    equal_size = equal_size,
                    transform=None, 
                    data_aug=None,
                    seed=seed,
                    unique_patients=unique_patients,
                    subgroup_diseases=subgroup_diseases,
                    patientIDs=test_patients,
                    intenper_upper=intenper_upper,
                    intenper_lower=intenper_lower)
    test_loader = DataLoader(dataset=testset,batch_size=batch_size,shuffle=False, num_workers=num_workers)

    model = ModelWrapper(n_class=n_class, model_class=pretrainedModel, repeatgray=repeatgray)
    model.to(device)    
    chk = torch.load(checkpoint2load)
    model.load_state_dict(chk['state_dict'])
    loss_fn = nn.BCEWithLogitsLoss()

    model.eval()
    runningLoss = 0.0
    runningLossCounter = 0.0
    val_loss = 0.0
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            images = Variable(data['img']).to(device)
            labels = Variable(data['lab']).to(device)
            if type(loss_fn) is nn.BCEWithLogitsLoss:
                outputs = torch.sigmoid(outputs)

            prediction = (outputs.detach().cpu().numpy()>confidence).astype(int)
            target = labels.detach().cpu().numpy()

            visualize_model(model, images, target, layerID=layerID, layer_name=layer_name, feature_mask=feature_mask, 
                            plt_fig_axis=plt_fig_axis, show_plt=show_plt, show_original=show_original)
            call_visualCNN(model, layerID, filter_position, im_path, classID, inputs, file_name_to_export)