import os
import gc
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
from models.modelwrapper import ModelWrapper, ModelWrapper4Pretrained
from models.ensamble import ModelEnsamble
from torchvision import models
from torch.utils.tensorboard import SummaryWriter
from torch.optim import Adam
from apex import amp
from utils.confusion_plotting import cm_as_img_pyt
from utils.helpers import result_analyze_multilabelclassify as result_analyze
from utils.captum.visualizer import visualize_model, visualize_model_multilabel
from utils.visualCNN.visualizer import visualize_model as visualize_model_visualCNN
import utils.visualCNN.generate_class_specific_samples as genClsSample

gpuID="0"
seed = 1701
batch_size = 1
num_workers=0
repeatgray=True
useClassWeight=False
confidence=0.51

foldJSONpath = r"/run/media/soumick/Data/COVID/Data/IEEE8023V2_60_40_00fold0.txt"
imgfolder = r"/run/media/soumick/Data/COVID/Data/imagesV2"
metapath = r"/run/media/soumick/Data/COVID/Data/metadataV2.csv"
log_path = r'/run/media/soumick/Data/COVID/Data/TBLogs'
save_path = r'/run/media/soumick/Data/COVID/Data/Output'
explain_out = r'/run/media/soumick/Data/COVID/Data/Explain'

foldJSONpath = r"S:\MEMoRIAL_SharedStorage_M1.2+4+7\Data\PublicDSs\Covid-19\IEEE8023_COVID_CHESTXRay\IEEE8023V2_60_40_00fold0.txt"
imgfolder = r"S:\MEMoRIAL_SharedStorage_M1.2+4+7\Data\PublicDSs\Covid-19\IEEE8023_COVID_CHESTXRay\imagesV2"
metapath = r"S:\MEMoRIAL_SharedStorage_M1.2+4+7\Data\PublicDSs\Covid-19\IEEE8023_COVID_CHESTXRay\metadataV2.csv"
log_path = r'S:\MEMoRIAL_SharedStorage_M1.2+4+7\Data\PublicDSs\Covid-19\IEEE8023_COVID_CHESTXRay\TBLogs'
save_path = r'S:\MEMoRIAL_SharedStorage_M1.2+4+7\Data\PublicDSs\Covid-19\IEEE8023_COVID_CHESTXRay\Output'
explain_out = r'S:\MEMoRIAL_SharedStorage_M1.2+4+7\Data\PublicDSs\Covid-19\IEEE8023_COVID_CHESTXRay\Explain'

foldJSONpath = r"/nfs1/schatter/Covid19/IEEE8023/IEEE8023V2_60_40_00fold0.txt"
imgfolder = r"/nfs1/schatter/Covid19/IEEE8023/imagesV2"
metapath = r"/nfs1/schatter/Covid19/IEEE8023/metadataV2.csv"
log_path = r'/nfs1/schatter/Covid19/IEEE8023/TBLogs'
save_path = r'/nfs1/schatter/Covid19/IEEE8023/Output'
explain_out = r'/nfs1/schatter/Covid19/IEEE8023/Explain'

#Ensamble
model_checkpoints = [r"S:\MEMoRIAL_SharedStorage_M1.2+4+7\Data\PublicDSs\Covid-19\IEEE8023_COVID_CHESTXRay\Finale\DenseNet161_intencrop_IEEE8023V2_60_40_00fold0_noCW_repeatGray.pth.tar",
# r"S:\MEMoRIAL_SharedStorage_M1.2+4+7\Data\PublicDSs\Covid-19\IEEE8023_COVID_CHESTXRay\Finale\IncepResV2_intencrop_IEEE8023V2_60_40_00fold0_noCW_repeatGray.pth.tar",
# r"S:\MEMoRIAL_SharedStorage_M1.2+4+7\Data\PublicDSs\Covid-19\IEEE8023_COVID_CHESTXRay\Finale\InceptionV3_intencrop_IEEE8023V2_60_40_00fold0_noCW_repeatGray.pth.tar",
# r"S:\MEMoRIAL_SharedStorage_M1.2+4+7\Data\PublicDSs\Covid-19\IEEE8023_COVID_CHESTXRay\Finale\ResNet18_intencrop_IEEE8023V2_60_40_00fold0_noCW_repeatGray.pth.tar",
r"S:\MEMoRIAL_SharedStorage_M1.2+4+7\Data\PublicDSs\Covid-19\IEEE8023_COVID_CHESTXRay\Finale\ResNet34_intencrop_IEEE8023V2_60_40_00fold0_noCW_repeatGray.pth.tar"]

#Ensamble
model_checkpoints = [r"/nfs1/schatter/Covid19/IEEE8023/Finale/DenseNet161_intencrop_IEEE8023V2_60_40_00fold0_noCW_repeatGray.pth.tar",
r"/nfs1/schatter/Covid19/IEEE8023/Finale/IncepResV2_intencrop_IEEE8023V2_60_40_00fold0_noCW_repeatGray.pth.tar",
r"/nfs1/schatter/Covid19/IEEE8023/Finale/InceptionV3_intencrop_IEEE8023V2_60_40_00fold0_noCW_repeatGray.pth.tar",
r"/nfs1/schatter/Covid19/IEEE8023/Finale/ResNet18_intencrop_IEEE8023V2_60_40_00fold0_noCW_repeatGray.pth.tar",
r"/nfs1/schatter/Covid19/IEEE8023/Finale/ResNet34_intencrop_IEEE8023V2_60_40_00fold0_noCW_repeatGray.pth.tar"]

use_ensamble = False

views=["PA", "AP", "AP Supine"]
norm_type = 'intencrop'
equal_size = 512
unique_patients=False
subgroup_diseases=True
useCuda=True
test_patients=None

#####Visualization parameters
file2run="16745_3_1.png"
##Captum
layerID=None #used for guidedGradCam. If None, then guidedGradCam won't be executed
layer_name='' #used for guidedGradCam
feature_mask=None #used in featureAblation and shapleyValues
plt_fig_axis=None
show_plt=False
show_original=True

#Visual CNN
generateClassSamples=False
iterations4clsGen=150
layerID4visualCNN = -1
filter_position = 0
#####

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
    explain_out = os.path.join(explain_out, '0 Specific Subjects')
    device = torch.device("cuda:0" if torch.cuda.is_available() and useCuda else "cpu")    
    model_names = []
    for checkpoint in model_checkpoints:
        model_names.append(os.path.basename(checkpoint).split('.')[0])
    

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

    if subgroup_diseases:
        losshandle = nn.BCEWithLogitsLoss
    else:
        losshandle = nn.CrossEntropyLoss

    if not useClassWeight:
        loss_fn = losshandle()
    else:
        class_weights = torch.FloatTensor(testset.class_weight).to(device)
        loss_fn = losshandle(weight=class_weights)

    models = []
    for checkpoint in model_checkpoints:
        chk = torch.load(checkpoint, map_location="cpu")
        model=ModelWrapper4Pretrained(chk, nn.Sigmoid) #TODO: make this sigmoid dynamic
        model.to(device)
        model.eval()
        models.append(model)

    if use_ensamble:
        sigmoidOnOut = type(loss_fn) is nn.BCEWithLogitsLoss
        model = ModelEnsamble(model_checkpoints=model_checkpoints, sigmoidOnOut=sigmoidOnOut)
        model.to(device)
        model.eval()
        models.append(model)    
        model_names.append('Ensamble')    

    if generateClassSamples:        
        for i in range(n_class):
            genClsSample.execute(root_model, i, os.path.join(explain_out, 'visualCNN', 'ImgGeneration'), iterations4clsGen)

    gc.collect()
    torch.cuda.empty_cache()

    runningLoss = 0.0
    runningLossCounter = 0.0
    val_loss = 0.0
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            idx = data['idx'][0].item()
            imgid = testset.csv.filename.iloc[idx]
            img_path = os.path.join(testset.imgpath, imgid)
            if file2run and imgid != file2run:
                continue
            
            images = Variable(data['img']).to(device).float()
            labels = Variable(data['lab']).to(device).int()
            if repeatgray:
                images = images.repeat(1,3,1,1)            
            target = labels.detach().cpu().numpy()
            images.requires_grad=True
            for i in range(len(models)):
                # outputs = models[i](images)
                # prediction = (outputs.detach().cpu().numpy()>confidence).astype(int)
                model_name = model_names[i]
                explain_out_model = os.path.join(explain_out, model_name)
                explain_out_img_cap = os.path.join(explain_out_model, imgid.split('.')[0], 'captum')
                # explain_out_img_viscnn = os.path.join(explain_out, imgid.split('.')[0], 'visualCNN')
                os.makedirs(explain_out_img_cap, exist_ok=True)
                # os.makedirs(explain_out_img_viscnn, exist_ok=True)
                if subgroup_diseases:
                    visualize_model_multilabel(models[i], images, target, layerID=layerID, layer_name=layer_name, feature_mask=labels, plt_fig_axis=plt_fig_axis,
                                            show_plt=show_plt, show_original=show_original, class_names=testset.pathologies, explain_out_img=explain_out_img_cap)
                else:
                    visualize_model(models[i], images, target, layerID=layerID, layer_name=layer_name, feature_mask=feature_mask, 
                                    plt_fig_axis=plt_fig_axis, show_plt=show_plt, show_original=show_original) #TODO: untested
                gc.collect()
                torch.cuda.empty_cache()
            
            # visualize_model_visualCNN(model, layerID=layerID4visualCNN, filter_position=filter_position, im_path=img_path, classID=2, inputs=images, file_name_to_export=explain_out_img_viscnn) #TODO: untested