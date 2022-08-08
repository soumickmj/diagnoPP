import os
import pickle
import pandas as pd
import json
import random
import numpy as np
from itertools import chain
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
from models.ensamble import ModelEnsamble
from torchvision import models
from torch.utils.tensorboard import SummaryWriter
from torch.optim import Adam
from apex import amp
from utils.confusion_plotting import cm_as_img_pyt
from utils.helpers import result_analyze_multilabelclassify as result_analyze

gpuID="0"
seed = 1701
batch_size = 8
num_epochs = 250
lr = 0.001
num_workers=0
repeatgray=True
useClassWeight=False
confidence=0.51
foldJSONpath = r"/run/media/soumick/Data/COVID/Data/IEEE8023_60_40_00fold3.txt"
imgfolder = r"/run/media/soumick/Data/COVID/Data/images"
metapath = r"/run/media/soumick/Data/COVID/Data/metadata.csv"
log_path = r'/run/media/soumick/Data/COVID/Data/TBLogs'
save_path = r'/run/media/soumick/Data/COVID/Data/Results'
checkpoint2load = None#r"/mnt/MEMoRIAL/MEMoRIAL_SharedStorage_M1.2+4+7/Data/PublicDSs/Covid-19/IEEE8023_COVID_CHESTXRay/Output/ResNet18_intencrop_IEEE8023V2_60_40_00fold0_noCW_repeatGray.pth.tar" #set to None for fresh

foldJSONpath = r"S:\MEMoRIAL_SharedStorage_M1.2+4+7\Data\PublicDSs\Covid-19\IEEE8023_COVID_CHESTXRay\IEEE8023V2_60_40_00fold3.txt"
imgfolder = r"S:\MEMoRIAL_SharedStorage_M1.2+4+7\Data\PublicDSs\Covid-19\IEEE8023_COVID_CHESTXRay\imagesV2"
metapath = r"S:\MEMoRIAL_SharedStorage_M1.2+4+7\Data\PublicDSs\Covid-19\IEEE8023_COVID_CHESTXRay\metadataV2.csv"
log_path = r'S:\MEMoRIAL_SharedStorage_M1.2+4+7\Data\PublicDSs\Covid-19\IEEE8023_COVID_CHESTXRay\TBLogs'
save_path = r'S:\MEMoRIAL_SharedStorage_M1.2+4+7\Data\PublicDSs\Covid-19\IEEE8023_COVID_CHESTXRay\Results'

views=["PA", "AP", "AP Supine"]
norm_type = 'intencrop'
equal_size = 512
unique_patients=False
subgroup_diseases=True
useCuda=True
usePretrainWeights=True

# #Ensamble
ensamble_checkpoints = [r"S:\MEMoRIAL_SharedStorage_M1.2+4+7\Data\PublicDSs\Covid-19\IEEE8023_COVID_CHESTXRay\Finale\DenseNet161_intencrop_IEEE8023V2_60_40_00fold3_noCW_repeatGray.pth.tar",
r"S:\MEMoRIAL_SharedStorage_M1.2+4+7\Data\PublicDSs\Covid-19\IEEE8023_COVID_CHESTXRay\Finale\IncepResV2_intencrop_IEEE8023V2_60_40_00fold3_noCW_repeatGray.pth.tar",
r"S:\MEMoRIAL_SharedStorage_M1.2+4+7\Data\PublicDSs\Covid-19\IEEE8023_COVID_CHESTXRay\Finale\InceptionV3_intencrop_IEEE8023V2_60_40_00fold3_noCW_repeatGray.pth.tar",
r"S:\MEMoRIAL_SharedStorage_M1.2+4+7\Data\PublicDSs\Covid-19\IEEE8023_COVID_CHESTXRay\Finale\ResNet18_intencrop_IEEE8023V2_60_40_00fold3_noCW_repeatGray.pth.tar",
r"S:\MEMoRIAL_SharedStorage_M1.2+4+7\Data\PublicDSs\Covid-19\IEEE8023_COVID_CHESTXRay\Finale\ResNet34_intencrop_IEEE8023V2_60_40_00fold3_noCW_repeatGray.pth.tar"]
trainID="EnsambleV1_intencrop_IEEE8023V2_60_40_00fold3_noCW_repeatGray"
# ensamble_checkpoints = None

# # #DenseNet161
# # pretrainedModel=models.densenet161
# # pretrained_modelpath=None
# # # import pretrainedmodels
# # # pretrainedModel = pretrainedmodels.polynet
# # checkpoint2load = r"S:\MEMoRIAL_SharedStorage_M1.2+4+7\Data\PublicDSs\Covid-19\IEEE8023_COVID_CHESTXRay\Finale\DenseNet161_intencrop_IEEE8023V2_60_40_00fold0_noCW_repeatGray.pth.tar"
# # trainID="DenseNet161_intencrop_IEEE8023V2_60_40_00fold0_noCW_repeatGray"

# # #IncepResV2
# # # pretrainedModel=models.densenet161
# # pretrained_modelpath=r"S:\MEMoRIAL_SharedStorage_M1.2+4+7\Data\PublicDSs\Covid-19\IEEE8023_COVID_CHESTXRay\pretrained_weights\inceptionresnetv2.pth"
# # import pretrainedmodels
# # pretrainedModel = pretrainedmodels.inceptionresnetv2
# # checkpoint2load = r"S:\MEMoRIAL_SharedStorage_M1.2+4+7\Data\PublicDSs\Covid-19\IEEE8023_COVID_CHESTXRay\Finale\IncepResV2_intencrop_IEEE8023V2_60_40_00fold0_noCW_repeatGray.pth.tar"
# # trainID="IncepResV2_intencrop_IEEE8023V2_60_40_00fold0_noCW_repeatGray"

# # #InceptionV3
# # pretrainedModel=models.inception_v3
# # pretrained_modelpath=None
# # # pretrained_modelpath=r"S:\MEMoRIAL_SharedStorage_M1.2+4+7\Data\PublicDSs\Covid-19\IEEE8023_COVID_CHESTXRay\pretrained_weights\inceptionresnetv2.pth"
# # # import pretrainedmodels
# # # pretrainedModel = pretrainedmodels.inceptionresnetv2
# # checkpoint2load = r"S:\MEMoRIAL_SharedStorage_M1.2+4+7\Data\PublicDSs\Covid-19\IEEE8023_COVID_CHESTXRay\Finale\InceptionV3_intencrop_IEEE8023V2_60_40_00fold0_noCW_repeatGray.pth.tar"
# # trainID="InceptionV3_intencrop_IEEE8023V2_60_40_00fold0_noCW_repeatGray"

# #ResNet18
# pretrainedModel=models.resnet18
# pretrained_modelpath=None
# # pretrained_modelpath=r"S:\MEMoRIAL_SharedStorage_M1.2+4+7\Data\PublicDSs\Covid-19\IEEE8023_COVID_CHESTXRay\pretrained_weights\inceptionresnetv2.pth"
# # import pretrainedmodels
# # pretrainedModel = pretrainedmodels.inceptionresnetv2
# checkpoint2load = r"S:\MEMoRIAL_SharedStorage_M1.2+4+7\Data\PublicDSs\Covid-19\IEEE8023_COVID_CHESTXRay\Finale\ResNet18_intencrop_IEEE8023V2_60_40_00fold0_noCW_repeatGray.pth.tar"
# trainID="ResNet18_intencrop_IEEE8023V2_60_40_00fold0_noCW_repeatGray"

# #ResNet34
# pretrainedModel=models.resnet34
# pretrained_modelpath=None
# # pretrained_modelpath=r"S:\MEMoRIAL_SharedStorage_M1.2+4+7\Data\PublicDSs\Covid-19\IEEE8023_COVID_CHESTXRay\pretrained_weights\inceptionresnetv2.pth"
# # import pretrainedmodels
# # pretrainedModel = pretrainedmodels.inceptionresnetv2
# checkpoint2load = r"S:\MEMoRIAL_SharedStorage_M1.2+4+7\Data\PublicDSs\Covid-19\IEEE8023_COVID_CHESTXRay\Finale\ResNet34_intencrop_IEEE8023V2_60_40_00fold0_noCW_repeatGray.pth.tar"
# trainID="ResNet34_intencrop_IEEE8023V2_60_40_00fold0_noCW_repeatGray"

n_class=14
log_freq = 10
intenper_upper=207.0 #95 percentile
intenper_lower=0.0

with open(foldJSONpath) as json_file:
    jData = json.load(json_file)
    train_patients = jData['train']
    test_patients = jData['test']

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

    if subgroup_diseases:
        losshandle = nn.BCEWithLogitsLoss
    else:
        losshandle = nn.CrossEntropyLoss

    if not useClassWeight:
        loss_fn = losshandle()
    else:
        class_weights = torch.FloatTensor(testset.class_weight).to(device)
        loss_fn = losshandle(weight=class_weights)

    if ensamble_checkpoints is None or len(ensamble_checkpoints) == 0:
        using_ensamble = False
        model = ModelWrapper(n_class=n_class, model_class=pretrainedModel, repeatgray=repeatgray, pretrained_modelpath=pretrained_modelpath, usePretrainWeights=usePretrainWeights, isMultiLabel=subgroup_diseases)
    else:
        using_ensamble = True
        sigmoidOnOut = type(loss_fn) is nn.BCEWithLogitsLoss
        model = ModelEnsamble(model_checkpoints=ensamble_checkpoints, sigmoidOnOut=sigmoidOnOut)
    model.to(device)
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=0.0001)

    if useCuda:
        opt_level = 'O1'
        model, optimizer = amp.initialize(model, optimizer, opt_level=opt_level)    

    if checkpoint2load:
        chk = torch.load(checkpoint2load)
        model.load_state_dict(chk['state_dict'])
        optimizer.load_state_dict(chk['optimizer'])
        amp.load_state_dict(chk['amp'])
    model.eval()
    runningLoss = 0.0
    runningLossCounter = 0.0
    val_loss = 0.0
    if subgroup_diseases:
        predictions = np.zeros((len(testset), n_class))
        targets = np.zeros((len(testset), n_class))
    else:
        predictions = np.zeros(len(testset))
        targets = np.zeros(len(testset))

    df = testset.csv
    with torch.no_grad():
        outs = []
        for i, data in enumerate(test_loader):
            print(i)
            images = Variable(data['img']).to(device)
            labels = Variable(data['lab']).to(device)
            if not useCuda:
                images = images.float()
            if repeatgray:
                images = images.repeat(1,3,1,1)
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            loss = round(loss.data.item(),4)

            if not using_ensamble and type(loss_fn) is nn.BCEWithLogitsLoss:
                outputs = torch.sigmoid(outputs)
            
            y_true = labels.detach().cpu().numpy()
            if subgroup_diseases:
                y_pred = (outputs.detach().cpu().numpy()>confidence).astype(int)
                predictions[i * batch_size:(i+1) * batch_size, :] = y_pred
                targets[i * batch_size:(i+1) * batch_size, :] = y_true
            else:
                y_pred = np.argmax(outputs.detach().cpu().numpy(), axis=1)
                predictions[i * batch_size:(i+1) * batch_size] = y_pred
                targets[i * batch_size:(i+1) * batch_size] = y_true

            for r in range(len(data['idx'])):
                idx = data['idx'][r].item()
                outs.append([idx, testset.csv.patientid.iloc[idx], testset.csv.filename.iloc[idx], testset.csv.finding.iloc[idx]] + list(y_pred[r]))                   

            val_loss += loss
            runningLoss += loss
            runningLossCounter += 1
    avg_loss = val_loss/len(test_loader)
    metrics = result_analyze(targets, predictions, subgroup_diseases, list(range(n_class)))
    df = pd.DataFrame(outs, columns=['ID', 'PatientID', 'File', 'Finding'] + testset.pathologies)
    df.to_excel(os.path.join(save_path, trainID+'_test.xlsx'))
    with open(os.path.join(save_path, trainID+'_test.pkl'), 'wb') as outfile:
        pickle.dump(metrics, outfile, protocol=pickle.HIGHEST_PROTOCOL)
        # cm = cm_as_img_pyt(targets, predictions, list(range(n_class)), trainset.pathologies[:n_class],
        #                      normalize=False)
        # tb_writer.add_image('Val/ConfusionMatrix', cm, 0)