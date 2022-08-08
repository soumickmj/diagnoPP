import os
import pickle
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
from torchvision import models
from torch.utils.tensorboard import SummaryWriter
from torch.optim import Adam
from apex import amp
from utils.confusion_plotting import cm_as_img_pyt
from utils.helpers import result_analyze_multilabelclassify as result_analyze

gpuID="1"
seed = 1701
batch_size = 8
num_epochs = 250
lr = 0.001
num_workers=4
repeatgray=True
useClassWeight=False
confidence=0.51
foldJSONpath = r"/run/media/soumick/Data/COVID/Data/IEEE8023V2_60_40_00fold2.txt"
imgfolder = r"/run/media/soumick/Data/COVID/Data/imagesV2"
metapath = r"/run/media/soumick/Data/COVID/Data/metadataV2.csv"
log_path = r'/run/media/soumick/Data/COVID/Data/TBLogs'
save_path = r'/run/media/soumick/Data/COVID/Data/Output'
checkpoint2load = None#r"/mnt/MEMoRIAL/MEMoRIAL_SharedStorage_M1.2+4+7/Data/PublicDSs/Covid-19/IEEE8023_COVID_CHESTXRay/Output/ResNet18_intencrop_IEEE8023V2_60_40_00fold0_noCW_repeatGray.pth.tar" #set to None for fresh
views=["PA", "AP", "AP Supine"]
norm_type = 'intencrop'
equal_size = 512
unique_patients=False
subgroup_diseases=True
useCuda=True

usePretrainWeights=True
# pretrainedModel=models.resnet18
pretrained_modelpath=r"/mnt/MEMoRIAL/MEMoRIAL_SharedStorage_M1.2+4+7/Data/PublicDSs/Covid-19/IEEE8023_COVID_CHESTXRay/pretrained_weights/inceptionresnetv2.pth"
import pretrainedmodels
pretrainedModel = pretrainedmodels.inceptionresnetv2

trainID="IncepResV2_intencrop_IEEE8023V2_60_40_00fold2_noCW_repeatGray"
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
    tb_writer = SummaryWriter(log_dir = os.path.join(log_path,trainID))

    trainset = COVID19_Dataset(imgpath=imgfolder, 
                    csvpath=metapath, 
                    views=views,
                    norm_type = norm_type, #'max1' or 'normalize1024' (between -1024, +1024) or 'minmax (between -1, +1)' or 'minmaxPOS (between 0, +1)'
                    equal_size = equal_size,
                    transform=None, 
                    data_aug=None,
                    seed=seed,
                    unique_patients=unique_patients,
                    subgroup_diseases=subgroup_diseases,
                    patientIDs=train_patients,
                    intenper_upper=intenper_upper,
                    intenper_lower=intenper_lower)
    train_loader = DataLoader(dataset=trainset,batch_size=batch_size,shuffle=True, num_workers=num_workers)

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

    model = ModelWrapper(n_class=n_class, model_class=pretrainedModel, repeatgray=repeatgray, pretrained_modelpath=pretrained_modelpath, usePretrainWeights=usePretrainWeights, isMultiLabel=subgroup_diseases)
    model.to(device)
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=0.0001)

    if subgroup_diseases:
        losshandle = nn.BCEWithLogitsLoss
    else:
        losshandle = nn.CrossEntropyLoss

    if not useClassWeight:
        loss_fn = losshandle()
    else:
        class_weights = torch.FloatTensor(trainset.class_weight).to(device)
        loss_fn = losshandle(weight=class_weights)


    opt_level = 'O1'
    model, optimizer = amp.initialize(model, optimizer, opt_level=opt_level)    

    if checkpoint2load:
        chk = torch.load(checkpoint2load)
        model.load_state_dict(chk['state_dict'])
        optimizer.load_state_dict(chk['optimizer'])
        amp.load_state_dict(chk['amp'])
        start_epoch = chk['epoch'] + 1
        best_loss = chk['loss'] 
    else:
        start_epoch = 0
        best_loss = 100000

    for epoch in range(start_epoch, num_epochs):
        #Train
        model.train()
        runningLoss = 0.0
        runningLossCounter = 0.0
        train_loss = 0.0
        for i, data in enumerate(train_loader):
            images = Variable(data['img']).to(device)
            labels = Variable(data['lab']).to(device)
            if repeatgray:
                images = images.repeat(1,3,1,1)
            optimizer.zero_grad()
            if pretrainedModel is models.inception_v3:
                outputs, aux_outputs = model(images)
                loss1 = loss_fn(outputs, labels)
                loss2 = loss_fn(aux_outputs, labels)
                loss = loss1 + 0.4*loss2
            else:
                outputs = model(images)
                loss = loss_fn(outputs, labels)
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
            optimizer.step()
            loss = round(loss.data.item(),4)
            train_loss += loss
            runningLoss += loss
            runningLossCounter += 1
            print('[%d/%d][%d/%d] Train Loss: %.4f' % ((epoch+1), num_epochs, i, len(train_loader), loss))
            #For tensorboard
            if i % log_freq == 0:
                niter = epoch*len(train_loader)+i
                tb_writer.add_scalar('Train/Loss', runningLoss/runningLossCounter, niter)
                runningLoss = 0.0
                runningLossCounter = 0.0
        checkpoint = {
            'model': model,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'amp': amp.state_dict()
        }
        torch.save(checkpoint, os.path.join(save_path, trainID+".pth.tar"))
        tb_writer.add_scalar('Train/AvgLossEpoch', train_loss/len(train_loader), epoch)

        #Validate
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
        with torch.no_grad():
            for i, data in enumerate(test_loader):
                images = Variable(data['img']).to(device)
                labels = Variable(data['lab']).to(device)
                if repeatgray:
                    images = images.repeat(1,3,1,1)
                outputs = model(images)
                loss = loss_fn(outputs, labels)
                loss = round(loss.data.item(),4)

                if type(loss_fn) is nn.BCEWithLogitsLoss:
                    outputs = torch.sigmoid(outputs)
                
                if subgroup_diseases:
                    predictions[i * batch_size:(i+1) * batch_size, :] = (outputs.detach().cpu().numpy()>confidence).astype(int)
                    targets[i * batch_size:(i+1) * batch_size, :] = labels.detach().cpu().numpy()
                else:
                    predictions[i * batch_size:(i+1) * batch_size] = np.argmax(outputs.detach().cpu().numpy(), axis=1)
                    targets[i * batch_size:(i+1) * batch_size] = labels.detach().cpu().numpy()

                val_loss += loss
                runningLoss += loss
                runningLossCounter += 1
                print('[%d/%d][%d/%d] Val Loss: %.4f' % ((epoch+1), num_epochs, i, len(test_loader), loss))
                #For tensorboard
                if i % log_freq == 0:
                    niter = epoch*len(test_loader)+i
                    tb_writer.add_scalar('Val/Loss', runningLoss/runningLossCounter, niter)
                    runningLoss = 0.0
                    runningLossCounter = 0.0            
        avg_loss = val_loss/len(test_loader)
        tb_writer.add_scalar('Val/AvgLossEpoch', avg_loss, epoch)
        if best_loss > avg_loss:
            print('best encountered')
            best_loss = avg_loss
            checkpoint = {
                'model': model,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'amp': amp.state_dict(),
                'epoch': epoch,
                'loss': best_loss
            }
            torch.save(checkpoint, os.path.join(save_path, trainID+"_best.pth.tar"))
        metrics = result_analyze(targets, predictions, subgroup_diseases, list(range(n_class)))
        metrics['epoch'] = epoch
        with open(os.path.join(save_path, trainID+str(epoch)+'.pkl'), 'wb') as outfile:
            pickle.dump(metrics, outfile, protocol=pickle.HIGHEST_PROTOCOL)
        # cm = cm_as_img_pyt(targets, predictions, list(range(n_class)), trainset.pathologies[:n_class],
        #                      normalize=False)
        # tb_writer.add_image('Val/ConfusionMatrix', cm, 0)