import json
import os
import gc
import pickle
# import json
import random
import numpy as np
# from itertools import chain
# from sklearn.metrics import multilabel_confusion_matrix
import torch
import torch.nn as nn
# import torchvision
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

from utils.fetch_activations import Hooker
# from torch.utils.tensorboard import SummaryWriter
# from torch.optim import Adam
# from apex import amp
# from utils.confusion_plotting import cm_as_img_pyt
# from utils.helpers import result_analyze_multilabelclassify as result_analyze
# from utils.captum.visualizer import visualize_model, visualize_model_multilabel
# from utils.visualCNN.visualizer import visualize_model as visualize_model_visualCNN
# import utils.visualCNN.generate_class_specific_samples as genClsSample

print(torch.cuda.device_count())

gpuID="0"
seed = 1701
batch_size = 1
num_workers=0
repeatgray=True
useClassWeight=False
confidence=0.51

# foldJSONpath = r"/run/media/soumick/Data/COVID/Data/IEEE8023V2_60_40_00fold0.txt"
# imgfolder = r"/run/media/soumick/Data/COVID/Data/imagesV2"
# metapath = r"/run/media/soumick/Data/COVID/Data/metadataV2.csv"
# log_path = r'/run/media/soumick/Data/COVID/Data/TBLogs'
# save_path = r'/run/media/soumick/Data/COVID/Data/Output'
# explain_out = r'/run/media/soumick/Data/COVID/Data/Explain'

# foldJSONpath = r"S:\MEMoRIAL_SharedStorage_M1.2+4+7\Data\PublicDSs\Covid-19\IEEE8023_COVID_CHESTXRay\IEEE8023V2_60_40_00fold0.txt"
# imgfolder = r"S:\MEMoRIAL_SharedStorage_M1.2+4+7\Data\PublicDSs\Covid-19\IEEE8023_COVID_CHESTXRay\imagesV2"
# metapath = r"S:\MEMoRIAL_SharedStorage_M1.2+4+7\Data\PublicDSs\Covid-19\IEEE8023_COVID_CHESTXRay\metadataV2.csv"
# log_path = r'S:\MEMoRIAL_SharedStorage_M1.2+4+7\Data\PublicDSs\Covid-19\IEEE8023_COVID_CHESTXRay\TBLogs'
# save_path = r'S:\MEMoRIAL_SharedStorage_M1.2+4+7\Data\PublicDSs\Covid-19\IEEE8023_COVID_CHESTXRay\Output'
# explain_out = r'S:\MEMoRIAL_SharedStorage_M1.2+4+7\Data\PublicDSs\Covid-19\IEEE8023_COVID_CHESTXRay\Explain'

# foldJSONpath = r"/nfs1/schatter/Covid19/IEEE8023/IEEE8023V2_60_40_00fold0.txt"
# imgfolder = r"/nfs1/schatter/Covid19/IEEE8023/imagesV2"
# metapath = r"/nfs1/schatter/Covid19/IEEE8023/metadataV2.csv"
# log_path = r'/nfs1/schatter/Covid19/IEEE8023/TBLogs'
# save_path = r'/nfs1/schatter/Covid19/IEEE8023/Output'
# explain_out = r'/nfs1/schatter/Covid19/IEEE8023/Explain2'

#######FCM Scratch
foldJSONpath = r"/scratch/schatter/ExplainCOVID/IEEE8023V2_60_40_00fold0.txt"
# imgfolder = r"/scratch/schatter/ExplainCOVID/imagesShortlisted"#imagesV2"
imgfolder = r"/scratch/schatter/ExplainCOVID/imagesV2"#imagesV2"
metapath = r"/scratch/schatter/ExplainCOVID/metadataV2.csv"
# log_path = r'/nfs1/schatter/Covid19/IEEE8023/TBLogs'
# save_path = r'/nfs1/schatter/Covid19/IEEE8023/Output'
explain_out = r'/scratch/schatter/ExplainCOVID/Activations/AllTestFold0'
ONNX_out = r'/scratch/schatter/ExplainCOVID/WeightsONNX'

# #Ensamble
# model_checkpoints = [r"S:\MEMoRIAL_SharedStorage_M1.2+4+7\Data\PublicDSs\Covid-19\IEEE8023_COVID_CHESTXRay\Finale\DenseNet161_intencrop_IEEE8023V2_60_40_00fold0_noCW_repeatGray.pth.tar",
# # r"S:\MEMoRIAL_SharedStorage_M1.2+4+7\Data\PublicDSs\Covid-19\IEEE8023_COVID_CHESTXRay\Finale\IncepResV2_intencrop_IEEE8023V2_60_40_00fold0_noCW_repeatGray.pth.tar",
# # r"S:\MEMoRIAL_SharedStorage_M1.2+4+7\Data\PublicDSs\Covid-19\IEEE8023_COVID_CHESTXRay\Finale\InceptionV3_intencrop_IEEE8023V2_60_40_00fold0_noCW_repeatGray.pth.tar",
# # r"S:\MEMoRIAL_SharedStorage_M1.2+4+7\Data\PublicDSs\Covid-19\IEEE8023_COVID_CHESTXRay\Finale\ResNet18_intencrop_IEEE8023V2_60_40_00fold0_noCW_repeatGray.pth.tar",
# r"S:\MEMoRIAL_SharedStorage_M1.2+4+7\Data\PublicDSs\Covid-19\IEEE8023_COVID_CHESTXRay\Finale\ResNet34_intencrop_IEEE8023V2_60_40_00fold0_noCW_repeatGray.pth.tar"]

# #Ensamble
# model_checkpoints = [r"/nfs1/schatter/Covid19/IEEE8023/Finale/DenseNet161_intencrop_IEEE8023V2_60_40_00fold0_noCW_repeatGray.pth.tar",
# r"/nfs1/schatter/Covid19/IEEE8023/Finale/IncepResV2_intencrop_IEEE8023V2_60_40_00fold0_noCW_repeatGray.pth.tar",
# r"/nfs1/schatter/Covid19/IEEE8023/Finale/InceptionV3_intencrop_IEEE8023V2_60_40_00fold0_noCW_repeatGray.pth.tar",
# r"/nfs1/schatter/Covid19/IEEE8023/Finale/ResNet18_intencrop_IEEE8023V2_60_40_00fold0_noCW_repeatGray.pth.tar",
# r"/nfs1/schatter/Covid19/IEEE8023/Finale/ResNet34_intencrop_IEEE8023V2_60_40_00fold0_noCW_repeatGray.pth.tar"]

#Ensamble - checkpoints in FCM scratch
model_checkpoints = [
    r"/scratch/schatter/ExplainCOVID/Weights/DenseNet161_intencrop_IEEE8023V2_60_40_00fold0_noCW_repeatGray.pth.tar",
    r"/scratch/schatter/ExplainCOVID/Weights/IncepResV2_intencrop_IEEE8023V2_60_40_00fold0_noCW_repeatGray.pth.tar",
    r"/scratch/schatter/ExplainCOVID/Weights/InceptionV3_intencrop_IEEE8023V2_60_40_00fold0_noCW_repeatGray.pth.tar",
    r"/scratch/schatter/ExplainCOVID/Weights/ResNet18_intencrop_IEEE8023V2_60_40_00fold0_noCW_repeatGray.pth.tar",
    r"/scratch/schatter/ExplainCOVID/Weights/ResNet34_intencrop_IEEE8023V2_60_40_00fold0_noCW_repeatGray.pth.tar"
]

use_ensamble = False

views=["PA", "AP", "AP Supine"]
norm_type = 'intencrop'
equal_size = 512
unique_patients=False
subgroup_diseases=True
useCuda=True
test_patients=None

#####Visualization parameters
# file2run=[
#     "covid-19-caso-70-1-PA.jpg", #Fig. 4,5,6 (the big figures comparing the different techniques)
#     "9C34AF49-E589-44D5-92D3-168B3B04E4A6.jpeg" #Fig. 9 (model and pathology comparison)
# ]
file2run = None

with open(foldJSONpath) as json_file:
    jData = json.load(json_file)
    test_patients = jData['test']

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
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False

if __name__ == "__main__" :
    explain_out = os.path.join(explain_out, '0 Specific Subjects')
    device = torch.device("cuda:0" if torch.cuda.is_available() and useCuda else "cpu")  
    model_names = []
    for checkpoint in model_checkpoints:
        model_names.append(os.path.basename(checkpoint).split('.')[0])
    
    if len(sys.argv) > 1:
        modelID = int(sys.argv[1])
        print("Only explaining Model ID : "+str(modelID) +" "+ model_names[modelID])
        model_names = [model_names[modelID]]
        model_checkpoints = [model_checkpoints[modelID]]
    print(str(len(model_checkpoints))+" model(s) explaining")

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

    # if subgroup_diseases:
    #     losshandle = nn.BCEWithLogitsLoss
    # else:
    #     losshandle = nn.CrossEntropyLoss

    # if not useClassWeight:
    #     loss_fn = losshandle()
    # else:
    #     class_weights = torch.FloatTensor(testset.class_weight).to(device)
    #     loss_fn = losshandle(weight=class_weights)

    # models = []
    # for checkpoint in model_checkpoints:
    #     chk = torch.load(checkpoint, map_location="cpu")
    #     model=ModelWrapper4Pretrained(chk, nn.Sigmoid) #TODO: make this sigmoid dynamic
    #     model.to(device)
    #     model.eval()
    #     models.append(model)

    # if use_ensamble:
    #     sigmoidOnOut = type(loss_fn) is nn.BCEWithLogitsLoss
    #     model = ModelEnsamble(model_checkpoints=model_checkpoints, sigmoidOnOut=sigmoidOnOut)
    #     model.to(device)
    #     model.eval()
    #     models.append(model)    
    #     model_names.append('Ensamble')    

    # if generateClassSamples:        
    #     for i in range(n_class):
    #         genClsSample.execute(root_model, i, os.path.join(explain_out, 'visualCNN', 'ImgGeneration'), iterations4clsGen)

    # gc.collect()
    # torch.cuda.empty_cache()

    runningLoss = 0.0
    runningLossCounter = 0.0
    val_loss = 0.0
    with torch.no_grad():
        for i, data in tqdm(enumerate(test_loader)):
            idx = data['idx'][0].item()
            imgid = testset.csv.filename.iloc[idx]
            img_path = os.path.join(testset.imgpath, imgid)
            if (file2run is not None) and (imgid not in file2run):
                continue
            images = Variable(data['img']).to(device).float()
            labels = Variable(data['lab']).to(device).int()
            if repeatgray:
                images = images.repeat(1,3,1,1)            
            target = labels.detach().cpu().numpy()
            images.requires_grad=True
            for i in range(len(model_checkpoints)):
                chk = torch.load(model_checkpoints[i], map_location="cpu")
                model = ModelWrapper4Pretrained(chk, nn.Sigmoid) #TODO: make this sigmoid dynamic
                model.to(device)
                model.eval()
                # outputs = model(images)
                # prediction = (outputs.detach().cpu().numpy()>confidence).astype(int)
                model_name = model_names[i]
                explain_out_model = os.path.join(explain_out, model_name)
                os.makedirs(explain_out_model, exist_ok=True)

                if 'DenseNet161' in model_name:
                    # all layers after downsampling to avoid same-sized output layers
                    # not output of blocks, but the following relu, as they cancel a lot of information
                    layers_of_interest = ['net.0.net.features.denseblock1.denselayer1.relu1',
                                        'net.0.net.features.denseblock2.denselayer1.relu1',
                                        'net.0.net.features.denseblock3.denselayer1.relu1',
                                        'net.0.net.features.denseblock4.denselayer1.relu1',
                                        'net.0.net.features']
                    
                elif 'IncepResV2' in model_name:
                    # all layers with dim red and all layers with increase of FM number
                    layers_of_interest = ['net.0.net.maxpool_3a',
                                        'net.0.net.maxpool_5a',
                                        'net.0.net.mixed_5b',
                                        'net.0.net.mixed_6a',
                                        'net.0.net.mixed_7a',
                                        'net.0.net.conv2d_7b']
                elif 'InceptionV3' in model_name:
                    # all layers that reduce dim or change FM number
                    layers_of_interest = ['net.0.net.Conv2d_4a_3x3',
                                        'net.0.net.Mixed_5b',
                                        'net.0.net.Mixed_5c',
                                        'net.0.net.Mixed_6a',
                                        'net.0.net.Mixed_7a',
                                        'net.0.net.Mixed_7b',
                                        'net.0.net.Mixed_7c']
                elif 'ResNet18' in model_name:
                    layers_of_interest = ['net.0.net.maxpool',
                                        'net.0.net.layer1',
                                        'net.0.net.layer2',
                                        'net.0.net.layer3',
                                        'net.0.net.layer4']
                elif 'ResNet34' in model_name:
                    layers_of_interest = ['net.0.net.maxpool',
                                        'net.0.net.layer1',
                                        'net.0.net.layer2',
                                        'net.0.net.layer3',
                                        'net.0.net.layer4']
                else:
                    sys.exit("Invalid model!")

                try:
                    hk = Hooker(model)
                    o = model(images)
                    act = hk.get_hooked_activations()
                    
                    act = {k: v for k, v in act.items() if k in layers_of_interest}

                    out = {
                        "input": images.detach().cpu().numpy(),
                        "target": target,
                        "activations": act
                    }

                    with open(f"{explain_out_model}/{imgid}.npy", 'wb') as f:
                        np.save(f, out)
                except:
                    pass
                
                # torch.onnx.export(model, images, f"{ONNX_out}/{model_name}.onnx")

                del chk, model
                gc.collect()
                torch.cuda.empty_cache()
            
            # visualize_model_visualCNN(model, layerID=layerID4visualCNN, filter_position=filter_position, im_path=img_path, classID=2, inputs=images, file_name_to_export=explain_out_img_viscnn) #TODO: untested

    print("All Done!!")