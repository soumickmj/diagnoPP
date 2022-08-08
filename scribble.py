try:
    a=5/0
except Exception as ex:
    print(ex)
    print(type(ex))
    print(str(ex))
    


import os
import pandas as pd
import numpy as np
import json
from shutil import copyfile

seed = int('08061994')

###########For creating the fold splits
metaclean = pd.read_csv(r"S:\MEMoRIAL_SharedStorage_M1.2+4+7\Data\PublicDSs\Covid-19\IEEE8023_COVID_CHESTXRay\metadata_cleanV2.csv")
train = 0.6
test = 0.4
val = 0.0
trainpatients = []
testpatients = []
valpatients = []
for finding in metaclean.finding.unique():
    n = len(metaclean[metaclean.finding==finding].patientid.unique())
    nTrain = int(n*train)
    nTest = int(n*test)
    nVal = n - (nTrain+nTest)
    temp_train = list(metaclean[metaclean.finding==finding].sample(n=nTrain, random_state=seed).patientid)
    temp_test = list(metaclean[(metaclean.finding==finding) & (~metaclean.patientid.isin(temp_train))].sample(n=nTest, random_state=seed).patientid)
    trainpatients += temp_train
    testpatients += temp_test
    if val != 0:
        temp_val = list(metaclean[(metaclean.finding==finding) & (~metaclean.patientid.isin(temp_train+temp_test))].patientid)
        valpatients += temp_val

with open(r'S:\MEMoRIAL_SharedStorage_M1.2+4+7\Data\PublicDSs\Covid-19\IEEE8023_COVID_CHESTXRay\IEEE8023V2_60_40_00fold4.txt', 'w') as outfile:
    jData = {}
    jData['seed'] = seed
    jData['train'] = list(set(trainpatients))
    jData['test'] = list(set(testpatients))
    jData['val'] = list(set(valpatients))
    json.dump(jData, outfile)
###########


# ####################Copy Healthy X-Rays from Kaggle to IEEE
# path = r"S:\MEMoRIAL_SharedStorage_M1.2+4+7\Data\PublicDSs\Covid-19\Kaggle\Chest_X-Ray_Images_Pneumonia\train\NORMAL"
# n_files = 500
# outpath = r"S:\MEMoRIAL_SharedStorage_M1.2+4+7\Data\PublicDSs\Covid-19\IEEE8023_COVID_CHESTXRay\imagesV2"
# metapath = r"S:\MEMoRIAL_SharedStorage_M1.2+4+7\Data\PublicDSs\Covid-19\IEEE8023_COVID_CHESTXRay\metadata_copy.csv"
# metaoutpath = r"S:\MEMoRIAL_SharedStorage_M1.2+4+7\Data\PublicDSs\Covid-19\IEEE8023_COVID_CHESTXRay\metadataV2_temp.csv"
# files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
# np.random.shuffle(files)

# meta = pd.read_csv(metapath)
# pID = meta.patientid.max()
# count = 0
# for f in files:
#     print(count)
#     if count == n_files:
#         break
#     if f not in list(meta.filename):
#         copyfile(os.path.join(path, f), os.path.join(outpath, f))
#         df = pd.DataFrame(columns=meta.columns)
#         pID += 1
#         df.patientid = [pID]
#         df.finding = ['No Finding']
#         df.view = ['PA'] #just fake
#         df.modality = ['X-ray']
#         df.folder = ['images']
#         df.filename = [f]
#         df.other_notes = ['Normal cases from Kaggle dataset: Chest_X-Ray_Images_Pneumonia, View unknown, PA added by default']
#         meta = meta.append(df)
#         count += 1
# meta.to_csv(metaoutpath)

# # ####################Copy Pneumonia X-Rays from Kaggle to IEEE
# path = r"S:\MEMoRIAL_SharedStorage_M1.2+4+7\Data\PublicDSs\Covid-19\Kaggle\Chest_X-Ray_Images_Pneumonia\train\PNEUMONIA"
# n_files = 500
# unique_patients = True
# outpath = r"S:\MEMoRIAL_SharedStorage_M1.2+4+7\Data\PublicDSs\Covid-19\IEEE8023_COVID_CHESTXRay\imagesV2"
# metapath = r"S:\MEMoRIAL_SharedStorage_M1.2+4+7\Data\PublicDSs\Covid-19\IEEE8023_COVID_CHESTXRay\metadataV2_temp.csv"
# metaoutpath = r"S:\MEMoRIAL_SharedStorage_M1.2+4+7\Data\PublicDSs\Covid-19\IEEE8023_COVID_CHESTXRay\metadataV2.csv"
# files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
# np.random.shuffle(files)

# nVirs = n_files//2
# nBac = n_files - nVirs

# meta = pd.read_csv(metapath)
# pID_initial = meta.patientid.max()

# count_vir = 0
# count_bac = 0
# for f in files:
#     print(str(count_vir)+':'+str(count_bac))
#     if (count_vir == nVirs) and (count_bac == nBac):
#         break
#     if f not in list(meta.filename):
#         if 'bacteria' in f:
#             isBac = True
#             if count_bac == nBac:
#                 continue
#         else:
#             isBac = False
#             if count_vir == nVirs:
#                 continue
#         tmp = f.split('_')
#         pID = pID_initial + int(tmp[0].replace('person',''))
#         if pID not in list(meta.patientid):
#             offset = 0
#         else:
#             if unique_patients:
#                 continue
#             offset = meta[meta.patientid==pID].offset.max() + 1
#         copyfile(os.path.join(path, f), os.path.join(outpath, f))
#         df = pd.DataFrame(columns=meta.columns)
#         df.patientid = [pID]
#         df.offset = [offset]
#         df.finding = ["Bacterial Pneumonia" if isBac else "Viral Pneumonia"]
#         df.view = ['PA'] #just fake
#         df.modality = ['X-ray']
#         df.folder = ['images']
#         df.filename = [f]
#         df.other_notes = ['Pneumonia cases from Kaggle dataset: Chest_X-Ray_Images_Pneumonia, View unknown, PA added by default']
#         meta = meta.append(df)
#         if isBac:
#             count_bac += 1
#         else:
#             count_vir += 1
# meta.to_csv(metaoutpath)