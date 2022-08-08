import glob
import sys
import os
import numpy as np
import pandas as pd
from helpers import *

#Dataset IDs:
#Covid19CTSeg20: 0 [Covid]
#MedDecathlon: 1 [Tumor]
#NSCLC: 2 [Effusion]
#LNDb: 3 [Nodules-Tumor]
#StructSeg: 4 (not using)

def Covid19CTSeg20(root):
    #Data reader for COVID-19-CT-Seg of CT Seg Benchmark
    #Dataype of the files: .nii.gz
    #Folders:-
    #   COVID-19-CT-Seg_20cases
    #   Infection_Mask: 1=infection, 0=background
    #   Lung_and_Infection_Mask: 1=Left Lung, 2=Right lung, 3=infection, 0=back
    #   Lung_Mask: 1=Left Lung, 2=Right lung, 0=back
    vol_root = os.path.join(root, 'COVID-19-CT-Seg_20cases')
    infec_root = os.path.join(root, 'Infection_Mask')
    lungEinfec_root = os.path.join(root, 'Lung_and_Infection_Mask')
    lung_root = os.path.join(root, 'Lung_Mask')
    files = [f for f in os.listdir(vol_root) if os.path.isfile(os.path.join(vol_root, f))]
    
    subjects = []
    vol_paths = []
    infec_paths = []
    lungEinfec_paths = []
    lung_paths = []
    sicks = []
    for filename in files:
        subjects.append(filename.split('.')[0])
        vol_paths.append(os.path.join(vol_root, filename))
        lungEinfec_paths.append(os.path.join(lungEinfec_root, filename))
        lung_paths.append(os.path.join(lung_root, filename))
        infec_path = os.path.join(infec_root, filename)
        infec_paths.append(infec_path)

        # sicks.append(np.any(ReadNIFTI(infec_path)))
        sicks.append(True) #Already checked that all are sick, so not reading them to save time

    return {'subjects': subjects, 'vol_paths': vol_paths, 'lung_paths':lung_paths, 'infec_paths':infec_paths, 'is_sick':sicks}#, 'meta':[-1]*len(subjects)}

def MedDecathlon(root, isTrainSet=True):
    #Data reader for Medical decathlon datasets, for Covid-19 work will be only with Lung
    #TODO: implement for imagesTs too. For now, only implimenting for imagesTr
    #Dataype of the files: .nii.gz
    #Folders:-
    #   imagesTr, imagesTs 
    #   labelsTr - For Lung: 1=Cancer, 0=back
    vol_root = os.path.join(root, 'imagesTr' if isTrainSet else 'imagesTs')
    infec_root = os.path.join(root, 'labelsTr') if isTrainSet else None
    metaJSON = os.path.join(root, 'dataset.json') #nothing useful. Just details about the mask and data
    files = [f for f in os.listdir(vol_root) if os.path.isfile(os.path.join(vol_root, f)) and f.split('.')[0]]

    subjects = []
    vol_paths = []
    infec_paths = []
    sicks = []
    for filename in files:
        subjects.append(filename.split('.')[0])
        vol_paths.append(os.path.join(vol_root, filename))
        infec_path = os.path.join(infec_root, filename)
        infec_paths.append(infec_path)

        # sicks.append(np.any(ReadNIFTI(infec_path)))
        sicks.append(True) #Already checked that all are sick, so not reading them to save time

    return {'subjects': subjects, 'vol_paths': vol_paths, 'lung_paths':[-1]*len(subjects), 'infec_paths':infec_paths, 'is_sick':sicks}#, 'meta':pd.read_json(metaJSON)}

def NSCLC(root, ignore_noneffusion=True):
    #Data reader for NSCLC dataset
    #Dataype of the files: .nii.gz
    #Folders:-
    #   OriginalVol - dicoms
    #   Pleural Effusion Segmentations April 2020 - lesion masks (effusion.nii)
    #   Thoracic Segmentations April 2020 - lung masks (lungMask_edit.nii)
    vol_root = os.path.join(root, 'OriginalVol', 'NSCLC-Radiomics')
    infec_root = os.path.join(root, 'Pleural Effusion Segmentations April 2020', 'Pleural.Effusion.Segmentations')
    lung_root = os.path.join(root, 'Thoracic Segmentations April 2020', 'Thoracic Segmentations')
    csv_root = os.path.join(root, "Thoracic and Pleural Effusion Segmentations April 2020.csv")
    subjects = [f for f in os.listdir(vol_root) if os.path.isdir(os.path.join(vol_root, f))]

    csv = pd.read_csv(csv_root)
    
    sublist = []
    vol_paths = []
    infec_paths = []
    lung_paths = []
    sicks = []
    for sub in subjects:
        sublist.append(sub)
        vol_path = os.path.join(vol_root, sub)
        vol_path = os.path.join(vol_path, getTopFolderItems(vol_path, 1, True)[0])
        vol_path = os.path.join(vol_path, getTopFolderItems(vol_path, 1, True)[0])
        vol_paths.append(vol_path)
        lung_path = os.path.join(lung_root, sub, "lungMask_edit.nii")
        lung_paths.append(lung_path)
        infec_path = os.path.join(infec_root, sub, "effusion.nii")
        infec_paths.append(infec_path)

        is_sick_csv = True if csv[csv['PatientID'] == sub]["Effusion.Event"].iloc[0] == 1.0 else False

        if ignore_noneffusion and not is_sick_csv:
            sublist.remove(sub)
            vol_paths.remove(vol_path)
            lung_paths.remove(lung_path)
            infec_paths.remove(infec_path)
            continue

        # try:
        #     is_sick_label = np.any(ReadNIFTI(infec_path))
        # except:
        #     is_sick_label = False #Not having mask means that subject doesn't have effusion (presumed)
        
        # if is_sick_csv != is_sick_label:
        #     sublist.remove(sub)
        #     vol_paths.remove(vol_path)
        #     lung_paths.remove(lung_path)
        #     infec_paths.remove(infec_path)
        #     print('Ignored as CSV and Label not matching: '+ sub)
        # else:
        #     sicks.append(True) 
        sicks.append(is_sick_csv) #Already checked with CSV, so not reading them to save time


    return {'subjects': sublist, 'vol_paths': vol_paths, 'lung_paths':lung_paths, 'infec_paths':infec_paths, 'is_sick':sicks}#, 'meta':[-1]*len(subjects)}

def LNDb(root, minPixelAgreePercent=33.33):
    #Data reader for LNDb dataset
    #Dataype of the files: .mhd + .raw
    #Folders:-
    #   data - input vols
    #   masks - lesion masks 
    vol_root = os.path.join(root, 'data') 
    infec_root = os.path.join(root, 'masks', 'voted')
    csv_root = os.path.join(root, 'trainset_csv', 'trainNodules.csv')
    filterXLSX = os.path.join(root, 'labels.xlsx')
    files = [f for f in os.listdir(vol_root) if os.path.isfile(os.path.join(vol_root, f)) and f.split('.')[-1] == 'mhd']

    xlsx = pd.read_excel(filterXLSX)

    subjects = []
    vol_paths = []
    infec_paths = []
    sicks = []
    for filename in files:
        datum = xlsx[xlsx['files'] == filename]
        if datum["votedlabels"].iloc[0] == True and abs(datum["VotedPixelPercent"].iloc[0]) >= minPixelAgreePercent:
            subjects.append(filename.split('.')[0])
            vol_paths.append(os.path.join(vol_root, filename))
            infec_path = os.path.join(infec_root, filename)
            infec_paths.append(infec_path)

            # [label,_,_,_] = readMhd(infec_path)
            # sicks.append(np.any(label))
            sicks.append(True) #Already checked with CSV, so not reading them to save time

    return {'subjects': subjects, 'vol_paths': vol_paths, 'lung_paths':[-1]*len(subjects), 'infec_paths':infec_paths, 'is_sick':sicks}#, 'meta':pd.read_csv(csv_root)}

def createDataDict(root_Covid19CTSeg20, root_MedDecathlon, root_NSCLC, root_LNDb, output_path):
    datadict = pd.DataFrame()
    if root_Covid19CTSeg20 is not None:
        ds = Covid19CTSeg20(root_Covid19CTSeg20)
        df = pd.DataFrame.from_dict(ds,orient='index').transpose()
        df['DSID'] = 0
        datadict = datadict.append(df, ignore_index = True)
    if root_MedDecathlon is not None:
        ds = MedDecathlon(root_MedDecathlon)
        df = pd.DataFrame.from_dict(ds,orient='index').transpose()
        df['DSID'] = 1
        datadict = datadict.append(df, ignore_index = True)
    if root_NSCLC is not None:
        ds = NSCLC(root_NSCLC)
        df = pd.DataFrame.from_dict(ds,orient='index').transpose()
        df['DSID'] = 2
        datadict = datadict.append(df, ignore_index = True)
    if root_LNDb is not None:
        ds = LNDb(root_LNDb)
        df = pd.DataFrame.from_dict(ds,orient='index').transpose()
        df['DSID'] = 3
        datadict = datadict.append(df, ignore_index = True)
    datadict.to_excel(os.path.join(output_path, 'datadict_0123_NSCLCEffOnly_LNDb33.xlsx'))

def readDatum(datum, readInfec=True, readLung=True):
    subject = datum.subjects.iloc[0]
    DSID = datum.DSID.iloc[0]
    vol_path = datum.vol_paths.iloc[0]
    infec_path = datum.infec_paths.iloc[0]
    lung_path = datum.lung_paths.iloc[0]

    #Read the volume and anotomically correct them in terms of roataions
    if DSID==0 or DSID==1:
        vol = ReadNIFTI(vol_path)
        vol = np.flip(vol, 1)#horizontal flip = 0, vertical flip=1
    elif DSID==2:
        vol = np.transpose(readDICOMFolder(vol_path))
    elif DSID==3:        
        [vol,_,_,_] = readMhd(vol_path)
        vol = np.transpose(vol)
    else:
        sys.exit("Invalid DSID")

    #Read the infec masks, if set to true and anotomically correct them in terms of roataions
    if readInfec:
        if DSID==0 or DSID==1 or DSID==2:
            infec = ReadNIFTI(infec_path)
            infec = np.flip(infec, 1)#horizontal flip = 0, vertical flip=1
        elif DSID==3:     
            [infec,_,_,_] = readMhd(infec_path)
            infec = np.transpose(infec)
        else:
            sys.exit("Invalid DSID for infec mask")
    else:
        infec = None

    #Read the lung masks, if set to true and anotomically correct them in terms of roataions
    if readLung:
        if DSID==0 or DSID==2:
            lung = ReadNIFTI(lung_path)
            lung = np.flip(lung, 1) #vertical flip
        else:
            sys.exit("Invalid DSID for lung mask")
    else:
        lung = None

    #Multi channel infec, 
    return AdvDict({'subject': subject, 'DSID': DSID, 'vol':vol, 'infec':infec, 'lung':lung})

def readAllData(datadict_path):
    #Just a temp function
    datadict = pd.read_excel(datadict_path)
    for datum in datadict:
        readDatum(datum)

if __name__ == "__main__" :
    # NSCLC(r"S:\MEMoRIAL_SharedStorage_M1.2+4+7\Data\PublicDSs\Covid-19\CT_Seg_Benchmark\NSCLC")
    # Covid19CTSeg20(r"S:\MEMoRIAL_SharedStorage_M1.2+4+7\Data\PublicDSs\Covid-19\CT_Seg_Benchmark\COVID-19-CT-Seg")
    # MedDecathlon(r"S:\MEMoRIAL_SharedStorage_M1.2+4+7\Data\PublicDSs\Covid-19\CT_Seg_Benchmark\MSD Lung Tumor\Task06_Lung\Task06_Lung")
    # LNDb(r"S:\MEMoRIAL_SharedStorage_M1.2+4+7\Data\Challanges\LNDb\LNDb dataset")

    # createDataDict(r"S:\MEMoRIAL_SharedStorage_M1.2+4+7\Data\PublicDSs\Covid-19\CT_Seg_Benchmark\COVID-19-CT-Seg", 
    #                 r"S:\MEMoRIAL_SharedStorage_M1.2+4+7\Data\PublicDSs\Covid-19\CT_Seg_Benchmark\MSD Lung Tumor\Task06_Lung\Task06_Lung", 
    #                 r"S:\MEMoRIAL_SharedStorage_M1.2+4+7\Data\PublicDSs\Covid-19\CT_Seg_Benchmark\NSCLC", 
    #                 r"S:\MEMoRIAL_SharedStorage_M1.2+4+7\Data\Challanges\LNDb\LNDb dataset",
    #                 r"S:\MEMoRIAL_SharedStorage_M1.2+4+7\Data\PublicDSs\Covid-19\CT_Seg_Benchmark")

    readDatum(r"S:\MEMoRIAL_SharedStorage_M1.2+4+7\Data\PublicDSs\Covid-19\CT_Seg_Benchmark\datadict_0123_NSCLCEffOnly_LNDb33.xlsx")
    print('s')