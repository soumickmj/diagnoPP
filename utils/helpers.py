import os 
import numpy as np
import SimpleITK as sitk
# import nibabel as nib
import pandas as pd
from sklearn.metrics import multilabel_confusion_matrix, confusion_matrix
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from utils.dataloader_pyt_CXR import COVID19_Dataset

class AdvDict(object):
    def __init__(self, d):
        self.__dict__ = d

    def __len__(self):
        len(self.__dict__)

def getTopFolderItems(path, n, isFolder=True):
    return [f for f in os.listdir(path) if (os.path.isdir(os.path.join(path, f)) if isFolder else os.path.isfile(os.path.join(path, f)) )][:n]

def readDICOMFolder(path, returnArray=True, returnSITKimg=False):
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(path)
    reader.SetFileNames(dicom_names)
    image = reader.Execute()
    if returnArray and returnSITKimg:
        return sitk.GetArrayFromImage(image), image
    elif returnArray and not returnSITKimg:
        return sitk.GetArrayFromImage(image)
    elif not returnArray and returnSITKimg:
        return image
    else:
        return dicom_names

def ReadNIFTI(file_path):
    nii = nib.load(file_path)
    data = nii.get_data()
    return data

def WriteNIFTI(data, file_path):
    """Save a NIFTI file using given file path from an array
    Using: NiBabel"""
    if(np.iscomplex(data).any()):
        data = abs(data)
    nii = nib.Nifti1Image(data, np.eye(4)) 
    nib.save(nii, file_path)

def readMhd(filename):
    # read mhd/raw image
    itkimage = sitk.ReadImage(filename)
    scan = sitk.GetArrayFromImage(itkimage) #3D image
    spacing = itkimage.GetSpacing() #voxelsize
    origin = itkimage.GetOrigin() #world coordinates of origin
    transfmat = itkimage.GetDirection() #3D rotation matrix
    return scan,spacing,origin,transfmat

def writeMhd(filename,scan,spacing,origin,transfmat):
    # write mhd/raw image
    itkim = sitk.GetImageFromArray(scan, isVector=False) #3D image
    itkim.SetSpacing(spacing) #voxelsize
    itkim.SetOrigin(origin) #world coordinates of origin
    itkim.SetDirection(transfmat) #3D rotation matrix
    sitk.WriteImage(itkim, filename, False)    

def readCsv(csvfname):
    # read csv to list of lists
    with open(csvfname, 'r') as csvf:
        reader = csv.reader(csvf)
        csvlines = list(reader)
    return csvlines


def writeCsv(csfname,rows):
    # write csv from list of lists
    with open(csfname, 'w', newline='') as csvf:
        filewriter = csv.writer(csvf)
        filewriter.writerows(rows)
        
def viewWithFiji(vol):
    image = sitk.GetImageFromArray(np.transpose(vol))
    sitk.Show(image)

def VoteLNDb(root):
    #Data reader for LNDb dataset
    #Dataype of the files: .mhd + .raw
    #Folders:-
    #   data - input vols
    #   masks - lesion masks 
    vol_root = os.path.join(root, 'data') 
    label_root = os.path.join(root, 'masks', 'masks')
    votedlabel_root = os.path.join(root, 'masks', 'voted')
    files = [f for f in os.listdir(vol_root) if os.path.isfile(os.path.join(vol_root, f)) and f.split('.')[-1] == 'mhd']

    labels = []
    votedlabels = []
    pixB4Vote = []
    pixAfVote = []
    for filename in files:
        print(filename)
        vol_path = os.path.join(vol_root, filename)

        [vol,spacing,origin,transfmat] = readMhd(vol_path)
        votedmask = np.zeros(vol.shape)
        
        isPositive=False
        noOfRad = 0
        for i in [1,2,3]:
            maskfilename = filename.split('.')[0] + '_rad'+str(i)+'.' + filename.split('.')[1]
            label_path = os.path.join(label_root, maskfilename)
            if os.path.isfile(label_path):
                [label,_,_,_] = readMhd(label_path)
                label = np.where(label>=1, 1, 0) #Binarize all the findings of a particular radiologist
                votedmask += label
                isPositive=True
                noOfRad += 1

        print(noOfRad)
        pixB4Vote.append(np.count_nonzero(votedmask))
        if noOfRad > 2:
            votedmask = np.where(votedmask>=2, 1, 0) #Binarize the region which has been predicted by atleast two radiaologist
            pixAfVote.append(np.count_nonzero(votedmask))
        elif noOfRad > 1:
            votedmask = np.where(votedmask>=1, 1, 0) #Binarize the region which has been predicted by atleast one radiaologist
            pixAfVote.append(-np.count_nonzero(votedmask)) #Minus is just to mark
        else:
            pixAfVote.append(-1) #Just to mark that only one radiologist
            

        votedlabels.append(np.any(votedmask))
        labels.append(isPositive)
        writeMhd(os.path.join(votedlabel_root, filename),votedmask,spacing,origin,transfmat)

    df = pd.DataFrame(data={'files': files, 'labels': labels, 'votedlabels':votedlabels, 'pixB4Vote':
    pixB4Vote, 'pixAfVote':pixAfVote})
    df.to_excel(os.path.join(root, "labels.xlsx"))
    
def ViewLNDb(root, filename):
    #Data reader for LNDb dataset
    #Dataype of the files: .mhd + .raw
    #Folders:-
    #   data - input vols
    #   masks - lesion masks 
    vol_root = os.path.join(root, 'data') 
    label_root = os.path.join(root, 'masks', 'voted')
    
    vol_path = os.path.join(vol_root, filename)
    label_path = os.path.join(label_root, filename)

    [vol,_,_,_] = readMhd(vol_path)
    [label,_,_,_] = readMhd(label_path)

    vol = np.transpose(vol)
    label = np.transpose(label)
    viewWithFiji(vol)
    viewWithFiji(label)

def SaveLNDbASNifti(root, filename, outputpath):
    #Data reader for LNDb dataset
    #Dataype of the files: .mhd + .raw
    #Folders:-
    #   data - input vols
    #   masks - lesion masks 
    vol_root = os.path.join(root, 'data') 
    label_root = os.path.join(root, 'masks', 'voted')
    
    vol_path = os.path.join(vol_root, filename)
    label_path = os.path.join(label_root, filename)

    [vol,_,_,_] = readMhd(vol_path)
    [label,_,_,_] = readMhd(label_path)

    vol = np.transpose(vol)
    label = np.transpose(label)
    WriteNIFTI(vol, os.path.join(outputpath, filename+'_vol.nii.gz'))
    WriteNIFTI(label, os.path.join(outputpath, filename+'_mask.nii.gz'))

def get_inten_percentiles(imgfolder, metapath, views=["PA", "AP", "AP Supine"], equal_size=512, lower_bound=5, upper_bound=95):
    ds = COVID19_Dataset(imgpath=imgfolder, 
                    csvpath=metapath, 
                    views=views,
                    equal_size = equal_size)
    return ds.intenper_lower, ds.intenper_upper

def create_clean_meta(imgfolder, metapath, clean_metapath, views=["PA", "AP", "AP Supine"]):
    ds = COVID19_Dataset(imgpath=imgfolder, 
                        csvpath=metapath, 
                        views=views)
    meta = ds.csv
    meta.to_csv(clean_metapath)
    print(ds.intenper_lower)
    print(ds.intenper_upper)

def result_analyze_multilabelclassify(y_true, y_pred, isMultiLabel=True, labels='auto', eps=0.00000000000000001):
    if not isMultiLabel:
        cm = confusion_matrix(y_true, y_pred, labels=labels if labels != 'auto' else None)
        p = np.zeros((y_pred.shape[0], len(labels)))
        t = np.zeros((y_pred.shape[0], len(labels)))
        for i in range(y_pred.shape[0]):
            p[i, int(y_pred[i])] = 1
            t[i, int(y_true[i])] = 1
        y_pred = p
        y_true = t
    else:
        cm = None

    mcm = multilabel_confusion_matrix(y_true, y_pred)
    tn = mcm[:, 0, 0]
    tp = mcm[:, 1, 1]
    fn = mcm[:, 1, 0]
    fp = mcm[:, 0, 1]
    accuracy = (tn + tp) / (tn+tp+fn+fp+eps)
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps) #true positive rate or the sensitivity
    specificity = tn / (tn + fp + eps) #true negative rate
    fallout = fp / (fp + tn + eps) #false positive rate
    missrate = fn / (fn + tp + eps) #false negative rate
    f1 = 2 * ((precision*recall) / (precision+recall+eps))   

    #Micro Average
    micro_precision = sum(tp) / (sum(tp) + sum(fp) + eps)
    micro_recall = sum(tp) / (sum(tp) + sum(fn) + eps) 

    return {'y_true':y_true, 'y_pred':y_pred, 'mcm':mcm, 'cm':cm, 'accuracy':accuracy, 'precision':precision, 
            'recall':recall, 'specificity':specificity, 'fallout':fallout, 'missrate':missrate, 'f1':f1,
            'micro_precision':micro_precision, 'micro_recall':micro_recall}

if __name__ == "__main__" :
    # create_clean_meta(r"S:\MEMoRIAL_SharedStorage_M1.2+4+7\Data\PublicDSs\Covid-19\IEEE8023_COVID_CHESTXRay\imagesV2",
    #                                      r"S:\MEMoRIAL_SharedStorage_M1.2+4+7\Data\PublicDSs\Covid-19\IEEE8023_COVID_CHESTXRay\metadataV2.csv",
    #                                      r"S:\MEMoRIAL_SharedStorage_M1.2+4+7\Data\PublicDSs\Covid-19\IEEE8023_COVID_CHESTXRay\metadata_cleanV2.csv",
    #                                      )
    lower, upper = get_inten_percentiles(r"S:\MEMoRIAL_SharedStorage_M1.2+4+7\Data\PublicDSs\Covid-19\IEEE8023_COVID_CHESTXRay\imagesV2",
                                         r"S:\MEMoRIAL_SharedStorage_M1.2+4+7\Data\PublicDSs\Covid-19\IEEE8023_COVID_CHESTXRay\metadataV2.csv")
    print(lower)
    print(upper)
    # # VoteLNDb(r"S:\MEMoRIAL_SharedStorage_M1.2+4+7\Data\Challanges\LNDb\LNDb dataset")
    # VoteLNDb(r"D:\Soumick\LNDb")
    # SaveLNDbASNifti(r"S:\MEMoRIAL_SharedStorage_M1.2+4+7\Data\Challanges\LNDb\LNDb dataset", 'LNDb-0024.mhd', r"S:\MEMoRIAL_SharedStorage_M1.2+4+7\Data\Challanges\LNDb\LNDb dataset\temp")
    # ViewLNDb(r"S:\MEMoRIAL_SharedStorage_M1.2+4+7\Data\Challanges\LNDb\LNDb dataset", 'LNDb-0001.mhd')