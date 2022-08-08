import glob
import sys
import os
import numpy as np
import pandas as pd
# from skimage.io import imread, imsave
from PIL import Image, ImageOps
from torch.utils.data import Dataset, DataLoader

thispath = os.path.dirname(os.path.realpath(__file__))

def normalize1024(sample, maxval):
    """Scales images to be roughly [-1024 1024]."""
    sample = (2 * (sample.astype(np.float32) / maxval) - 1.) * 1024
    #sample = sample / np.std(sample)
    return sample

class Flip4View(object):

    def __init__(self, destin_view='PA'):
        self.destin_view = destin_view

    def __call__(self, img, view, destin_view=None):
        if destin_view is None:
            destin_view = self.destin_view

        if view == destin_view or destin_view is None:
            return img, view
        else:
            return np.flip(img, 1).copy(), destin_view

class SizeEqualizer(object):

    def __init__(self, destin_size=256, interpol=Image.BICUBIC):
        self.destin_size = destin_size
        self.interpol = interpol

    def __call__(self, img):
        if img.size[0] > img.size[1]:
            ratio = img.size[1] / img.size[0]
            new_size = (self.destin_size, int(self.destin_size*ratio))
            delta = self.destin_size - int(self.destin_size*ratio)
            padding = (0, delta//2, 0, delta-(delta//2))
        elif img.size[0] < img.size[1]:
            ratio = img.size[0] / img.size[1]
            new_size = (int(self.destin_size*ratio), self.destin_size)
            delta = self.destin_size - int(self.destin_size*ratio)
            padding = (delta//2, 0, delta-(delta//2), 0)
        else:
            new_size = (self.destin_size, self.destin_size)
            padding = (0, 0, 0, 0)

        img = img.resize(size=new_size, resample=self.interpol)
        img = ImageOps.expand(img, padding)
        return img

class IntenCropNorm(object):

    def __init__(self, intenper_upper, intenper_lower):
        self.intenper_upper = intenper_upper
        self.intenper_lower = intenper_lower

    def __call__(self, image):
        image = np.where(np.less_equal(image, self.intenper_lower), self.intenper_lower, image)
        image = np.where(np.greater_equal(image, self.intenper_upper), self.intenper_upper, image)
        image = (image - self.intenper_lower) / (self.intenper_upper - self.intenper_lower)
        return image

class COVID19_Dataset(Dataset):
    """
    COVID-19 image data collection

    Dataset: https://github.com/ieee8023/covid-chestxray-dataset
    
    Paper: https://arxiv.org/abs/2003.11597

    Original Code: from torchxrayvision (https://github.com/mlmed/torchxrayvision)
    """
    
    def __init__(self, 
                 imgpath=os.path.join(thispath, "covid-chestxray-dataset", "images"), 
                 csvpath=os.path.join(thispath, "covid-chestxray-dataset", "metadata.csv"), 
                 views=["PA"],
                 destination_view = None,#'PA', #set to None if not needed
                 modalities = ['X-ray'], #currently not in use, as CT has Axial or Coronal views, not PA, AP or AP Supine. So, by choosing the view, we are already choosing the modality
                 norm_type = 'intencrop', # 'intencrop': cropping intencities based on percentile; or 'max1' or 'normalize1024' (between -1024, +1024) or 'minmax (between -1, +1)' or 'minmaxPOS (between 0, +1)'
                 equal_size = 512,
                 transform=None, 
                 data_aug=None, 
                 nrows=None, 
                 seed=0,
                 unique_patients=False,
                 subgroup_diseases=True,
                 patientIDs=None,
                 intenper_upper=None,
                 intenper_lower=None,
                 interp_lowerbound=5,
                 interp_upperbound=95): #Supply a list with patient IDs or None for all

        super(COVID19_Dataset, self).__init__()
        np.random.seed(seed)  # Reset the seed so all runs are the same.
        self.imgpath = imgpath
        self.norm_type = norm_type
        self.transform = transform
        self.data_aug = data_aug
        self.views = views
        self.patientIDs = patientIDs
        self.subgroup_diseases = subgroup_diseases

        self.flip4View = Flip4View(destination_view)
        self.equalsize = SizeEqualizer(equal_size)
        
        # defined here to make the code easier to read
        pneumonias = ["COVID-19", "SARS", "MERS", "ARDS", "Streptococcus", "Pneumocystis", "Klebsiella", "Chlamydophila", "Legionella"]
        
        self.pathologies = ["Pneumonia","Viral Pneumonia", "Bacterial Pneumonia", "Fungal Pneumonia", "No Finding"] + pneumonias
        self.pathologies = sorted(self.pathologies)

        mapping = dict()
        mapping["Pneumonia"] = pneumonias
        mapping["Viral Pneumonia"] = ["COVID-19", "SARS", "MERS"]
        mapping["Bacterial Pneumonia"] = ["Streptococcus", "Klebsiella", "Chlamydophila", "Legionella"]
        mapping["Fungal Pneumonia"] = ["Pneumocystis"]
        
        # Load data
        self.csvpath = csvpath
        csv = pd.read_csv(self.csvpath, nrows=nrows)
        self.MAXVAL = 255  # Range [0 255]

        # Keep only the selected views.
        idx_pa = csv.view.isin(self.views)

        #selected patients only
        if self.patientIDs is not None and len(self.patientIDs) > 0:
            idx_pa = idx_pa & csv.patientid.isin(self.patientIDs)

        #apply the filters
        csv = csv[idx_pa]

        if unique_patients:
            filtered = pd.DataFrame(columns=csv.columns)
            for patientid in csv.patientid.unique():
                filtered = filtered.append(csv[csv.patientid.eq(patientid)].sort_values('offset').iloc[0])
            csv = filtered
    
        clean_csv = pd.DataFrame(columns=csv.columns)
        imgs = []
        for _, datum in csv.iterrows():
            imgid = datum.filename
            img_path = os.path.join(self.imgpath, imgid)
            if os.path.isfile(img_path):
                clean_csv = clean_csv.append(datum)
                if intenper_upper is None or intenper_lower is None:
                    img = Image.open(img_path)
                    if len(img.size) >= 2:
                        datum['dim0'] = img.size[0]
                        datum['dim1'] = img.size[1]
                        img = np.array(self.equalsize(img))
                        if len(img.shape) > 2:
                            img = img[:, :, 0]
                        imgs.append(img)
        if intenper_upper is None or intenper_lower is None:
            imgs = np.array(imgs)
            intenper_lower = np.percentile(imgs, interp_lowerbound)
            intenper_upper = np.percentile(imgs, interp_upperbound)

        self.intenCropNorm = IntenCropNorm(intenper_upper, intenper_lower)
        self.intenper_upper = intenper_upper
        self.intenper_lower = intenper_lower

        self.csv = clean_csv

        
        self.labels = []
        for pathology in self.pathologies:
            mask = self.csv["finding"].str.contains(pathology)
            if subgroup_diseases:
                if pathology in mapping:
                    for syn in mapping[pathology]:
                        mask |= self.csv["finding"].str.contains(syn)
            self.labels.append(mask.values)
        self.labels = np.asarray(self.labels).T
            
        self.labels = self.labels.astype(np.float32)
        
        #class-weight calculation
        classN = np.sum(self.labels, axis=0)
        maxN = classN.max()
        self.class_weight = np.divide(maxN, classN, where=classN!=0)


    def __repr__(self):
        pprint.pprint(self.totals())
        return self.__class__.__name__ + " num_samples={} views={}".format(len(self), self.views)
    
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        imgid = self.csv.filename.iloc[idx]
        img_path = os.path.join(self.imgpath, imgid)
        img = Image.open(img_path)
        img = np.array(self.equalsize(img))
        
        if self.norm_type == 'normalize1024':
            img = normalize1024(img, self.MAXVAL)  
        elif self.norm_type == 'max1':
            img = img / img.max()
        elif self.norm_type == 'intencrop':
            img = self.intenCropNorm(img)
        else:
            sys.exit('Normalization type not implimented')

        view = self.csv.view.iloc[idx]
        img, view = self.flip4View(img, view)

        # Check that images are 2D arrays
        if len(img.shape) > 2:
            img = img[:, :, 0]
        if len(img.shape) < 2:
            print("error, dimension lower than 2 for image")

        # Add color channel
        img = img[None, :, :]                    
                               
        if self.transform is not None:
            img = self.transform(img)

        if self.data_aug is not None:
            img = self.data_aug(img)

        label = self.labels[idx]

        if not self.subgroup_diseases:
            label = np.argmax(label, axis=0)
            
        return {"img":img, "lab":label, "idx":idx}
