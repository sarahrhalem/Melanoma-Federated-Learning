
import os
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms.functional as tf
from torch.utils.data import Dataset
from torchvision import transforms
import albumentations as A
from tqdm.notebook import tqdm

from sklearn.model_selection import StratifiedShuffleSplit
import matplotlib
import matplotlib.pyplot as plt




class MelanomaDataset(Dataset):
    def __init__(self, mode, img_dir, metadata_df=None, metadata_dir=None,transform=False, transform_prob=0.5,
    seed=0,X_as_feature=False,feature_dir='',use_memory_cache=False):
        
        self.img_dir= img_dir
        self.mode= mode
        self.metadata_df= metadata_df if (metadata_dir is None) else pd.read_csv(metadata_dir)
        self.transform= transform if(mode == 'train') else False
        self.transform_prob=transform_prob
        self.X_as_feature = X_as_feature
        self.feature_dir = feature_dir
        self.feature_extractor = None
        self.feature_extractor_device =None
        self.feature_cache = None# actual cache for data
        self.is_cached = None#store for which data is chached
        self.use_memory_cache = use_memory_cache
        torch.manual_seed(seed)
    
    def cache_feature(self,X,idx,t_idx=None):
        if(self.use_memory_cache):
            if(self.feature_cache is None):#we initialize the cache if it is not already done
                cache_t_size = 11 if(self.transform) else 1
                self.feature_cache = np.zeros((self.__len__(),cache_t_size,1280))
                self.is_cached = np.zeros((self.__len__(),cache_t_size))
                
            c_idx = 0 if(t_idx==None) else (1+t_idx)
            self.feature_cache[idx,c_idx]=X
            self.is_cached[idx,c_idx]=1
        return

    def start_feature_extraction(self,feature_extractor,feature_dir,device):
        self.X_as_feature = False
        self.feature_dir = feature_dir
        self.feature_extractor = feature_extractor
        self.feature_extractor_device =device
        return

    def end_feature_extraction(self):
        self.X_as_feature = True
        self.feature_extractor = None
        self.feature_extractor_device = None
        return

    def image_name(self,idx):
        return self.metadata_df.iloc[idx]['image_name']

    def load_img(self, idx):
        
        #img_path = self.img_dir + os.listdir(self.img_dir)[idx] 
        #img_path = self.img_dir + self.metadata_df.iloc[idx]['image_name']+".jpg"
        img_path = self.img_dir + self.image_name(idx)+".jpg"
        for data in img_path:
            img=Image.open(img_path)
        
        image= np.asarray(img)
        
        return image

    def load_feature(self,idx):
        if(self.transform):
            a = torch.rand(1).item()
            if(a>=self.transform_prob):
                if((self.is_cached is not None) and (self.is_cached[idx,0]==1)):
                    X = self.feature_cache[idx,0]
                    return torch.tensor(X)
                else:    
                    img_path = self.feature_dir + self.image_name(idx)+".txt"
                    #X = np.genfromtxt(img_path,delimiter=",")
                    X = np.loadtxt(img_path,delimiter=",")
                    self.cache_feature(X,idx)
                    return torch.tensor(X)
            else:
                t_idx = int(torch.rand(1).item()*10)
                if((self.is_cached is not None) and (self.is_cached[idx,t_idx+1]==1)):
                    X = self.feature_cache[idx,t_idx+1]
                    return torch.tensor(X)
                else:
                    img_path = self.feature_dir + self.image_name(idx)+"_T"+str(t_idx)+".txt"
                    #X = np.genfromtxt(img_path,delimiter=",")
                    X = np.loadtxt(img_path,delimiter=",")
                    self.cache_feature(X,idx,t_idx)
                    return torch.tensor(X)
        else:
            if((self.is_cached is not None) and (self.is_cached[idx,0]==1)):
                X = self.feature_cache[idx,0]
                return torch.tensor(X)
            else:    
                img_path = self.feature_dir + self.image_name(idx)+".txt"
                #X = np.genfromtxt(img_path,delimiter=",")
                X = np.loadtxt(img_path,delimiter=",")
                self.cache_feature(X,idx)
                return torch.tensor(X)

    def load_label(self, idx):
        return self.metadata_df.iloc[idx]['target']
    
    def __len__(self):
        return self.metadata_df.shape[0]
    
    def __getitem__(self, idx):
        if(self.X_as_feature):
            X= self.load_feature(idx)   
        else:
            if(self.feature_extractor is None):
                X= self.load_img(idx) 
                X= self.transform_img(X) 
            else:
                t_value = self.transform
                p_value = self.transform_prob
                
                self.feature_extractor.eval()
                self.feature_extractor.to(self.feature_extractor_device)
                #first we load the image without transfrom 
                self.transform =False

                X= self.load_img(idx) 
                X= self.transform_img(X)
                X=X[np.newaxis,:]
                X= X.to(self.feature_extractor_device,dtype=torch.float)
                X=self.feature_extractor(X)
                np.savetxt(self.feature_dir + self.image_name(idx)+".txt",X.cpu().numpy(),delimiter=",")
                # we save a list of features from transfromed image, these images will be needed when we train with transfrom using feature data
                self.transform =True
                self.transform_prob=1
                for t_idx in range(10):
                    X= self.load_img(idx) 
                    X= self.transform_img(X)
                    X=X[np.newaxis,:]
                    X=X.to(self.feature_extractor_device,dtype=torch.float)
                    X=self.feature_extractor(X)
                    np.savetxt(self.feature_dir + self.image_name(idx)+"_T"+str(t_idx)+".txt",X.cpu().numpy(),delimiter=",")
                
                self.transform =t_value
                self.transform_prob=p_value

        y= self.load_label(idx)    
        return X,y

### data augmentation allows for 16 scenarios 
### probability of 80% for the transformation in the data set is choosen to allows us to have enough images with oversampling: change from 2% to 20%
        
    def transform_img(self, image):
        
        
        transform_crop=A.CenterCrop(224,224, p=1)
        X=transform_crop(image=image)
        
        if self.transform:
            transform_train=A.Compose([
                A.OneOf([
                    A.HorizontalFlip(p=0.25),
                    A.VerticalFlip(p=0.25),
                    A.Rotate(p=0.25),
                    A.Transpose(p=0.25)     
                ], p=1),
                A.OneOf([A.GaussNoise(p=0.25),
                        A.ElasticTransform(alpha=3, p=0.25),
                        A.OpticalDistortion(p=0.25),
                        A.MotionBlur(blur_limit=5, p=0.25)
                        ], p=1)], p=self.transform_prob)
            X=transform_train(image=X['image'])
            
        transform_norm=A.Normalize()
        X=transform_norm(image=X['image'])
        
        to_tensor= transforms.ToTensor()
        X=to_tensor(X['image'])
   
        return X


def split_datasets(img_dir,metadata_dir,val_size=0.2,test_size=0.1,
                              split_seed=0,transform=True,transform_prob=0.8,dataseed=0,
                              X_as_feature=False,feature_dir='',use_memory_cache=False):
    
    splitter = StratifiedShuffleSplit(n_splits=1,test_size=(val_size+test_size), random_state=split_seed)
    metadata_df= pd.read_csv(metadata_dir)
    
    patient_tgt_1=metadata_df[metadata_df['target']==1]['patient_id'].unique()
    patient_tgt_0=metadata_df[~metadata_df['patient_id'].isin(patient_tgt_1)]['patient_id'].unique()
    
    patient_list=list(patient_tgt_0)+list(patient_tgt_1)
    y=[0]*len(patient_tgt_0)+[1]*len(patient_tgt_1)
    X=range(len(y))
    
    
    train_patient_idx, val_test_patient_idx = next(iter(splitter.split(X, y)))
    train_patient=[patient_list[idx] for idx in train_patient_idx]
    
    val_test_patient=[patient_list[idx] for idx in val_test_patient_idx]
    val_test_y = [y[idx] for idx in val_test_patient_idx]
    X_val_test = range(len(val_test_y))
    
    val_test_splitter = StratifiedShuffleSplit(n_splits=1,test_size=(test_size)/(val_size+test_size), random_state=split_seed)
    val_patient_idx, test_patient_idx= next(iter(val_test_splitter.split(X_val_test,val_test_y)))
    
    val_patient=[val_test_patient[idx] for idx in val_patient_idx]
    test_patient=[val_test_patient[idx] for idx in test_patient_idx]
    
    train_df=metadata_df[metadata_df['patient_id'].isin(train_patient)]
    val_df  =metadata_df[metadata_df['patient_id'].isin(val_patient)]
    test_df  =metadata_df[metadata_df['patient_id'].isin(test_patient)]
    
    train_df.reset_index(drop=True,inplace=True)
    val_df.reset_index(drop=True,inplace=True)
    test_df.reset_index(drop=True,inplace=True)
    
    train_dataset = MelanomaDataset('train', img_dir, train_df, transform=transform, transform_prob=transform_prob, seed=dataseed)
    val_dataset = MelanomaDataset('validate', img_dir, val_df, transform=False, seed=dataseed)
    test_dataset = MelanomaDataset('test', img_dir, test_df, transform=False, seed=dataseed)
    
    train_dataset.X_as_feature=X_as_feature 
    train_dataset.feature_dir=feature_dir
    train_dataset.use_memory_cache=use_memory_cache

    val_dataset.X_as_feature=X_as_feature
    val_dataset.feature_dir=feature_dir
    val_dataset.use_memory_cache=use_memory_cache

    test_dataset.X_as_feature=X_as_feature
    test_dataset.feature_dir=feature_dir

    return train_dataset,val_dataset,test_dataset

def create_dataloaders(train_set, val_set, test_set, batch_size=32, oversampling_ratio=0.2):

    weights = np.array(train_set.metadata_df['target'].value_counts())
    weights = len(train_set)/weights
    weights[1]=oversampling_ratio*weights[1]/weights[0]
    weights[0]=1
    sample_weights = np.array([weights[t] for t in train_set.metadata_df['target']])
    sample_weights = torch.from_numpy(sample_weights)

    sampler = torch.utils.data.WeightedRandomSampler(sample_weights,len(train_set))

    train_loader= torch.utils.data.DataLoader(train_set, batch_size=batch_size, num_workers=0, sampler=sampler)
    val_loader= torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader=torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0) 
    
    return train_loader, val_loader, test_loader

def generate_data_feature_cache(dataset,feature_extractor,feature_dir,device,extraction_batch_size=32):

    
    loader= torch.utils.data.DataLoader(dataset, batch_size=extraction_batch_size, num_workers=0)
    loader.dataset.start_feature_extraction(feature_extractor,feature_dir,device)
    for batch_idx, (image, label) in tqdm(enumerate(loader), total=len(loader)):
        print("Batch "+str(batch_idx)+" Feature data saving done")
    
    loader.dataset.end_feature_extraction()
    return
    