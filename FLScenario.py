
import os
import sys
import numpy as np
import pandas as pd
import time
import pickle
import random
import sklearn
import copy


import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms.functional as tf
from torch.utils.data import Dataset
from torchvision import transforms


import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

from utils import *
from Visualise import *
from ResizeImages import *
from MelanomaDataset import *
from MelanomaEfficientNet import *
from TrainVal import *
from Test import *
from Plot import *


def create_workerdataloader(worker_dataset, trainloader,worker_oversampling,local_batchsize):
    #compute the oversampling ration from the main train loader
    if(worker_oversampling is None):
        return torch.utils.data.DataLoader(worker_dataset, batch_size=local_batchsize,num_workers=0)
    else:
        worker_weights = np.array(worker_dataset.metadata_df['target'].value_counts())
        if(worker_weights.shape[0]<2):
            return torch.utils.data.DataLoader(worker_dataset, batch_size=trainloader.batch_size,num_workers=0)
        else:   
            worker_weights = len(worker_dataset)/worker_weights
            train_weights = np.array(trainloader.dataset.metadata_df['target'].value_counts())
            train_weights = len(trainloader.dataset)/train_weights

            trainloader_weights = set(list(trainloader.sampler.weights.numpy()))
            trainloader_weights = [min(trainloader_weights),max(trainloader_weights)]
            train_oversampling = trainloader_weights[1]*train_weights[0]/train_weights[1]
            if(worker_oversampling=="uniform"):
                worker_weights[1]=train_oversampling*worker_weights[1]/worker_weights[0]
                worker_weights[0]=1

            elif(worker_oversampling=="max-train-worker"):
                if((1/worker_weights[1])<=(train_oversampling/(1+train_oversampling))):
                    worker_weights[1]=train_oversampling*worker_weights[1]/worker_weights[0]
                else:
                    worker_weights[1]=1
                worker_weights[0]=1


            sample_weights = np.array([worker_weights[t] for t in worker_dataset.metadata_df['target']])
            sample_weights = torch.from_numpy(sample_weights)

            sampler = torch.utils.data.WeightedRandomSampler(sample_weights,len(worker_dataset))

            return torch.utils.data.DataLoader(worker_dataset, batch_size=local_batchsize,num_workers=0, sampler=sampler)

  

class DataScenarioFL:
    def __init__(self,worker_oversampling=None,
                 unbalanced_datasize=False,lower_datasize=1,lower_data_first=False,split_by='patient_id',seed=0):
        self.worker_oversampling=worker_oversampling
        self.rng=np.random.default_rng(seed)
        self.unbalanced_datasize = unbalanced_datasize
        self.lower_datasize = lower_datasize
        self.lower_data_first = lower_data_first
        self.split_by = split_by
        
    def split(self,dataloader,n_workers,local_batchsize):
        all_metadata = dataloader.dataset.metadata_df
        size_list = self.create_datasize_list(dataloader,n_workers)
        w_patients = self.worker_patient_list(dataloader,n_workers,size_list)   
        dataloaders=[]
        for patient_list in w_patients:
            df = all_metadata[all_metadata[self.split_by].isin(patient_list)]
            df.reset_index(drop=True,inplace=True)
            ###TO CHECK THE SEED
            worker_data = MelanomaDataset('train', dataloader.dataset.img_dir,
                                       df, transform=dataloader.dataset.transform, 
                                       transform_prob=dataloader.dataset.transform_prob, seed=0,
                                       X_as_feature=dataloader.dataset.X_as_feature,
                                       feature_dir=dataloader.dataset.feature_dir,
                                       use_memory_cache=dataloader.dataset.use_memory_cache)
            
            
            worker_dataloader = create_workerdataloader(worker_data,dataloader,self.worker_oversampling,local_batchsize)
            dataloaders+=[worker_dataloader]
        return dataloaders
    
    def worker_patient_list(self,dataloader,n_workers,size_list):
        return
    
    def create_datasize_list(self,dataloader,n_workers):
        all_metadata = dataloader.dataset.metadata_df
        if(self.unbalanced_datasize):
        
            ratio = (len(all_metadata[self.split_by].unique()) - self.lower_datasize*n_workers)/((n_workers-1)*n_workers/2)
            ratio = int(ratio)
            if(self.lower_data_first):   
                size_list = [self.lower_datasize + idx*ratio for idx in range(n_workers)]
                return size_list
            else:
                size_list = [self.lower_datasize + (n_workers-idx)*ratio for idx in range(1,n_workers+1)]
                return size_list
        else:
            n_patient = int(len(all_metadata[self.split_by].unique())/n_workers)
            return [int(len(all_metadata[self.split_by].unique())/n_workers)]*n_workers
    
class RandomScenarioFL(DataScenarioFL):
    def __init__(self,worker_oversampling=None,unbalanced_datasize=False,
                 lower_datasize=1,lower_data_first=False,split_by='patient_id',seed=0):
        super(RandomScenarioFL, self).__init__(worker_oversampling,unbalanced_datasize,
                                                lower_datasize,lower_data_first,split_by,seed)
        
    def worker_patient_list(self,dataloader,n_workers,size_list):
        all_metadata = dataloader.dataset.metadata_df
        
        patient_ids = list(all_metadata[self.split_by].unique())
        self.rng.shuffle(patient_ids)
        
        w_patients_list=[]
        idx = 0
        for s in size_list:
            patient_w = patient_ids[idx:idx+s]
            w_patients_list+=[patient_w]
            idx+=s
        
        return w_patients_list
    
        
class FeatureLabelUnbalancedScenarioFL(DataScenarioFL):
    def __init__(self,feature_name=None,feature_value=None,f_unbal_ratio=None,label_unbal_ratio=None,
                 worker_oversampling=None,unbalanced_datasize=False,lower_datasize=1,lower_data_first=False,
                dirichlet_alpha=None,split_by='patient_id',seed=0):
        super(FeatureLabelUnbalancedScenarioFL, self).__init__(worker_oversampling,
                                                               unbalanced_datasize,lower_datasize,
                                                               lower_data_first,split_by,seed)
        self.feature_name=feature_name
        self.feature_value=feature_value
        self.f_unbal_ratio=f_unbal_ratio
        self.label_unbal_ratio=label_unbal_ratio
        self.dirichlet_alpha = dirichlet_alpha

    def worker_patient_list(self,dataloader,n_workers,size_list):
        
        all_metadata = dataloader.dataset.metadata_df
        
        if((self.feature_name is not None) and (self.label_unbal_ratio is not None)):
            return
        elif(self.feature_name is None):
            if((self.label_unbal_ratio is not None) or (self.dirichlet_alpha is not None)):
                tgt1_patients = list(all_metadata[all_metadata['target']==1][self.split_by].unique())
                self.rng.shuffle(tgt1_patients)
                tgt0_patients = list(all_metadata[~all_metadata[self.split_by].isin(tgt1_patients)][self.split_by].unique())
                self.rng.shuffle(tgt0_patients)
                w_patients_list=[]
                idx_tgt1=0
                idx_tgt0=0
                tgt1_sizes = self.unbalanced_labels_sizes(len(tgt1_patients),size_list)
                for idx,s in enumerate(size_list):              
                    #n_tgt1= min(int(round(self.label_unbal_ratio*len(tgt1_patients[idx_tgt1:]))),s)
                    n_tgt1=tgt1_sizes[idx]
                    n_tgt0= max(s-n_tgt1,1)

                    patient_w=tgt1_patients[idx_tgt1:idx_tgt1+n_tgt1]
                    patient_w+=tgt0_patients[idx_tgt0:idx_tgt0+n_tgt0]
                    idx_tgt1+=n_tgt1
                    idx_tgt0+=n_tgt0
                    w_patients_list+=[patient_w]
                    
                return w_patients_list
            else:
                return
        elif((self.feature_name is not None)and (self.label_unbal_ratio is None)):
            
            if(self.feature_name=='age_approx'):
                df = all_metadata.sort_values(by=['age_approx'],ascending=False)
                fv_patients = list(df[df[self.feature_name]>=self.feature_value][self.split_by].unique())
                fnv_patients = list(df[~df[self.split_by].isin(fv_patients)][self.split_by].unique())
            
            else:
                fv_patients = list(all_metadata[all_metadata[self.feature_name]==self.feature_value][self.split_by].unique())
                fnv_patients = list(all_metadata[~all_metadata[self.split_by].isin(fv_patients)][self.split_by].unique())
            
            if(self.feature_name!='age_approx'):    
                self.rng.shuffle(fnv_patients)
                self.rng.shuffle(fv_patients)
            
            patient_per_worker = int(round(len(all_metadata[self.split_by].unique())/n_workers))
            w_patients_list=[]
            idx_tgt1=0
            idx_tgt0=0
            
            for s in size_list:              
                n_tgt1= min(int(round(self.f_unbal_ratio*len(fv_patients[idx_tgt1:]))),s)
                n_tgt0= max(s-n_tgt1,1)

                patient_w=fv_patients[idx_tgt1:idx_tgt1+n_tgt1]
                patient_w+=fnv_patients[idx_tgt0:idx_tgt0+n_tgt0]
                idx_tgt1+=n_tgt1
                idx_tgt0+=n_tgt0
                w_patients_list+=[patient_w]
                
            return w_patients_list
        
        else:
            return
    
    def unbalanced_labels_sizes(self,label1_size,split_size):
        if(self.dirichlet_alpha is not None):
            if(self.unbalanced_datasize):
                raise Exception(" Dirichlet unbalanced split supported for same size data splits")
            else:
                alpha = [self.dirichlet_alpha]*len(split_size)
                ratios = self.rng.dirichlet(alpha=alpha)
                n_label1 = [int(r*label1_size) for r in ratios]
                missing_labels = [(r*label1_size-int(r*label1_size)) for r in ratios]
                n_missing_label = label1_size-sum(n_label1)
                
                while(n_missing_label>0):
                    m_idx = missing_labels.index(max(missing_labels))
                    n_label1[m_idx]+=1
                    missing_labels[m_idx]=-2
                    n_missing_label-=1
                return n_label1
        else:
            idx_tgt1=0
            tgt1_patients = [idx for idx in range(label1_size)]
            n_label1=[]
            for s in split_size:              
                n_tgt1= min(int(round(self.label_unbal_ratio*len(tgt1_patients[idx_tgt1:]))),s)
                n_label1+=[n_tgt1]
                idx_tgt1+=n_tgt1
            return n_label1


 