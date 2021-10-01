
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
from FLWorker import *
from FLScenario  import *
from FLAdjustment import *

class SimpleIndexDataSet:
    def __init__(self,size):
        self.size = size
    def __getitem__(self,idx):
        if(idx<self.size):
            return idx
        else:
            return
    def __len__(self):
        return self.size
    
class ClientSampler:
    def __init__(self,c_replacment=True,shuffle=True,seed=0):
        self.c_replacment=c_replacment
        self.rng=np.random.default_rng(seed)
        self.shuffle=shuffle
        self.workers_set=None
        self.client_ratio=None
        self.n_rounds=None
        self.current_idx = 0 
        self.max_samples = None
        
    def reset(self,workers_set,client_ratio,n_rounds):
        self.workers_set=workers_set
        self.client_ratio=client_ratio
        self.n_rounds=n_rounds
        self.current_idx = 0 
        self.max_samples 
        
        if(self.c_replacment):
            self.max_samples = self.n_rounds
        else:
            client_epoch  = max(int(round(self.client_ratio*self.n_rounds)),1)
            self.max_samples =  int(client_epoch/self.client_ratio)
        return
    def get_sample(self,weights=None):
        return []

    
class ClientSimpleSampler(ClientSampler):
    def __init__(self,c_replacment=True,shuffle=True,seed=0):
        super().__init__(c_replacment,shuffle,seed)
        self.all_samples = []
    def reset(self,workers_set,client_ratio,n_rounds):
        super().reset(workers_set,client_ratio,n_rounds)
        self.all_samples = self.samples()

    def get_sample(self,weights=None):
        client_sample = self.all_samples[self.current_idx]
        self.current_idx+=1
        return client_sample
    
    def samples(self):
        rounds = []      
        if(self.c_replacment):
            worker_per_round = int(round(self.client_ratio*len(self.workers_set)))
            for i in range(self.n_rounds):
                workers_idx = [idx for idx in range(len(self.workers_set))]
                self.rng.shuffle(workers_idx)
                rounds+=[workers_idx[:worker_per_round]]
        else:
            client_epoch  = max(int(round(self.client_ratio*self.n_rounds)),1)
            for ce in range(client_epoch):
                workers_idx = [idx for idx in range(len(self.workers_set))]
                
                if(self.shuffle):
                    self.rng.shuffle(workers_idx)
                    
                worker_per_round = int(round(self.client_ratio*len(self.workers_set)))
                idx = 0
                while idx<len(workers_idx):
                    rounds+=[workers_idx[idx:idx+worker_per_round]]
                    idx=idx+worker_per_round
        
        return rounds

class ClientWeightedSampler(ClientSampler):
    def __init__(self,c_replacment=True,shuffle=True,seed=0):
        super().__init__(c_replacment,shuffle,seed)
        self.c_replacment=True

    def get_sample(self,weights=None):
        worker_per_round = int(round(self.client_ratio*len(self.workers_set)))

        workers_idx = [idx for idx in range(len(self.workers_set))]
        self.current_idx+=1
        if(weights is not None):
            return self.rng.choice(workers_idx,worker_per_round,p=weights,replace=False)
        else:
            return self.rng.choice(workers_idx,worker_per_round,replace=False)
    
class FedAvgAggregator:
    def __init__(self,n_rounds,client_ratio,seed=0,lr_decay = 0,min_local_updates=None,
                 fixed_local_updates=None,client_sampling='RandomReplace',
                 averaging='TotalDataSetWightedAveraging'):
        
        self.n_rounds = n_rounds
        self.client_ratio = client_ratio
        self.workers_set = []
        self.rounds = []
        self.data_size = 0
        self.rng=np.random.default_rng(seed)
        self.model_param={}
        self.pres_model_param={}
        self.lr_decay = lr_decay
        self.lr_multiplier = 1
        self.min_local_updates = min_local_updates
        self.fixed_local_updates=fixed_local_updates
        self.epoch_ratio = []
        self.client_sampling=client_sampling
        self.averaging=averaging
        self.fl_adjustments=None

        if('Weighted' in client_sampling):
            self.client_ratio_sampler = ClientWeightedSampler()
        elif(client_sampling=='RandomNoReplace'):
            self.client_ratio_sampler = ClientSimpleSampler(c_replacment=False,shuffle=True)
        elif client_sampling=='Sequential':
            self.client_ratio_sampler = ClientSimpleSampler(c_replacment=False,shuffle=False)
        elif client_sampling=='RandomReplace':
            self.client_ratio_sampler = ClientSimpleSampler()
        else:
            raise Exception("Client Sampling type not supported")
        self.sampling_weights=None
        self.round_avg_weights = None
        
    def reset(self,workers_set):
        self.workers_set = workers_set
        if(self.fixed_local_updates is not None):
            for worker in self.workers_set:
                worker.set_local_updates(self.fixed_local_updates)
        self.lr_multiplier = 1
        
        if(self.min_local_updates is not None):
            workers_dls = [len(worker.dataloader) for worker in workers_set]
            self.epoch_ratio = [int(max(dl_size,self.min_local_updates)/dl_size) for dl_size in workers_dls]    
        else:
            self.epoch_ratio = [1]*len(workers_set)
        self.data_size = 0
        
        for worker in self.workers_set:
            self.data_size+=len(worker.dataloader.dataset)
        
        self.client_ratio_sampler.reset(workers_set,self.client_ratio,self.n_rounds)
          
        self.model_param = {}
        self.pres_model_param={}
        
        if(self.client_sampling == 'DataSetWeightedSampling'): 
            self.sampling_weights = [len(worker.dataloader.dataset)/self.data_size for worker in self.workers_set]
            
        if(self.client_sampling == 'DataSetInvWeightedSampling'):
            weights = [self.data_size/len(worker.dataloader.dataset) for worker in self.workers_set]
            self.sampling_weights = [w/sum(weights) for w in weights]

        self.round_avg_weights = None
        
        return
    def update_avg_weights(self,round_workers):

        self.round_avg_weights = {}
        rnd_avg = []
        if(self.averaging=='TotalDataSetWightedAveraging'):
            rnd_avg = [len(self.workers_set[worker_idx].dataloader.dataset)/self.data_size for worker_idx in round_workers]
        elif(self.averaging=='RoundDataSetWightedAveraging'):
            weights = [len(self.workers_set[worker_idx].dataloader.dataset) for worker_idx in round_workers]
            rnd_avg = [w/sum(weights) for w in weights]
        elif(self.averaging=='FedNovaWightedAveraging'):
            ds_weights = [len(self.workers_set[worker_idx].dataloader.dataset) for worker_idx in round_workers]
            nds_weights = [w/sum(ds_weights) for w in ds_weights]
            
            lu_weights = [len(self.workers_set[worker_idx].dataloader) for worker_idx in round_workers]
            weights = [nds_weights[idx]/lu_weights[idx] for idx in range(len(round_workers))]
            eff_tau = 0
            for tau,nds in zip(nds_weights,lu_weights):
                eff_tau+=tau*nds
                
            rnd_avg = [eff_tau*w for w in weights]
        
        elif(self.averaging=='FedNovaUniformAveraging'):
            
            nds_weights = [1/len(round_workers)]*len(round_workers)
            
            lu_weights = [len(self.workers_set[worker_idx].dataloader) for worker_idx in round_workers]
            weights = [nds_weights[idx]/lu_weights[idx] for idx in range(len(round_workers))]
            eff_tau = 0
            for tau,nds in zip(nds_weights,lu_weights):
                eff_tau+=tau*nds
                
            rnd_avg = [eff_tau*w for w in weights]
        elif(self.averaging=='Uniform'):
            rnd_avg = [1/len(round_workers)]*len(round_workers)
        else:
            raise Exception("Averaging type not supported")

        for rnd_idx in range(len(round_workers)):
            self.round_avg_weights[round_workers[rnd_idx]]=rnd_avg[rnd_idx]
        return
        
    def start_round(self,model):
        self.model_param = {}
        self.pres_model_param = model.state_dict()
        
        round_workers = self.client_ratio_sampler.get_sample(self.sampling_weights)
        self.update_avg_weights(round_workers)
        return round_workers
    
    def add_local_model(self,model,worker_idx,sum_lr):
        #weight = len(self.workers_set[worker_idx].dataloader.dataset)/self.data_size
        weight = self.round_avg_weights[worker_idx]

        local_model_param = model.state_dict()
        for key in local_model_param.keys():
            if(key not in self.model_param.keys()):
                self.model_param[key] = self.pres_model_param[key]+(local_model_param[key]-self.pres_model_param[key]) * weight
            else:
                self.model_param[key] += (local_model_param[key]-self.pres_model_param[key]) * weight
                 
        return
    def update_model(self,model):
        model.load_state_dict(self.model_param)
        self.lr_multiplier /=(1+self.lr_decay)
        self.round_avg_weights = None
        return

 
class FedAdamAggregator(FedAvgAggregator):
    def __init__(self,n_rounds,client_ratio,global_lr,beta1,beta2,tau,seed=0,lr_decay=0,min_local_updates=None,fixed_local_updates=None,client_sampling='RandomReplace'):
        
        super().__init__(n_rounds,client_ratio,seed=seed,lr_decay=lr_decay,min_local_updates=min_local_updates,
                    fixed_local_updates=fixed_local_updates,client_sampling=client_sampling,averaging='Uniform')
        self.global_lr=global_lr
        self.beta1=beta1
        self.beta2=beta2
        self.tau=tau
        self.mt={}
        self.vt={}

    def reset(self,workers_set):
        super().reset(workers_set)
        self.mt={}
        self.vt={}
        return

    def update_model(self,model):

        local_model_param = model.state_dict()
        new_model_param = {}
        for key in self.model_param.keys():
            if key not in self.mt.keys():
                self.mt[key]=self.model_param[key]-local_model_param[key]
                self.vt[key]=self.beta2*self.tau**2+(1-self.beta2)*((self.model_param[key]-local_model_param[key])**2)

            else:
                self.mt[key]=self.beta1*self.mt[key]+(1-self.beta1)*(self.model_param[key]-local_model_param[key])
                self.vt[key]=self.beta2*self.vt[key]+(1-self.beta2)*((self.model_param[key]-local_model_param[key])**2)

            new_model_param[key] = local_model_param[key] + self.global_lr*self.mt[key]/(self.vt[key]**0.5+self.tau)

        model.load_state_dict(new_model_param)
        self.lr_multiplier /=(1+self.lr_decay)
        self.round_avg_weights = None
        return

class FedCMAggregator(FedAvgAggregator):
    def __init__(self,n_rounds,client_ratio,global_lr,alpha,global_lr_decay=0,seed=0,lr_decay=0,min_local_updates=None,fixed_local_updates=None,client_sampling='RandomReplace',log=False):
        super().__init__(n_rounds,client_ratio,seed=seed,lr_decay=lr_decay,min_local_updates=min_local_updates,
                    fixed_local_updates=fixed_local_updates,client_sampling=client_sampling,averaging='Uniform')
        self.global_lr=global_lr
        self.global_lr_decay=global_lr_decay
        self.alpha=alpha
        self.delta_t={}
        self.fl_adjustments = FLAdjustments(adjust_loss=False,adjust_grad=True,adjust_grad_type='FedCM',global_grad_w=(1-self.alpha),local_grad_w=self.alpha,reg_loss_w=0)
        self.log=log
    def add_local_model(self,model,worker_idx,sum_lr=None):
        #weight = len(self.workers_set[worker_idx].dataloader.dataset)/self.data_size
        weight = self.round_avg_weights[worker_idx]
        if(self.log):
            print("worker Idx: \n")
            print("sum lr:\n")
            print(sum_lr)
        local_model_param = model.state_dict()
        
        for key in local_model_param.keys():
            if(key not in self.model_param.keys()):
                self.model_param[key] = -(local_model_param[key]-self.pres_model_param[key]) * weight / sum_lr[key]
                if(self.log):
                    t=-(local_model_param[key]-self.pres_model_param[key]) * weight
                    log = str(key)+" Average param update:"+str(t.mean().item())+"\n"
                    log+=str(key)+" Average param update:"+str(t.max().item())+"\n"
                    log+=str(key)+" Average param update:"+str(t.min().item())+"\n"
                    print(log)
            else:
                self.model_param[key] -= (local_model_param[key]-self.pres_model_param[key]) * weight
                if(self.log):
                    t=-(local_model_param[key]-self.pres_model_param[key]) * weight 
                    log = str(key)+" Average param update:"+str(t.mean().item())+"\n"
                    log+=str(key)+" Average param update:"+str(t.max().item())+"\n"
                    log+=str(key)+" Average param update:"+str(t.min().item())+"\n"
                    print(log)        
        return
    
    def update_model(self,model):
        new_model_param = {}
        local_model_param = model.state_dict()
        for key in self.model_param.keys():
            new_model_param[key] = local_model_param[key]-self.global_lr*self.model_param[key]

        model.load_state_dict(new_model_param)
        self.fl_adjustments.global_grad ={}
        for key in self.model_param:
            self.fl_adjustments.global_grad[key]=torch.clone(self.model_param[key])
            if(self.log):
                log = str(key)+" Average global grad: "+str(self.model_param[key].mean().item())+"\n"
                log+= str(key)+" Max global grad: "+str(self.model_param[key].max().item())+"\n"
                log+= str(key)+" Min global grad: "+str(self.model_param[key].min().item())+"\n"
                print(log)


        self.lr_multiplier /=(1+self.lr_decay)
        self.global_lr /=(1+self.global_lr_decay)
        self.round_avg_weights = None
        return

class FedProx(FedAvgAggregator):
    def __init__(self,n_rounds,client_ratio,mu,seed=0,lr_decay=0,min_local_updates=None,fixed_local_updates=None,client_sampling='RandomReplace'):
        super().__init__(n_rounds,client_ratio,seed=seed,lr_decay=lr_decay,min_local_updates=min_local_updates,
                    fixed_local_updates=fixed_local_updates,client_sampling=client_sampling,averaging='Uniform')

        self.mu = mu
        self.fl_adjustments = FLAdjustments(adjust_loss=True,adjust_grad=False,reg_loss_w=self.mu)

    def start_round(self,model):
        
        self.fl_adjustments.prev_model = model.state_dict()
        return super().start_round(model)
