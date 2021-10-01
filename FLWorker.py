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
from FLScenario import *
from FLutils import *
from FLAdjustment import *

def create_flworker_set(train_loader,n_workers,fl_scenario,local_batchsize):
    dataloaders = fl_scenario.split(train_loader,n_workers,local_batchsize)
    worker_set = []
    for dataloader in dataloaders:
            worker_set+=[FLWorker(dataloader)]
    return worker_set

class FLWorker:
    def __init__(self,dataloader):
        self.dataloader = dataloader
        self.fixed_local_updates = None

    def features_label_dist(self):
        features = ['sex', 'age_approx','anatom_site_general_challenge']
        
        res=[metadata_proba(self.dataloader.dataset.metadata_df,'sex','female')]
        res+=[self.dataloader.dataset.metadata_df['age_approx'].mean(),
                          self.dataloader.dataset.metadata_df['age_approx'].std()]
        
        all_anatom = ['upper extremity', 'lower extremity', 'torso', 'unknown','head/neck', 'palms/soles', 'oral/genital']
        
        for anatom in all_anatom:  
            p=metadata_proba(self.dataloader.dataset.metadata_df,'anatom_site_general_challenge',anatom)
            res.append(p)
            
        res.append(metadata_proba(self.dataloader.dataset.metadata_df,'target',1))
        res.append(len(self.dataloader.dataset.metadata_df['patient_id'].unique()))
        col_name = ['sex (female)', 'age_approx (mean)','age_approx (std)']+all_anatom+['target (1)','Total patients']
        res_df = pd.DataFrame([res],columns=col_name)
        return res_df
    
    def set_local_updates(self,fixed_local_updates=None):
        self.fixed_local_updates = fixed_local_updates
        return
    
    def train(self,model,epochs,optimizer_name,learning_rate,criterion,device,fl_adjusments=None,display_output = False,log_fedcm=False):
    
        if(optimizer_name=="Adam"):
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        else:
            optimizer = optim.SGD(model.parameters(), lr=learning_rate)
        
        start_training_time=time.time()

        train_loss_hist=[]
        train_acc_hist=[]
        l_updates = 0
        model.to(device)
        
        sum_lr = {}
        for e in range(epochs):
            train_loss_b=0
            train_correct_b=0
            model.train()
            train_loss_epoch=0
            train_acc_epoch=0
            for batch_idx, (image, label) in tqdm(enumerate(self.dataloader), total=len(self.dataloader),disable=not display_output):
                if((self.fixed_local_updates is not None)and (l_updates>self.fixed_local_updates)):
                    break
                l_updates+=1
                image= image.to(device, dtype=torch.float)
                label= label.to(device, dtype=torch.float)
                model.zero_grad()
                
                preds=model(image)
                train_loss=criterion(preds, label.unsqueeze(1))
                if((fl_adjusments is not None) and fl_adjusments.adjust_loss):
                    train_loss+=fl_adjusments.reg_loss(model,device)

                if(True in torch.isnan(train_loss)):
                    performance_hist ={}
                    checkpoint = {
                        'state_dict': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'performance_history': performance_hist,
                        'epoch': e,
                    }
                    save_checkpoint(checkpoint, f"./TrainedEffNetcheckpoint_ERROR.pth.tar")
                    print("Training Stopped due to worker loss nan before training")
                    raise ValueError
                pred_labels = torch.sigmoid(preds.detach())
                labels = label.unsqueeze(1).to('cpu').numpy()
                train_correct_b+= (torch.round(pred_labels).cpu().numpy()== labels).sum().item()
                train_loss_b+=train_loss.item()

                optimizer.zero_grad()
                train_loss.backward()

                
                current_param = copy.deepcopy(model).state_dict()

                # Apply gradient clipping
                if((fl_adjusments is not None) and fl_adjusments.adjust_grad):
                    if(log_fedcm):
                        if((e==0)and(batch_idx==0)):
                            for key in model.state_dict().keys():
                                p = key.split(".")

                                log = str(key)+" Average worker grad: "+str(getattr(getattr(model,p[0]),p[1]).grad.mean().item())+"\n"
                                log+= str(key)+" Max worker grad: "+str(getattr(getattr(model,p[0]),p[1]).grad.max().item())+"\n"
                                log+= str(key)+" Min worker grad: "+str(getattr(getattr(model,p[0]),p[1]).grad.min().item())+"\n"
                                print(log)

                        if((e==epochs-1)and(batch_idx==len(self.dataloader)-1)):
                            for key in model.state_dict().keys():
                                p = key.split(".")

                                log = str(key)+" Average worker grad: "+str(getattr(getattr(model,p[0]),p[1]).grad.mean().item())+"\n"
                                log+= str(key)+" Max worker grad: "+str(getattr(getattr(model,p[0]),p[1]).grad.max().item())+"\n"
                                log+= str(key)+" Min worker grad: "+str(getattr(getattr(model,p[0]),p[1]).grad.min().item())+"\n"
                                print(log)

                    fl_adjusments.adjust_model_grad(model)
                
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                if((fl_adjusments is not None) and fl_adjusments.adjust_grad):
                    
                    for key in model.state_dict().keys():
                        p = key.split(".")
                        gradient = getattr(getattr(model,p[0]),p[1]).grad.cpu()
                        sign = gradient.sign()
                        gradient = gradient.abs_().clamp_(min=1e-8)
                        gradient *= sign
                        if(key in sum_lr.keys()):
                            sum_lr[key]+=(-(getattr(getattr(model,p[0]),p[1]).cpu() - current_param[key].cpu())/gradient).clamp(min=0.0001*learning_rate)
                        else:
                            sum_lr[key]=(-(getattr(getattr(model,p[0]),p[1]).cpu() - current_param[key].cpu())/gradient).clamp(min=0.0001*learning_rate)

                torch.cuda.empty_cache()

            train_loss_epoch=train_loss_b/len(self.dataloader)
            train_acc_epoch=train_correct_b/len(self.dataloader.dataset)
            train_loss_hist.append(train_loss_epoch)
            train_acc_hist.append(train_acc_epoch)
            if(display_output):
                print(f' epoch: {e}, train loss: {train_loss_epoch:.6f}, train acc: {train_acc_epoch:.4f}')            

        # Record results
        performance_hist= {'train_loss': train_loss_hist,'train_acc':train_acc_hist,'num_epochs': epochs}
        if(display_output):
            print("Training Finished")

        end_training_time=time.time()
        if(display_output):
            print('Training:', end_training_time - start_training_time, 'seconds')
        return performance_hist,sum_lr



def workers_feature_label_dist(workerset,add_distances=False):
    dist = [worker.features_label_dist() for worker in workerset]
    res = pd.concat(dist)
    res.reset_index(inplace=True,drop=True)
    return res