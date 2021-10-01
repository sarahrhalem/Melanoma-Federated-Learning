
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
from FedAvg import *



def fl_train_val(model, fl_workerset, val_loader, local_epochs, optimizer_name,learning_rate,criterion,
                 fl_aggregator, device, seed=5,worker_validation=False,checkpoint_name=None,display_worker_output=True,log_fedcm=False):
    set_seed(seed)
    start_training_time=time.time()
    
    train_loss_hist=[]
    train_acc_hist=[]
    
    val_loss_hist=[]
    val_acc_hist=[]
    val_roc_hist=[]
    
    w_val_loss_hist=[]
    w_val_acc_hist=[]
    w_val_roc_hist=[]
    
    fl_aggregator.reset(fl_workerset)
    model.to(device)
    rnd_idx = 0
    if(fl_aggregator.fl_adjustments is not None):
        fl_aggregator.fl_adjustments.device = device
    while rnd_idx<fl_aggregator.client_ratio_sampler.max_samples:
        
        model.to('cpu')
        round_workers = fl_aggregator.start_round(model)
        model.to(device)

        train_loss = 0
        train_acc  = 0
        
        w_val_loss=[]
        w_val_acc=[]
        w_val_roc=[]
        current_lr = learning_rate*fl_aggregator.lr_multiplier
        for worker_idx in round_workers:
            print(f' round:{rnd_idx}, worker: {worker_idx} Training')
            base_model = copy.deepcopy(model)
            epoch_ratio = fl_aggregator.epoch_ratio[worker_idx]
            train_res,sum_lr = fl_aggregator.workers_set[worker_idx].train(base_model,epoch_ratio*local_epochs,optimizer_name,
                                                                    current_lr,criterion,device,fl_aggregator.fl_adjustments,display_worker_output,log_fedcm)

            n_updates= epoch_ratio*local_epochs*len(fl_aggregator.workers_set[worker_idx].dataloader)
            base_model.to('cpu')                                                       
            fl_aggregator.add_local_model(base_model,worker_idx,sum_lr)
            base_model.to(device)
            train_loss+=train_res['train_loss'][-1]
            train_acc+=train_res['train_acc'][-1]
            if(worker_validation):
                try:
                    print(f' round:{rnd_idx}, worker: {worker_idx} Validation')
                    w_val_res = validate(base_model, val_loader, device,criterion)
                    w_val_loss.append(w_val_res['val_loss'])
                    w_val_acc.append(w_val_res['val_acc'])
                    w_val_roc.append(w_val_res['val_roc'])
                except ValueError:
                    if(checkpoint_name is not None):
                        performance_hist ={}
                        checkpoint = {
                            'state_dict': base_model.state_dict(),
                            'performance_history': performance_hist,
                            'epoch': len(fl_aggregator.rounds),
                        }
                        save_checkpoint(checkpoint, f"./TrainedEffNetcheckpoint_"+checkpoint_name+"_ERROR.pth.tar")
                    print("Training Stopped due to worker validation Error")
                    return model.state_dict()

        train_loss=train_loss/len(round_workers)
        train_acc=train_acc/len(round_workers)
        
        train_loss_hist.append(train_loss)
        train_acc_hist.append(train_acc)
        
        print(f' round: {rnd_idx}, avg train loss: {train_loss:.6f}, avg train acc: {train_acc:.4f}')            
        
        if(worker_validation):
            w_val_loss_hist.append(w_val_loss)
            w_val_acc_hist.append(w_val_acc)
            w_val_roc_hist.append(w_val_roc)
            avg_loss = sum(w_val_loss)/len(w_val_loss)
            avg_acc  = sum(w_val_acc)/len(w_val_acc)
            avg_roc  = sum(w_val_roc)/len(w_val_roc)
            print(f' round: {rnd_idx}, avg val loss: {avg_loss:.6f}, avg val acc: {avg_acc:.4f}, avg val roc:{avg_roc:.4f}')            
        model.to('cpu')
        fl_aggregator.update_model(model)
        model.to(device)
        print(f' round:{rnd_idx}, Center Validation')
        val_res = validate(model, val_loader, device,criterion)
    
        val_loss_hist.append(val_res['val_loss'])
        val_acc_hist.append(val_res['val_acc'])
        val_roc_hist.append(val_res['val_roc'])
        rnd_idx+=1
        
    performance_hist= {'avg_train_loss':train_loss_hist,'avg_train_acc':train_acc_hist,
                       'val_loss': val_loss_hist,'val_acc': val_acc_hist,'val_roc': val_roc_hist}
    
    if(worker_validation):
        performance_hist['avg_val_loss']=w_val_loss_hist
        performance_hist['avg_val_acc']=w_val_acc_hist
        performance_hist['avg_val_roc']=w_val_roc_hist
        
        # Save model checkpoint and results
    if(checkpoint_name is not None):
        checkpoint = {
            'state_dict': model.state_dict(),
            'performance_history': performance_hist,
            'epoch': len(fl_aggregator.rounds),
        }
        save_checkpoint(checkpoint, f"./TrainedEffNetcheckpoint_"+checkpoint_name+".pth.tar")
    print("Training Finished")

    end_training_time=time.time()
    print('Training and validation time:', end_training_time - start_training_time, 'seconds')
    return performance_hist



def validate(model, val_loader, device,criterion):
    
    model.to(device)
    model.eval()
    val_loss_b=0
    val_correct_b=0
    val_roc_score=0
    val_loss_epoch=0
    val_acc_epoch=0
    epoch_labels = None
    epoch_val_preds = None
    with torch.no_grad():
        for batch_idx, (image, label) in tqdm(enumerate(val_loader), total=len(val_loader)):
            image= image.to(device, dtype=torch.float)
            label= label.to(device, dtype=torch.float)


            preds=model(image)
            val_loss=criterion(preds, label.unsqueeze(1))

            labels = label.to('cpu').unsqueeze(1).numpy()
            val_preds = torch.sigmoid(preds.detach())

            val_correct_b+= (torch.round(val_preds).cpu().numpy()== labels).sum().item()

            if(batch_idx==0):
                epoch_labels=labels
                epoch_val_preds = val_preds.cpu().numpy()
            else:
                epoch_labels   = np.vstack((epoch_labels,labels))
                epoch_val_preds = np.vstack((epoch_val_preds,val_preds.cpu().numpy()))

            torch.cuda.empty_cache()
            val_loss_b+=val_loss.item()

    val_loss_epoch = val_loss_b/len(val_loader)
    val_acc_epoch=val_correct_b/len(val_loader.dataset)
    val_roc_score = roc_auc_score(epoch_labels, epoch_val_preds)
    print(f' val loss: {val_loss_epoch:.6f}, val acc: {val_acc_epoch:.4f}, val roc:{val_roc_score:.4f}')            
        
    return {'val_loss':val_loss_epoch,'val_acc':val_acc_epoch,'val_roc':val_roc_score}