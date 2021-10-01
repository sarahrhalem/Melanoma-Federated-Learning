import os
import sys
import numpy as np
import pandas as pd
import time
import pickle
import random
import sklearn


from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.metrics import  confusion_matrix, roc_curve

import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms.functional as tf
from torch.utils.data import Dataset
from torchvision import transforms
import albumentations as A
from torch.cuda import amp
from tqdm.notebook import tqdm

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

from utils import *
from MelanomaDataset import *
from MelanomaEfficientNet import *






def train_val_clip(model, train_loader, val_loader, epochs, optimizer, criterion, device, seed=5, checkpoint_name=None,clip_grad=True):
    set_seed(seed)
    start_training_time=time.time()
    
    train_loss_hist=[]
    train_acc_hist=[]
    
    val_loss_hist=[]
    val_acc_hist=[]
    val_roc_hist=[]
    model.to(device)
    for e in range(epochs):
        train_loss_b=0
        train_correct_b=0
        model.train()
        train_loss_epoch=0
        train_acc_epoch=0
        for batch_idx, (image, label) in tqdm(enumerate(train_loader), total=len(train_loader)):
            image= image.to(device, dtype=torch.float)
            label= label.to(device, dtype=torch.float)
            model.zero_grad()
            

            preds=model(image)

            train_loss=criterion(preds, label.unsqueeze(1))
            pred_labels = torch.sigmoid(preds.detach())
            labels = label.unsqueeze(1).to('cpu').numpy()
            train_correct_b+= (torch.round(pred_labels).cpu().numpy()== labels).sum().item()
            train_loss_b+=train_loss.item()
                
            optimizer.zero_grad()
            train_loss.backward()
            # Apply gradient clipping
            if(clip_grad):
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()

            torch.cuda.empty_cache()
        
        train_loss_epoch=train_loss_b/len(train_loader)
        train_acc_epoch=train_correct_b/len(train_loader.dataset)
        train_loss_hist.append(train_loss_epoch)
        train_acc_hist.append(train_acc_epoch)
    
    
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


                labels = label.unsqueeze(1).to('cpu').numpy()
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
        
        val_loss_hist.append(val_loss_epoch)
        val_acc_hist.append(val_acc_epoch)
        val_roc_hist.append(val_roc_score)
    
        print(f' epoch: {e}, train loss: {train_loss_epoch:.6f}, val loss: {val_loss_epoch:.6f}, train acc: {train_acc_epoch:.4f}, val acc: {val_acc_epoch:.4f}, val roc:{val_roc_score:.4f}')            

        # Record results
        performance_hist= {'train_loss': train_loss_hist, 'val_loss': val_loss_hist,
                           'train_acc':train_acc_hist,'val_acc': val_acc_hist,
                           'val_roc': val_roc_hist, 'num_epochs': epochs}

        # Save model checkpoint and results
    if(checkpoint_name is not None):
        checkpoint = {
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'performance_history': performance_hist,
            'epoch': epochs,
        }
        save_checkpoint(checkpoint, f"./TrainedEffNetcheckpoint_"+checkpoint_name+".pth.tar")
    print("Training Finished")

    end_training_time=time.time()
    print('Training and validation time:', end_training_time - start_training_time, 'seconds')
    return performance_hist