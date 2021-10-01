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
from TrainVal import *

def test(model, test_loader, device):
    model.to(device)
    model.eval()
    test_correct_b=0
    all_labels=None
    all_test_preds=None
    
    with torch.no_grad():
            for batch_idx, (image, label) in tqdm(enumerate(test_loader), total=len(test_loader)):
                image= image.to(device, dtype=torch.float)
                label= label.to(device, dtype=torch.float)

                preds=model(image)
                labels = label.to('cpu').unsqueeze(1).numpy()
                
                test_preds = torch.sigmoid(preds.detach())

                test_correct_b+= (torch.round(test_preds).cpu().numpy()== labels).sum().item()

                if(batch_idx==0):
                    all_labels=labels
                    all_test_preds = test_preds.cpu().numpy()
                else:
                    all_labels   = np.vstack((all_labels,labels))
                    all_test_preds = np.vstack((all_test_preds,test_preds.cpu().numpy()))
                        
                torch.cuda.empty_cache()

    test_acc=test_correct_b/len(test_loader.dataset)
    test_roc_score = roc_auc_score(all_labels, all_test_preds)
    fpr, tpr, thresholds= roc_curve(all_labels, all_test_preds)
    test_cm= confusion_matrix(all_labels, np.round(abs(all_test_preds)))
        
    
    # Record results                         
                             
    test_results= {'test_accuracy':test_acc,'test_roc_score': test_roc_score,
                  'CM': test_cm}
    
    roc_curve_stat= {'fpr': fpr, 'tpr': tpr, 'thresholds': thresholds, 'AUC': test_roc_score}

    return test_results, roc_curve_stat