
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

class FLAdjustments:
    def __init__(self,adjust_loss=False,adjust_grad=False,adjust_grad_type=None,global_grad_w=0,local_grad_w=1,reg_loss_w=0):
        self.adjust_loss=adjust_loss
        self.adjust_grad=adjust_grad
        self.adjust_grad_type=adjust_grad_type
        self.prev_model=None
        self.global_grad=None
        self.global_grad_w=global_grad_w
        self.local_grad_w=local_grad_w
        self.reg_loss_w=reg_loss_w
        self.device = 'cpu'
    def reg_loss(self,model,device):
        if(self.adjust_loss):
            loss=0 
            for key in self.prev_model:
                 p = key.split(".")
                 loss+=(getattr(getattr(model,p[0]),p[1])-self.prev_model[key].to(device)).norm(2)**2

            loss*=self.reg_loss_w/2
            return loss
        else:    
            return 0

    def adjust_model_grad(self,model):
        if(self.global_grad is not None):
            for key in self.global_grad.keys():
                p = key.split(".")
                if(self.adjust_grad_type=='FedCM'):
                    getattr(getattr(model,p[0]),p[1]).grad*=self.local_grad_w
                    getattr(getattr(model,p[0]),p[1]).grad+=self.global_grad_w*self.global_grad[key].to(self.device)
        return

    