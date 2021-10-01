
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms.functional as tf
from torch.utils.data import Dataset




class FocalLoss(nn.Module):
    def __init__(self,alpha=(0.5,0.5),gamma=2):
        super(FocalLoss,self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        
    def forward(self,preds,label):
        p = torch.sigmoid(preds)
        p=torch.clip(p,1e-6,1-1e-6)
        fc_loss = -self.alpha[0]*((1-p)**self.gamma)*torch.log(p)*label
        fc_loss = fc_loss -(1-self.alpha[1])*(p**self.gamma)*torch.log(1-p)*(1-label)
        
        return 2*torch.mean(fc_loss)