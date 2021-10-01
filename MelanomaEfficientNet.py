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
import matplotlib
import matplotlib.pyplot as plt
from efficientnet_pytorch import EfficientNet
from MelanomaDataset import *



class MelanomaEfficientNet(nn.Module):
    def __init__(self, model='efficientnet-b0',en_trained_layers=0):
        super(MelanomaEfficientNet, self).__init__()
        self.model= model
        
        # Pretrained Architecture
        self.backbone= EfficientNet.from_pretrained(model)
    
        # Modify fully connected layer
        
        in_features=getattr(self.backbone, '_fc').in_features
        self.backbone._fc= nn.Linear(in_features, out_features=512, bias=True)
        self.fc2=nn.Linear(512, 128)
        self.output=nn.Linear(128, 1)
        
        # Freeze layers except fully connected layer
        for param in self.backbone.parameters():
            param.requires_grad=False
        if(en_trained_layers!=0):  # If en_trained_layers is 1 then we unfreeze the last block of the backbone for training
            for param in self.backbone._blocks[-1].parameters():
                param.requires_grad=True

        for param in self.backbone._fc.parameters():
            param.requires_grad=True
    
    def load_state_dict(self,state_dict,strict=True):
        super(MelanomaEfficientNet, self).load_state_dict(state_dict,strict)
        # Freeze layers except fully connected layer
        for param in self.backbone.parameters():
            param.requires_grad=False

        for param in self.backbone._fc.parameters():
            param.requires_grad=True
        
        return
    
    def forward(self, image):
        X= image
        X= self.backbone(X)
        X= self.fc2(X)
        X= self.output(X)
        return X


class MelanomaEfficientNetDropout(nn.Module):
    def __init__(self, model='efficientnet-b0', dp=0.2):
        super(MelanomaEfficientNetDropout, self).__init__()
        self.model= model
        
        
        # Pretrained Architecture
        self.backbone= EfficientNet.from_pretrained(model)
    
        # Modify fully connected layer
        
        in_features=getattr(self.backbone, '_fc').in_features
        self.backbone._fc= nn.Linear(in_features, out_features=512, bias=True)
        self.dropout1=nn.Dropout(p=dp)
        self.fc2=nn.Linear(512, 128)
        self.dropout2=nn.Dropout(p=dp)
        self.output=nn.Linear(128, 1)
        
        # Freeze layers except fully connected layer
        for param in self.backbone.parameters():
            param.requires_grad=False

        for param in self.backbone._fc.parameters():
            param.requires_grad=True
        
    def load_state_dict(self,state_dict,strict=True):
        super(MelanomaEfficientNetDropout, self).load_state_dict(state_dict,strict)
        # Freeze layers except fully connected layer
        for param in self.backbone.parameters():
            param.requires_grad=False

        for param in self.backbone._fc.parameters():
            param.requires_grad=True
        
        return

    def forward(self, image):
        X= image
        X= self.backbone(X)
        X= self.dropout1(X)
        X= self.fc2(X)
        X= self.dropout2(X)
        X= self.output(X)
        return X


class MelanomaEfficientNetSingle(nn.Module):
    def __init__(self, model='efficientnet-b0'):
        super(MelanomaEfficientNetSingle, self).__init__()
        self.model= model
        
        # Pretrained Architecture
        self.backbone= EfficientNet.from_pretrained(model)
    
        # Modify fully connected layer
        
        in_features=getattr(self.backbone, '_fc').in_features
        self.backbone._fc= nn.Linear(in_features, out_features=1, bias=True)
        
        # Freeze layers except fully connected layer
        for param in self.backbone.parameters():
            param.requires_grad=False

        for param in self.backbone._fc.parameters():
            param.requires_grad=True
    
    def load_state_dict(self,state_dict,strict=True):
        super(MelanomaEfficientNetSingle, self).load_state_dict(state_dict,strict)
        # Freeze layers except fully connected layer
        for param in self.backbone.parameters():
            param.requires_grad=False

        for param in self.backbone._fc.parameters():
            param.requires_grad=True
        
        return
    
    def forward(self, image):
        X= image
        X= self.backbone(X)
        return X

class SimpleLinear(nn.Module):
    def __init__(self):
        super(SimpleLinear, self).__init__()
        self.fc1= nn.Linear(1280, out_features=1, bias=True)
    
    def forward(self, image):
        X= image
        
        X= self.fc1(X)
        return X

class DoubleLinear(nn.Module):
    def __init__(self):
        super(DoubleLinear, self).__init__()  
        self.fc1= nn.Linear(1280, out_features=512, bias=True)
        self.fc2= nn.Linear(512, out_features=1, bias=True)
    
    def forward(self, image):
        X= image        
        X= self.fc1(X)
        X= self.fc2(X)
        return X

class DoubleLinearSigmoid(nn.Module):
    def __init__(self,n1=512):
        super(DoubleLinearSigmoid, self).__init__()  
        self.fc1= nn.Linear(1280, out_features=n1, bias=True)
        self.fc2= nn.Linear(n1, out_features=1, bias=True)

    def forward(self, image):
        X= image        
        X= self.fc1(X)
        X=torch.sigmoid(X)
        X= self.fc2(X)
        return X

class TripleLinear(nn.Module):
    def __init__(self):
        super(TripleLinear, self).__init__()  
        self.fc1= nn.Linear(1280, out_features=512, bias=True)
        self.fc2= nn.Linear(512, out_features=128, bias=True)
        self.fc3= nn.Linear(128, out_features=1, bias=True)

    def forward(self, image):
        X= image        
        X= self.fc1(X)
        X= self.fc2(X)
        X= self.fc3(X)
        return X

class TripleLinearSigmoid(nn.Module):
    def __init__(self,n1=512,n2=256):
        super(TripleLinearSigmoid, self).__init__()  
        self.fc1= nn.Linear(1280, out_features=n1, bias=True)
        self.fc2= nn.Linear(n1, out_features=n2, bias=True)
        self.fc3= nn.Linear(n2, out_features=1, bias=True)

    def forward(self, image):
        X= image        
        X= self.fc1(X)
        X=torch.sigmoid(X)
        X= self.fc2(X)
        X=torch.sigmoid(X)
        X= self.fc3(X)
        return X

def get_efficientnet_feature_extractor(fe_name='efficientnet-b0'):
    feature_extractor = EfficientNet.from_pretrained(fe_name)

    in_features=getattr(feature_extractor, '_fc').in_features


    feature_extractor._fc= nn.Linear(in_features, out_features=in_features, bias=False)

    for param in feature_extractor.parameters():
        param.requires_grad=False

    feature_extractor._fc.weight.fill_(0)
    for idx in range(in_features):
        feature_extractor._fc.weight[idx][idx]=1
    
    return feature_extractor