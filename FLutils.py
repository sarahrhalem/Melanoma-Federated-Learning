
import os
import sys
import numpy as np
import pandas as pd
import random
import sklearn
import torch
import torchvision

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

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.metrics import  confusion_matrix, roc_curve
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans


def metadata_proba(df,metadata,value):
    md_values = df[metadata].unique()
    if(value in md_values):
        return df[metadata].value_counts()[value]/len(df[metadata])
    else:
        return 0

    
def w_numeric_metadata(worker,anatom_site=True):
    new_df = worker.dataloader.dataset.metadata_df.copy()
    
    tgt1_patients = list(new_df[new_df['target']==1]['patient_id'].unique())
    
    sex_list = ['female','male']
    as_list = ['head/neck','lower extremity','oral/genital','palms/soles','torso','unknown','upper extremity']
    
    new_df['sex_digit'] = new_df['sex'].apply(lambda x: sex_list.index(x))
    new_df['as_digit'] = new_df['anatom_site_general_challenge'].apply(lambda x: as_list.index(x))
    new_df['p_target'] = new_df['patient_id'].apply(lambda x: 1 if x in tgt1_patients else 0)
    
    if(anatom_site):
        new_df = new_df[['sex_digit','age_approx','as_digit','p_target']]
    else:
        new_df = new_df[['sex_digit','age_approx','p_target']]
        
    new_df.drop_duplicates(inplace=True)
    return new_df

def w_metadata_X_y(worker_split,anatom_site=True):
    X=np.array(w_numeric_metadata(worker_split[0],anatom_site))
    y=np.array([0]*X.shape[0])
    for idx in range(1,len(worker_split)):
        new_X = np.array(w_numeric_metadata(worker_split[idx],anatom_site))
        new_y = np.array([idx]*new_X.shape[0])
        X=np.vstack((X,new_X))
        y=np.hstack((y,new_y))
    
    return X,y
    
def ws_silhouette_score(worker_split,anatom_site=True):
    
    X,y = w_metadata_X_y(worker_split,anatom_site)
    s_score = silhouette_score(X,y)
    return s_score

def feature_label_dist(metadata_df):
    total_data = metadata_df.shape[0]
    
    f_p = metadata_df[metadata_df['sex']=='female'].shape[0]/total_data
    sex_dist = [f_p,1-f_p]
    
    as_list = ['head/neck','lower extremity','oral/genital','palms/soles','torso','unknown','upper extremity']
    as_dist = [metadata_df[metadata_df['anatom_site_general_challenge']==a].shape[0]/total_data for a in as_list]
    
    tgt1_p = metadata_df[metadata_df['target']==1].shape[0]/total_data
    target_dist = [tgt1_p,1-tgt1_p]
    
    age_list = [10,20,30,40,50,60,70,80,90]
    age_dist = [metadata_df[(metadata_df['age_approx']>age_list[idx-1]) &(metadata_df['age_approx']<=age_list[idx])].shape[0]/total_data for idx in range(1,len(age_list))]
    age_dist = [1-sum(age_dist)]+age_dist
    
    res = {'sex_dist':sex_dist,
          'age_dist':age_dist,
          'anatom_dist':as_dist,
           'target_dist':target_dist}
    return res
def ws_jensen_shanon_distance(train_loader,worker_split,output='workers_distances'):
    train_dist = feature_label_dist(train_loader.dataset.metadata_df)
    res = {}
    for key in train_dist:
        res[key]=[]
        
    for worker in worker_split:
        w_dist = feature_label_dist(worker.dataloader.dataset.metadata_df)
        for key in train_dist:
            res[key]+=[jensen_shanon_discreate(train_dist[key],w_dist[key])]
    if(output=='workers_distances'):
        return pd.DataFrame(res)
    
    else:
        for key in train_dist:
            res[key]=max(res[key])
        return res

def jensen_shanon_discreate(p,q):
    d = 0
    for idx in range(len(p)):
        p_i=max(min(p[idx],0.9999),0.0001)
        q_i=max(min(q[idx],0.9999),0.0001)
        m_i=0.5*(p_i+q_i)
        d+=0.5*(p_i*np.log(p_i/m_i)+q_i*np.log(q_i/m_i))
        
    return d/np.log(2)