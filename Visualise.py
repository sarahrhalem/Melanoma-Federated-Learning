# Methods to visualise Melanoma images or handle their attributes
import os
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.image as mpimg
from PIL import Image
import numpy as np
import pandas as pd
import torch
import torchvision



def img_label(df):
    cols = df.columns
    label = ''
    for col in cols:
        label += col+': '+str(df[col].to_list()[0])+'\n'
    return label


def visualise_jpeg (data_source,data, n_images=1,seed=0, target=0, metadata_processed=False):

    data_dir= os.path.join('Data')
    metadata_dir = data_dir + "/"+str(data)+"_Metadata_Processed.csv" if(metadata_processed==True) else data_dir + "/"+str(data)+"_Metadata.csv"
    metadata_df = pd.read_csv(metadata_dir)
    img_path = data_dir + "/"+data_source+"/"+ str(data) + "/"
    
    rng = np.random.default_rng(seed=seed)
    
    image_names = rng.choice( metadata_df[metadata_df["target"]==target]['image_name'],n_images,replace=False)    
        
    for name in image_names:
        path = img_path+ str(name) + ".jpg"
        image=mpimg.imread(path)
        image_details= img_label( metadata_df[metadata_df['image_name']==name])
        image_title= ("Data source path: " + str(data_source)+ "/" + str(data))
        plt.imshow(image)
        plt.title(image_title)
        plt.ylabel(image_details, rotation=360, labelpad=300, fontsize=12, loc="bottom")
        plt.show()
        
    return


def get_image_size(data):
    data_dir= os.path.join('Data')
    metadata_df = pd.read_csv(data_dir + "/"+str(data)+"_Metadata.csv")
    img_path = data_dir + "/jpeg/"+ str(data) + "/"
    
    res = []
    for name in metadata_df['image_name']:
        path = img_path+ str(name) + ".jpg"
        image = Image.open(path)
        width, height = image.size
        res += [[name,width,height]]
    res_df = pd.DataFrame(res,columns=['image_name','width','height'])
    return res_df

def get_mean_rgbs(data_source,data,metadata_processed=False):
    data_dir= os.path.join('Data')
    metadata_dir = data_dir + "/"+str(data)+"_Metadata_Processed.csv" if(metadata_processed==True) else data_dir + "/"+str(data)+"_Metadata.csv"
    metadata_df = pd.read_csv(metadata_dir)
    img_path = data_dir + "/"+data_source+"/"+ str(data) + "/"
    
    res = []
    
    for name in metadata_df['image_name']:
        path = img_path+ str(name) + ".jpg"
        image = Image.open(path)
        rgbs = np.asarray(image)
        res += [[name,np.mean(rgbs[:,:,0]),np.mean(rgbs[:,:,1]),np.mean(rgbs[:,:,2])]]
        
    res_df = pd.DataFrame(res,columns=['image_name','mean_red','mean_green','mean_blue'])
    return res_df

# Function to map tensor values to visualize Normalized RGB image 
def vis_map_normalize(tensor):
    min_val= tensor.min()
    max_val= tensor.max()
    min_map = 0
    max_map=1
    return min_map + (max_map - min_map) * ((tensor - min_val) / (max_val - min_val))