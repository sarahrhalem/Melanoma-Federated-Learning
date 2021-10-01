# Method to crop and resize image

import os
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import pandas as pd




def center_crop(image):
    width, height = image.size
    if width>height:
        x=int(np.floor((width-height)/2))
        return Image.fromarray(np.asarray(image)[:,x:width-x,:])
    elif width<height:
        x=int(np.floor((height-width)/2))
        return Image.fromarray(np.asarray(image)[x:height-x,:,:])
    else:
        return image
        
def resize_images(data, new_size):
    
    data_dir= os.path.join('Data')
    img_path = data_dir + "/jpeg/"+ str(data) + "/"
    output_dir= data_dir +"/jpeg_crop_resize_"+str(new_size)+"x"+str(new_size)+"/"+  str(data) + "/"
    os.makedirs(output_dir)
    
    metadata_name = str(data)+"_Metadata"
    metadata_name += "_Processed.csv" if (data=="train") else ".csv"
    metadata_df = pd.read_csv(data_dir + "/"+metadata_name)
    
    for name in metadata_df['image_name']:
        path = img_path+ str(name) +".jpg"
        image=Image.open(path)
        crop_image = center_crop(image)
        new_img= crop_image.resize((new_size, new_size))
        new_img.save(output_dir + name + ".jpg", quality=100, subsampling=0)
        
    return