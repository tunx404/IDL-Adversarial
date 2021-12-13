#!/usr/bin/env python
# coding: utf-8

# # Necessary files

# In[ ]:


from google.colab import drive
drive.mount('/content/drive/')


# **/!\ Check the path to the main directory**
# 
# Here I put the **IDL_group_project** directory at the root of the Google Drive

# In[ ]:


# Check everytime
# cd to the main directory
get_ipython().run_line_magic('cd', "'/content/drive/MyDrive/IDL_group_project/Google Colab/'")

data_dir = '/content/' # Google Colab
# data_dir = './' # Local Jupyter


# ### PASCAL VOC2010

# In[ ]:


# # cd to the correct PASCAL VOC2010 dataset location, which should contain tar files
# %cd 'data/PASCAL VOC2010/'
# !ls

# # Extract the data to /content/data/

# !mkdir '/content/data'
# !mkdir '/content/data/PASCAL VOC2010'

# !mkdir '/content/data/PASCAL VOC2010/VOCtrainval_03-May-2010' # Train & val datasets
# !tar -xf VOCtrainval_03-May-2010.tar -C '/content/data/PASCAL VOC2010/VOCtrainval_03-May-2010' --skip-old-files # Train & val datasets

# # !mkdir '/content/PASCAL VOC2010/download' # Test dataset
# # !tar -xf download.tar -C '/content/data/PASCAL VOC2010/download' --skip-old-files # Test dataset

# # !mkdir '/content/PASCAL VOC2010/VOCdevkit_08-May-2010' # Optional
# # !tar -xf VOCdevkit_08-May-2010.tar -C '/content/data/PASCAL VOC2010/VOCdevkit_08-May-2010' --skip-old-files # Optional

# # cd back to the main working directory, all paths are referenced to this directory
# %cd ../../
# !ls


# ### PASCAL-Part

# In[ ]:


# cd to the correct PASCAL-Part dataset location, which should contain tar.gz file
get_ipython().run_line_magic('cd', "'data/PASCAL-Part/'")
get_ipython().system('ls')

# Extract the data to /content/data/

get_ipython().system("mkdir '/content/data'")
get_ipython().system("mkdir '/content/data/PASCAL-Part'")

get_ipython().system("mkdir '/content/data/PASCAL-Part/trainval'")
get_ipython().system("tar -xzf trainval.tar.gz -C '/content/data/PASCAL-Part/trainval' --skip-old-files")

# cd back to the main working directory, all paths are referenced to this directory
get_ipython().run_line_magic('cd', '../../')
get_ipython().system('ls')


# # Import

# In[ ]:


import os
import sys
import time
import csv

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

import torchvision
from torchvision import transforms
import torchvision.models as models

# !pip install torchinfo
# from torchinfo import summary

import numpy as np

from PIL import Image

import matplotlib.pyplot as plt

# !pip install adversarial-robustness-toolbox
# from art.estimators.classification import PyTorchClassifier
# from art.attacks.evasion import FastGradientMethod

sys.path.append('src')
from art_attack import test_art
from dataset import normalize, inverse_normalize, create_file_list, create_file_and_label_list, VOCImageFolder, visualize_images
from model import VOCDetector, train_detector, test_detector, write_csv_header, write_csv_params
from utilities import read_PASCAL_Part_to_dict


# In[ ]:


print('Python version  ' + sys.version)
print('PyTorch version ' + torch.__version__)
print('Numpy version   ' + np.__version__)


# # Config

# In[ ]:


category = 'car' # <---
categories = ['aeroplane', 'bicycle', 'boat', 'bus', 'car', 'motorbike', 'train']

get_ipython().system("mkdir 'data/PASCAL-Part-anno-pkl'")
get_ipython().system("mkdir 'data/PASCAL-Part-anno-pkl/aeroplane'")
get_ipython().system("mkdir 'data/PASCAL-Part-anno-pkl/bicycle'")
get_ipython().system("mkdir 'data/PASCAL-Part-anno-pkl/boat'")
get_ipython().system("mkdir 'data/PASCAL-Part-anno-pkl/bus'")
get_ipython().system("mkdir 'data/PASCAL-Part-anno-pkl/car'")
get_ipython().system("mkdir 'data/PASCAL-Part-anno-pkl/motorbike'")
get_ipython().system("mkdir 'data/PASCAL-Part-anno-pkl/train'")


# In[ ]:


device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} device'.format(device))
if device == 'cuda':
    print(torch.cuda.get_device_name(0))


# # Dataset

# ### PASCAL VOC2010

# Lists of filenames and labels

# In[ ]:


PASCAL_label_dir = data_dir + 'data/PASCAL VOC2010/VOCtrainval_03-May-2010/VOCdevkit/VOC2010/ImageSets/Main/'
PASCAL_image_dir = data_dir + 'data/PASCAL VOC2010/VOCtrainval_03-May-2010/VOCdevkit/VOC2010/JPEGImages/'

print(PASCAL_label_dir)
print(PASCAL_image_dir)


# ### PASCAL-Part

# In[ ]:


PASCAL_Part_dir = data_dir + 'data/PASCAL-Part/trainval/Annotations_Part/'

PASCAL_Part_file_name_list, PASCAL_Part_file_path_list = create_file_list(PASCAL_Part_dir)

print(PASCAL_Part_file_name_list[:5])
print(PASCAL_Part_file_path_list[:5])


# In[ ]:


index = PASCAL_Part_file_name_list.index('2008_000075.mat')
image_anno_dict = read_PASCAL_Part_to_dict(PASCAL_Part_file_path_list[index])

print(image_anno_dict['image_name'])
for i in range(image_anno_dict['num_objects']):
    obj = image_anno_dict['objects_list'][i]
    print(f'Object #{i}:')
    print('\tClass: ' + obj['obj_class'])
    print('\tMask: ' + str(obj['obj_mask'].shape))
    for j in range(obj['num_parts']):
        part = obj['parts_list'][j]
        print(f'\tPart #{j}:')
        print('\t\tPart name: ' + part['part_name'])
        print('\t\tMask: ' + str(part['part_mask'].shape))


# In[ ]:


import pickle

save_dir = 'data/PASCAL-Part-anno-pkl/'
count = 0
total_file = len(PASCAL_Part_file_path_list)
for file_path in PASCAL_Part_file_path_list:
    image_anno_dict = read_PASCAL_Part_to_dict(file_path)
    for i in range(image_anno_dict['num_objects']):
        obj = image_anno_dict['objects_list'][i]
        category = obj['obj_class']
        if category in categories:
            save_path = save_dir + category + '/' + image_anno_dict['image_name'] + '.pkl'
            pkl_file = open(save_path, 'wb')
            pickle.dump(image_anno_dict, pkl_file)
            pkl_file.close()
    count += 1
    if count%100 == 0:
        print(f'Processed {count}/{total_file} files!')


# In[ ]:


pkl_file = open(save_path, 'rb')
image_sample = pickle.load(pkl_file)

print(image_sample['image_name'])
for i in range(image_sample['num_objects']):
    obj = image_sample['objects_list'][i]
    print(f'Object #{i}:')
    print('\tClass: ' + obj['obj_class'])
    print('\tMask: ' + str(obj['obj_mask'].shape))
    for j in range(obj['num_parts']):
        part = obj['parts_list'][j]
        print(f'\tPart #{j}:')
        print('\t\tPart name: ' + part['part_name'])
        print('\t\tMask: ' + str(part['part_mask'].shape))

