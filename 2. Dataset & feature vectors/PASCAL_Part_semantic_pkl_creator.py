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
get_ipython().run_line_magic('cd', "'/content/drive/MyDrive/IDL_group_project/Google_Colab/'")

data_dir = '/content/' # Google Colab
# data_dir = './' # Local Jupyter


# # Import

# In[ ]:


import os
import sys

import numpy as np

from PIL import Image

import matplotlib.pyplot as plt

import pickle

# sys.path.append('src')
# from dataset import create_file_list


# In[ ]:



# Create a list of all files within a directory
def create_file_list(dir):
    file_name_list = os.listdir(dir)
    from tkinter import Tcl
    file_name_list = list(Tcl().call('lsort', '-dict', file_name_list)) # Sort by filename
    file_path_list = file_name_list.copy()
    for i in range(len(file_path_list)):
        file_path_list[i] = os.path.join(dir, file_name_list[i])
    return file_name_list, file_path_list


# # Config

# In[ ]:


category = 'bus' # <---
categories = ['aeroplane', 'bicycle', 'bus', 'car', 'motorbike', 'train']

get_ipython().system("mkdir 'data/PASCAL-Part-semantic-pkl'")
get_ipython().system("mkdir 'data/PASCAL-Part-semantic-pkl/aeroplane'")
get_ipython().system("mkdir 'data/PASCAL-Part-semantic-pkl/bicycle'")
get_ipython().system("mkdir 'data/PASCAL-Part-semantic-pkl/bus'")
get_ipython().system("mkdir 'data/PASCAL-Part-semantic-pkl/car'")
get_ipython().system("mkdir 'data/PASCAL-Part-semantic-pkl/motorbike'")
get_ipython().system("mkdir 'data/PASCAL-Part-semantic-pkl/train'")


# # Dataset

# ### PASCAL VOC2010

# Lists of filenames and labels

# In[ ]:


PASCAL_label_dir = data_dir + 'data/PASCAL VOC2010/VOCtrainval_03-May-2010/VOCdevkit/VOC2010/ImageSets/Main/'
PASCAL_image_dir = data_dir + 'data/PASCAL VOC2010/VOCtrainval_03-May-2010/VOCdevkit/VOC2010/JPEGImages/'

print(PASCAL_label_dir)
print(PASCAL_image_dir)


# ### PASCAL-Part

# Read all pkl files of a specific category

# In[ ]:


for category in categories:
    PASCAL_Part_pkl_dir = 'data/PASCAL-Part-anno-pkl/' + category
    image_anno_file_name_list, image_anno_file_path_list = create_file_list(PASCAL_Part_pkl_dir)

    print('Processing ' + category + '!')
    print('Total ' + category + ' images: ' + str(len(image_anno_file_name_list)))
    print('First 5 samples:')
    print(image_anno_file_name_list[:5])
    print(image_anno_file_path_list[:5])

    file_count = 0
    semantic_count = 0
    save_dir = 'data/PASCAL-Part-semantic-pkl/'
    total_file = len(image_anno_file_name_list)
    for i in range(len(image_anno_file_name_list)):
        # Load pickle file
        pkl_file = open(image_anno_file_path_list[i], 'rb')
        image_sample = pickle.load(pkl_file)
        file_count += 1

        for i in range(image_sample['num_objects']):
            obj = image_sample['objects_list'][i]
            if obj['obj_class'] == category:
                for j in range(obj['num_parts']):
                    semantic_count += 1
                    part = obj['parts_list'][j]
                    semantic_dict = {'image_name': image_sample['image_name'], 'label': obj['obj_class'], 'semantic_name': part['part_name'], 'semantic_mask': part['part_mask']}
                    save_path = save_dir + category + '/' + image_sample['image_name'] + '_obj' + str(i) + '_part' + str(j) + '.pkl'
                    pkl_file = open(save_path, 'wb')
                    pickle.dump(semantic_dict, pkl_file)
                    pkl_file.close()

        if file_count%100 == 0:
            print(f'Processed {file_count}/{total_file} files, containing {semantic_count} semantic parts!')
            # break

    print(f'Processed {file_count}/{total_file} files, containing {semantic_count} semantic parts!')
    # break


# <!-- The output is a dictionary, access data as the following example -->
