#!/usr/bin/env python
# coding: utf-8

# # Necessary files

# In[ ]:


from google.colab import drive
drive.mount('/content/drive')


# **/!\ Check the path to the main directory**
# 
# Here I put the **IDL_group_project** directory in the root

# In[ ]:


# Check everytime
# cd to the main directory
get_ipython().run_line_magic('cd', "'/content/drive/MyDrive/IDL_group_project/Google Colab'")

data_dir = '/content/' # Google Colab
# data_dir = './' # Local Jupyter


# ### PASCAL VOC2010

# In[ ]:


# cd to the correct PASCAL VOC2010 dataset location, which should contain tar files
get_ipython().run_line_magic('cd', "'data/PASCAL VOC2010'")
get_ipython().system('ls')

# Extract the content to /content/data

get_ipython().system("mkdir '/content/data'")
get_ipython().system("mkdir '/content/data/PASCAL VOC2010'")

get_ipython().system("mkdir '/content/data/PASCAL VOC2010/VOCtrainval_03-May-2010' # Train & val datasets")
get_ipython().system("tar -xf VOCtrainval_03-May-2010.tar -C '/content/data/PASCAL VOC2010/VOCtrainval_03-May-2010' --skip-old-files # Train & val datasets")

# !mkdir '/content/PASCAL VOC2010/download' # Test dataset
# !tar -xf download.tar -C '/content/data/PASCAL VOC2010/download' --skip-old-files # Test dataset

# !mkdir '/content/PASCAL VOC2010/VOCdevkit_08-May-2010' # Optional
# !tar -xf VOCdevkit_08-May-2010.tar -C '/content/data/PASCAL VOC2010/VOCdevkit_08-May-2010' --skip-old-files # Optional

# cd back to the main working directory, all paths are referenced to this directory
get_ipython().run_line_magic('cd', '../..')
get_ipython().system('ls')


# ### PASCAL-Part

# In[ ]:


# cd to the correct PASCAL-Part dataset location, which should contain tar.gz file
get_ipython().run_line_magic('cd', "'data/PASCAL-Part'")
get_ipython().system('ls')

# Extract the content to /content/data

get_ipython().system("mkdir '/content/data'")
get_ipython().system("mkdir '/content/data/PASCAL-Part'")

get_ipython().system("mkdir '/content/data/PASCAL-Part/trainval'")
get_ipython().system("tar -xzf trainval.tar.gz -C '/content/data/PASCAL-Part/trainval' --skip-old-files")

# cd back to the main working directory, all paths are referenced to this directory
get_ipython().run_line_magic('cd', '../..')
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
# from torchvision import datasets
from torchvision import transforms
import torchvision.models as models
# from torchinfo import summary

import numpy as np

from PIL import Image

import matplotlib.pyplot as plt

get_ipython().system('pip install adversarial-robustness-toolbox')
from art.estimators.classification import PyTorchClassifier
from art.attacks.evasion import FastGradientMethod

sys.path.append('src')
from art_attack import *
from dataset import *
from model import *


# In[ ]:


print('Python version  ' + sys.version)
print('PyTorch version ' + torch.__version__)
print('Numpy version   ' + np.__version__)


# # Config

# In[ ]:


# Load previous state
load_model = False
load_model = True # <---
model_load_index = 'car' # <---

# Training parameters
train_model = False
train_model = True # <---
category = 'car' # <---
model_train_index = 'car' # <---
batch_size = 64 # <---


# In[ ]:


device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} device'.format(device))
if device == 'cuda':
    print(torch.cuda.get_device_name(0))


# # Dataset

# ### PASCAL VOC2010

# Lists of filenames and labels

# In[ ]:


label_dir = data_dir + 'data/PASCAL VOC2010/VOCtrainval_03-May-2010/VOCdevkit/VOC2010/ImageSets/Main'
categories = ['aeroplane', 'bicycle', 'boat', 'bus', 'car', 'motorbike', 'train']

image_dir = data_dir + 'data/PASCAL VOC2010/VOCtrainval_03-May-2010/VOCdevkit/VOC2010/JPEGImages'
print(label_dir)
print(image_dir)

train_image_name, train_image_label = create_file_and_label_list(label_dir, category, 'train')
val_image_name,   val_image_label   = create_file_and_label_list(label_dir, category, 'val')


# Dataloaders

# In[ ]:


transform_train_data = torchvision.transforms.Compose([
    # transforms.RandomSizedCrop(224), # Crop a square
    transforms.Resize(size=(224, 224)), # Squeeze an image into a square
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(), 
    normalize,
])

transform_val_data = torchvision.transforms.Compose([
    transforms.Resize(size=(224, 224)), # Squeeze an image into a square
    transforms.ToTensor(),
    normalize,
])

train_image_folder = VOCImageFolder(image_dir, train_image_name, train_image_label, transform=transform_train_data)
val_image_folder   = VOCImageFolder(image_dir, val_image_name,   val_image_label,   transform=transform_val_data)

train_image_dataloader = DataLoader(train_image_folder, shuffle=True,  batch_size=batch_size, num_workers=4)
val_image_dataloader   = DataLoader(val_image_folder,   shuffle=False, batch_size=16, num_workers=4)


# Visualize the resize

# In[ ]:


visualize_images(val_image_dataloader)


# Dataset sizes

# In[ ]:


print(f'No. of train images: {train_image_folder.__len__()}')
print(f'No. of val images:   {val_image_folder.__len__()}')
print()

for X, y in train_image_dataloader:
    print("Shape of X [N, C, H, W]: ", X.shape)
    print("Shape of y: ", y.shape)
    break


# ### PASCAL-Part

# # Model

# In[ ]:


# Use a pretrained model from the pytorch model zoo
vgg16_bn = models.vgg16_bn(pretrained=True) # .to(device) # vgg16/vgg16_bn
vgg16_bn.eval() # Freeze the model
# print(vgg16_bn)
    
num_classes = 2 # Detector
voc_detector = VOCDetector(vgg16_bn, num_classes).to(device)
# print(voc_detector)
summary(voc_detector, (3, 224, 224))

loss_fn = nn.CrossEntropyLoss()

epochs = 20
learning_rate = 1e-2
weight_decay = 1e-4

optimizer = torch.optim.SGD(voc_detector.parameters(), lr=learning_rate, weight_decay=weight_decay, momentum=0.9)
# optimizer = torch.optim.Adam(vgg16_bn_classifier.parameters(), lr=learning_rate)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', factor=0.1, patience=3, threshold=1e-3)
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)


# ### Load previous state

# In[ ]:


last_epoch = 0

if load_model == True:
    checkpoint = torch.load('save/model_' + model_load_index + '.pth', map_location=torch.device(device))
    voc_detector.load_state_dict(checkpoint['model_state_dict'])
    if 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    if 'epoch' in checkpoint:
        last_epoch = checkpoint['epoch']
    # Set the learning rate manually if necessary
    for param_group in optimizer.param_groups:
        param_group['lr'] = 1e-3
    print(f'Loaded model {model_load_index}!')
    print(f'Current learning rate: ' + str(optimizer.param_groups[0]['lr']))


# ### Train

# In[ ]:


if train_model == True:
    if last_epoch == 0:
        write_csv_header(model_train_index)

    for epoch in range(last_epoch + 1, last_epoch + 1 + epochs):
        start_time = time.time()
        print(f'Epoch {epoch}\n-------------------------------')
        current_lr = optimizer.param_groups[0]['lr']
        print('Learning rate: ' + str(current_lr))
        if current_lr - 1e-4 < 1e-6:
            break
        train_loss = train(train_image_dataloader, voc_detector, loss_fn, optimizer, device)
        val_accuracy, val_loss = test(val_image_dataloader, voc_detector, loss_fn, device)
        scheduler.step(val_accuracy)
        checkpoint = {
            'model_state_dict': voc_detector.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'epoch': epoch
        }
        torch.save(checkpoint, 'save/model_' + model_train_index + '.pth')
        if epoch%5 == 0:
            torch.save(checkpoint, 'save/model_' + model_train_index + '_' + str(epoch) + '.pth')
        write_csv_params(model_train_index, [epoch, val_accuracy, val_loss, train_loss, current_lr])
        print(f'Elapsed time: {(time.time() - start_time):0.1f} s')
        print()
    print('Done!')


# # Attacks

# In[ ]:


max_pixel_value = 2.6400 # Due to the transformation
min_pixel_value = -2.1179

art_detector = PyTorchClassifier(
    model=voc_detector,
    clip_values=(min_pixel_value, max_pixel_value),
    loss=loss_fn,
    optimizer=optimizer,
    input_shape=(3, 224, 224),
    nb_classes=2,
)

print(art_detector)


# ### No attack test

# In[ ]:


val_accuracy, val_loss = test_art(val_image_dataloader, art_detector, loss_fn, attack=None)
write_csv_params(model_train_index, ['eps = null', val_accuracy, val_loss])


# ### FGM attack test

# In[ ]:


# for eps_FGM in [0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5]:
for eps_FGM in [0.05]:
    print(f'eps = {eps_FGM}')
    attack_FGM = FastGradientMethod(estimator=art_detector, eps=eps_FGM)
    val_accuracy, val_loss = test_art(val_image_dataloader, art_detector, loss_fn, attack=attack_FGM)
    write_csv_params(model_train_index, ['eps = ' + str(eps_FGM), val_accuracy, val_loss])
    visualize_images(val_image_dataloader, attack=attack_FGM)


# ### Visualize the attacked images

# In[ ]:


print(f'eps = {eps_FGM}')
visualize_images(val_image_dataloader, attack=attack_FGM)

