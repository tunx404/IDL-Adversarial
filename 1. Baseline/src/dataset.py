import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Required normalization for VGG-16 (https://pytorch.org/vision/stable/models.html)
normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
     std=[0.229, 0.224, 0.225]
)

inverse_normalize = transforms.Normalize(
    mean=[-0.485/0.229, -0.456/0.224, -0.406/0.255],
     std=[1.0/0.229, 1.0/0.224, 1.0/0.255]
)

# Create a list of all files within a directory
def create_file_list(dir):
    file_name_list = os.listdir(dir)
    from tkinter import Tcl
    file_name_list = list(Tcl().call('lsort', '-dict', file_name_list)) # Sort by filename
    file_path_list = file_name_list.copy()
    for i in range(len(file_path_list)):
        file_path_list[i] = os.path.join(dir, file_name_list[i])
    return file_name_list, file_path_list

def create_file_and_label_list(label_dir, category, dataset='trainval'):
    file_path = os.path.join(label_dir, category + '_' + dataset + '.txt')
    image_name_np, image_label = np.genfromtxt(file_path, dtype='S11,i8', unpack=True)
    image_name = []
    for i in range(len(image_name_np)):
        image_name.append(image_name_np[i].decode('UTF-8') + '.jpg')
        if(image_label[i] == -1):
            image_label[i] = 0
    assert(len(image_name) == len(image_label))
    return image_name, image_label

class VOCImageFolder(Dataset):
    def __init__(self, image_dir, image_name, image_label, transform):
        self.image_name = image_name
        self.image_label = image_label
        self.image_dir = image_dir
        self.transform = transform
        self.length = len(self.image_name)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        image_path = os.path.join(self.image_dir, self.image_name[index])
        image = Image.open(image_path)
        
        image = self.transform(image)
        label = torch.as_tensor(self.image_label[index], dtype=torch.int64)
        return image, label

def visualize_images(dataloader, attack=None):
    MAX_NUM_IMAGES = 10
    num_images = 0
    for data, label in dataloader:
        if attack is not None:
            data_attack = attack.generate(data.numpy())
            data_attack = torch.from_numpy(data_attack)

            data_attack = inverse_normalize(data_attack)
            data_attack = data_attack.numpy()
            data_attack = np.transpose(data_attack, (0, 2, 3, 1)) # NxHxWxC

        data = inverse_normalize(data)
        data = data.numpy()
        data = np.transpose(data, (0, 2, 3, 1)) # NxHxWxC
        
        for i in range(min(len(data), 5)):
            if label[i] == 1:
                num_images += 1

                plt.imshow(data[i])
                plt.show()
                
                if attack is not None:
                    plt.imshow(data_attack[i])
                    plt.show()

            if num_images >= MAX_NUM_IMAGES:
                break

        if num_images >= MAX_NUM_IMAGES:
            break

def visualize_image_data(image_data, box=None): # box: (Upper left corner X, Upper left corner Y, Lower right corner X, Lower right corner Y)
    image_data = inverse_normalize(image_data)
    image_data = image_data.numpy()
    print(image_data.shape)
    image_data = np.transpose(image_data, (1, 2, 0)) # CxHxW -> HxWxC
    print(image_data.shape)
    
    fig, ax = plt.subplots()
    ax.imshow(image_data)
    if box is not None:
        anchor = (box[0], box[1]) # (Upper left corner X, Upper left corner Y)
        width = box[2] - box[0] # Lower right corner X - Upper left corner X
        height = box[3] - box[1] # Lower right corner Y - Upper left corner Y
        rect = patches.Rectangle(anchor, width, height, linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
    plt.show()